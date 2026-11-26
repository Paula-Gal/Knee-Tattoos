"""Build an animated pelvis-plus-leg Blender scene from a pelvis-leg export."""
import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_common import add_camera_and_lights, make_material, to_blender
from blender_import_knee import LEG, animate, make_animated_mesh, make_keyframes

LEG_BONES = {'right': (1, 2, 3, 4, 5), 'left': (6, 7, 8, 9, 10)}
MARKER_DISPLAY_OFFSET_M = 0.0001


def read_arguments():
    """Arguments after '--' are for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='input from pelvis_leg_export.py')
    parser.add_argument('blend', help='output .blend file')
    parser.add_argument('--marker-geometry', help='generated marker geometry NPZ')
    parser.add_argument('--visual', action='store_true', help='smooth animation for viewing')
    parser.add_argument('--force', action='store_true', help='overwrite an existing .blend')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def color_bones(bones, bone_label, bone_faces, side):
    """Assign separate materials to the pelvis, femur, tibia and foot bones."""
    femur, tibia = LEG[side]
    labels = bone_label[bone_faces[:, 0]]
    leg_bones = LEG_BONES[side]
    index = np.where(labels == femur, 0,
                     np.where(labels == tibia, 1,
                              np.where(np.isin(labels, leg_bones[2:]), 2, 3)))
    bones.data.polygons.foreach_set('material_index', index)


def _scalar_text(data, key):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f'marker geometry field {key!r} must be a scalar string')
    return str(value.item())


def load_marker_geometry(path, source):
    """Load marker arrays only when their provenance matches the source export."""
    with np.load(path, allow_pickle=False) as archive:
        marker_data = {name: archive[name] for name in archive.files}
    required = ('marker_id', 'marker_label', 'size_m', 'texture', 'grid_verts', 'normal',
                'pose_index', 'gender', 'side', 'region', 'mesh_sha256')
    missing = [key for key in required if key not in marker_data]
    if missing:
        raise ValueError(f'marker geometry is missing {", ".join(missing)}')
    for key in ('gender', 'side', 'region'):
        if _scalar_text(marker_data, key) != _scalar_text(source, key):
            raise ValueError(f'marker geometry {key} does not match the pelvis-leg export')

    marker_ids = np.asarray(marker_data['marker_id'])
    if marker_ids.ndim != 1 or marker_ids.dtype.kind not in 'iu' or np.any(marker_ids < 0):
        raise ValueError('marker IDs must be a one-dimensional array of non-negative integers')
    if len(np.unique(marker_ids)) != len(marker_ids):
        raise ValueError('marker geometry contains duplicate IDs')
    marker_count = len(marker_ids)
    labels = np.asarray(marker_data['marker_label'])
    sizes = np.asarray(marker_data['size_m'], dtype=np.float64)
    textures = np.asarray(marker_data['texture'])
    raw_families = np.asarray(marker_data.get('family', np.full(marker_count, 'custom')))
    raw_shapes = np.asarray(marker_data.get('shape', np.full(marker_count, 'texture_alpha')))
    if raw_families.dtype.kind not in 'US' or raw_shapes.dtype.kind not in 'US':
        raise ValueError('marker family and shape fields must contain strings')
    families = raw_families.astype(str)
    shapes = raw_shapes.astype(str)
    if labels.shape != (marker_count,) or sizes.shape != (marker_count,) or textures.shape != (marker_count,):
        raise ValueError('marker labels, sizes, and texture paths must match marker IDs')
    if families.shape != (marker_count,) or shapes.shape != (marker_count,):
        raise ValueError('marker families and shapes must match marker IDs')
    if np.any(np.char.str_len(np.char.strip(families)) == 0):
        raise ValueError('marker families must be non-empty strings')
    if np.any(~np.isin(shapes, ('square', 'circle', 'texture_alpha'))):
        raise ValueError('marker shapes must be square, circle, or texture_alpha')
    paper_codes = np.asarray(marker_data.get('paper_code', labels)).astype(str)
    site_regions = np.asarray(marker_data.get('site_region', np.full(marker_count, 'leg'))).astype(str)
    table_rows = np.asarray(marker_data.get('paper_table_row', np.zeros(marker_count, dtype=np.int32)))
    if paper_codes.shape != (marker_count,) or site_regions.shape != (marker_count,):
        raise ValueError('paper codes and site regions must match marker IDs')
    if table_rows.shape != (marker_count,) or table_rows.dtype.kind not in 'iu':
        raise ValueError('paper table rows must be integer values matching marker IDs')
    if np.any(~np.isin(site_regions, ('leg', 'pelvis'))):
        raise ValueError('marker site regions must be leg or pelvis')
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('marker sizes must be positive and finite')
    if any(not isinstance(value, (str, np.str_)) or not os.path.isfile(str(value)) for value in textures):
        raise ValueError('a marker texture path is missing or does not exist')

    grid_verts = np.asarray(marker_data['grid_verts'], dtype=np.float64)
    normals = np.asarray(marker_data['normal'], dtype=np.float64)
    source_verts = np.asarray(source['skin_verts'])
    if grid_verts.ndim != 5 or grid_verts.shape[0] != len(source_verts) or grid_verts.shape[1] != marker_count:
        raise ValueError('grid_verts must have shape (source poses, markers, grid, grid, 3)')
    if grid_verts.shape[2] != grid_verts.shape[3] or grid_verts.shape[2] < 3 or grid_verts.shape[4] != 3:
        raise ValueError('marker grids must be square with at least 3 samples per edge')
    if normals.shape != (len(source_verts), marker_count, 3):
        raise ValueError('marker normals must have shape (source poses, markers, 3)')
    if not np.isfinite(grid_verts).all() or not np.isfinite(normals).all():
        raise ValueError('marker geometry contains non-finite coordinates')
    if np.any(np.linalg.norm(normals, axis=-1) <= 1e-12):
        raise ValueError('marker geometry contains a zero-length normal')
    pose_index = np.asarray(marker_data['pose_index'])
    if not np.array_equal(pose_index, np.arange(len(source_verts))):
        raise ValueError('marker pose order does not match the pelvis-leg export')

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    sys.path.insert(0, os.path.join(project_root, 'marker_tracking'))
    from layout import mesh_fingerprint
    if _scalar_text(marker_data, 'mesh_sha256') != mesh_fingerprint(source):
        raise ValueError('marker geometry mesh fingerprint does not match the pelvis-leg export')
    marker_data['marker_id'] = marker_ids.astype(np.int64)
    marker_data['family'] = families
    marker_data['shape'] = shapes
    marker_data['paper_code'] = paper_codes
    marker_data['site_region'] = site_regions
    marker_data['paper_table_row'] = table_rows.astype(np.int32)
    marker_data['grid_verts'] = grid_verts
    marker_data['normal'] = normals
    return marker_data


def marker_material(marker_id, texture_path, shape='texture_alpha'):
    """Build a textured marker material with an optional UV-space silhouette mask."""
    if shape not in ('square', 'circle', 'texture_alpha'):
        raise ValueError(f'unsupported marker shape for marker {marker_id}: {shape!r}')
    try:
        image = bpy.data.images.load(texture_path, check_existing=True)
    except RuntimeError as error:
        raise ValueError(f'cannot load texture for marker {marker_id}: {texture_path}') from error
    material = bpy.data.materials.new(f'Marker {marker_id} image')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = nodes.get('Principled BSDF')
    image_node = nodes.new('ShaderNodeTexImage')
    image_node.image = image
    material.node_tree.links.new(image_node.outputs['Color'], shader.inputs['Base Color'])
    alpha = image_node.outputs['Alpha']
    if shape != 'texture_alpha':
        links = material.node_tree.links
        coordinates = nodes.new('ShaderNodeTexCoord')
        separate = nodes.new('ShaderNodeSeparateXYZ')
        links.new(coordinates.outputs['UV'], separate.inputs['Vector'])

        def math_output(operation, first, second):
            node = nodes.new('ShaderNodeMath')
            node.operation = operation
            for index, value in enumerate((first, second)):
                if hasattr(value, 'node'):
                    links.new(value, node.inputs[index])
                else:
                    node.inputs[index].default_value = value
            return node.outputs[0]

        if shape == 'circle':
            delta_u = math_output('SUBTRACT', separate.outputs['X'], 0.5)
            delta_v = math_output('SUBTRACT', separate.outputs['Y'], 0.5)
            square_u = math_output('MULTIPLY', delta_u, delta_u)
            square_v = math_output('MULTIPLY', delta_v, delta_v)
            distance_squared = math_output('ADD', square_u, square_v)
            silhouette = math_output('LESS_THAN', distance_squared, 0.250001)
        else:
            u_lower = math_output('GREATER_THAN', separate.outputs['X'], -0.000001)
            u_upper = math_output('LESS_THAN', separate.outputs['X'], 1.000001)
            v_lower = math_output('GREATER_THAN', separate.outputs['Y'], -0.000001)
            v_upper = math_output('LESS_THAN', separate.outputs['Y'], 1.000001)
            u_mask = math_output('MULTIPLY', u_lower, u_upper)
            v_mask = math_output('MULTIPLY', v_lower, v_upper)
            silhouette = math_output('MULTIPLY', u_mask, v_mask)
        links.new(silhouette, shader.inputs['Alpha'])
    else:
        links = material.node_tree.links
        links.new(alpha, shader.inputs['Alpha'])
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'DITHERED'
    material.diffuse_color = (1, 1, 1, 1)
    return material


def make_marker_mesh(marker_id, label, size_m, texture_path, vertices, grid_resolution,
                     paper_code='', site_region='leg', paper_table_row=0,
                     family='custom', shape='texture_alpha'):
    """Create a textured quad grid whose vertices follow one marker's pose arrays."""
    cells = grid_resolution - 1
    faces = np.asarray([
        (row * grid_resolution + column,
         row * grid_resolution + column + 1,
         (row + 1) * grid_resolution + column + 1,
         (row + 1) * grid_resolution + column)
        for row in range(cells) for column in range(cells)
    ], dtype=np.int32)
    obj = make_animated_mesh(
        f'Marker_{marker_id:03d}', vertices.reshape(len(vertices), grid_resolution * grid_resolution, 3),
        faces, [marker_material(marker_id, texture_path, shape)],
    )
    uv_layer = obj.data.uv_layers.new(name='MarkerUV')
    for polygon in obj.data.polygons:
        row, column = divmod(polygon.index, cells)
        u0, u1 = column / cells, (column + 1) / cells
        v0, v1 = row / cells, (row + 1) / cells
        polygon_uvs = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
        for loop_index, uv in zip(polygon.loop_indices, polygon_uvs):
            uv_layer.data[loop_index].uv = uv
    obj['marker_id'] = int(marker_id)
    obj['label'] = str(label)
    obj['family'] = str(family)
    obj['shape'] = str(shape)
    obj['paper_code'] = str(paper_code)
    obj['site_region'] = str(site_region)
    obj['paper_table_row'] = int(paper_table_row)
    obj['size_m'] = float(size_m)
    obj['texture'] = os.path.abspath(texture_path)
    obj['display_offset_m'] = MARKER_DISPLAY_OFFSET_M
    return obj


def add_marker_meshes(marker_data):
    """Create animated, display-offset marker meshes and return their expected vertices."""
    grid_verts = to_blender(marker_data['grid_verts'])
    normals = to_blender(marker_data['normal'])
    grid_verts = grid_verts + MARKER_DISPLAY_OFFSET_M * normals[:, :, None, None, :]
    grid_resolution = grid_verts.shape[2]
    marker_objects = []
    for marker_index, marker_id in enumerate(marker_data['marker_id']):
        marker_objects.append(make_marker_mesh(
            int(marker_id), marker_data['marker_label'][marker_index], marker_data['size_m'][marker_index],
            str(marker_data['texture'][marker_index]), grid_verts[:, marker_index], grid_resolution,
            marker_data['paper_code'][marker_index], marker_data['site_region'][marker_index],
            marker_data['paper_table_row'][marker_index],
            marker_data['family'][marker_index], marker_data['shape'][marker_index],
        ))
    return marker_objects, grid_verts


def shape_key_error(obj, expected):
    """Return the greatest coordinate difference across every stored pose key."""
    keys = obj.data.shape_keys.key_blocks
    if len(keys) != len(expected):
        return np.inf
    worst = 0.0
    for pose, key in enumerate(keys):
        actual = np.asarray([point.co[:] for point in key.data])
        worst = max(worst, np.abs(actual - expected[pose]).max())
    return worst


def animation_error(scene, skin, bones, skin_verts, bone_verts, keyframes):
    """Check evaluated skin and bone meshes at every animation keyframe."""
    worst_skin = 0.0
    worst_bones = 0.0
    for frame, pose in keyframes:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj, expected, is_skin in (
            (skin, skin_verts[pose], True),
            (bones, bone_verts[pose], False),
        ):
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            actual = np.asarray([vertex.co[:] for vertex in mesh.vertices])
            evaluated.to_mesh_clear()
            difference = np.abs(actual - expected).max()
            if is_skin:
                worst_skin = max(worst_skin, difference)
            else:
                worst_bones = max(worst_bones, difference)
    return worst_skin, worst_bones


def marker_animation_error(scene, marker_objects, marker_verts, keyframes):
    """Check every marker object against its generated pose grid at each keyframe."""
    worst = 0.0
    for frame, pose in keyframes:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for marker_index, obj in enumerate(marker_objects):
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                actual = np.asarray([vertex.co[:] for vertex in mesh.vertices])
            finally:
                evaluated.to_mesh_clear()
            expected = marker_verts[pose, marker_index].reshape(-1, 3)
            if actual.shape != expected.shape:
                return np.inf
            worst = max(worst, np.abs(actual - expected).max())
    return worst


def main():
    args = read_arguments()
    npz_path = os.path.abspath(args.npz)
    out_path = os.path.abspath(args.blend)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'{out_path} already exists, use --force to overwrite it')

    with np.load(npz_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    side = str(data['side'])
    if side not in LEG_BONES:
        sys.exit(f'invalid side in export: {side}')
    marker_data = None
    marker_geometry_path = None
    if args.marker_geometry:
        marker_geometry_path = os.path.abspath(args.marker_geometry)
        if not os.path.isfile(marker_geometry_path):
            sys.exit(f'marker geometry does not exist: {marker_geometry_path}')
        try:
            marker_data = load_marker_geometry(marker_geometry_path, data)
        except (OSError, ValueError) as error:
            sys.exit(f'invalid marker geometry: {error}')
    skin_verts = to_blender(data['skin_verts'])
    bone_verts = to_blender(data['bone_verts'])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    skin = make_animated_mesh(
        'Skin', skin_verts, data['skin_faces'],
        [make_material('Skin', (0.93, 0.62, 0.52), alpha=0.3)],
    )
    bones = make_animated_mesh(
        'Bones', bone_verts, data['bone_faces'],
        [make_material('Femur', (0.93, 0.84, 0.64)),
         make_material('Tibia', (0.70, 0.82, 0.95)),
         make_material('Foot bones', (0.85, 0.85, 0.82)),
         make_material('Pelvis', (0.82, 0.72, 0.52))],
    )
    color_bones(bones, data['bone_label'], data['bone_faces'], side)
    marker_objects, marker_verts = add_marker_meshes(marker_data) if marker_data is not None else ([], None)

    keyframes, interpolation = make_keyframes(len(data['knee_deg']), args.visual)
    animate(skin, keyframes, interpolation)
    animate(bones, keyframes, interpolation)
    for marker_obj in marker_objects:
        animate(marker_obj, keyframes, interpolation)
    scene.frame_start = keyframes[0][0]
    scene.frame_end = keyframes[-1][0]
    scene.render.fps = 24
    scene['timeline'] = 'visual' if args.visual else 'experimental: frame n = SKEL pose n'
    scene['gender'] = str(data['gender'])
    scene['side'] = side
    scene['region'] = str(data['region'])
    if marker_data is not None:
        scene['marker_geometry_path'] = marker_geometry_path
        scene['marker_mesh_sha256'] = str(marker_data['mesh_sha256'])
        scene['marker_ids'] = marker_data['marker_id'].tolist()
        scene['marker_display_offset_m'] = MARKER_DISPLAY_OFFSET_M

    add_camera_and_lights(scene, skin_verts.reshape(-1, 3), mirror=1 if side == 'right' else -1)

    skin_key_error = shape_key_error(skin, skin_verts)
    bone_key_error = shape_key_error(bones, bone_verts)
    skin_frame_error, bone_frame_error = animation_error(
        scene, skin, bones, skin_verts, bone_verts, keyframes,
    )
    print(f'max shape-key difference Blender vs NPZ: skin {skin_key_error:.1e} m, bones {bone_key_error:.1e} m')
    print(f'max evaluated-frame difference Blender vs NPZ: skin {skin_frame_error:.1e} m, bones {bone_frame_error:.1e} m')
    marker_key_error = 0.0
    marker_frame_error = 0.0
    if marker_objects:
        marker_key_error = max(
            shape_key_error(obj, marker_verts[:, index].reshape(len(marker_verts), -1, 3))
            for index, obj in enumerate(marker_objects)
        )
        marker_frame_error = marker_animation_error(scene, marker_objects, marker_verts, keyframes)
        print(f'max marker shape-key/evaluated-frame difference Blender vs NPZ: '
              f'{marker_key_error:.1e} m / {marker_frame_error:.1e} m')
    if max(skin_key_error, bone_key_error, skin_frame_error, bone_frame_error,
           marker_key_error, marker_frame_error) >= 1e-5:
        sys.exit('Blender meshes do not match the pelvis-leg export, .blend not saved')

    scene.frame_set(scene.frame_start)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f'saved {out_path}, {side} pelvis-leg frames {scene.frame_start}-{scene.frame_end}')


if __name__ == '__main__':
    main()