"""Interactively author barycentric marker anchors on a pelvis-leg skin mesh."""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from bpy.props import CollectionProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy_extras import view3d_utils

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER_TRACKING_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(MARKER_TRACKING_DIR)
sys.path.insert(0, MARKER_TRACKING_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from blender_common import to_blender
from layout import LEG_BONES, LayoutError, create_layout, save_layout, source_metadata

SOURCE = None
SOURCE_PATH = None
SKIN_OBJECT = None
SKIN_VERTS_BLENDER = None
SKIN_FACES = None
SKIN_LABELS = None
REFERENCE_POSE = 0


def read_arguments():
    parser = argparse.ArgumentParser(description='Pick exact marker centers on an animated pelvis-leg skin.')
    parser.add_argument('--npz', required=True, help='pelvis_leg_anim_<gender>_<side>.npz')
    parser.add_argument('--output', help='layout JSON destination; defaults beside the NPZ')
    parser.add_argument('--pose', type=int, default=0, help='reference pose index (0-45)')
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


def _material(name, color, alpha):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, alpha)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, alpha)
    bsdf.inputs['Alpha'].default_value = alpha
    if alpha < 1 and hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'DITHERED'
    return material


def _make_skin(source, pose):
    verts = to_blender(source['skin_verts'][pose])
    faces = np.asarray(source['skin_faces'], dtype=np.int32)
    labels = np.asarray(source['skin_label'], dtype=np.int32)
    mesh = bpy.data.meshes.new('MarkerPickerSkinMesh')
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new('Skin', mesh)
    bpy.context.scene.collection.objects.link(obj)
    skin_material = _material('Leg skin', (0.78, 0.40, 0.32), 0.78)
    context_material = _material('Pelvis context', (0.45, 0.52, 0.62), 0.23)
    mesh.materials.append(skin_material)
    mesh.materials.append(context_material)
    selected_bones = set(LEG_BONES[str(source['side'])])
    for polygon, face in zip(mesh.polygons, faces):
        polygon.material_index = 0 if all(int(labels[index]) in selected_bones for index in face) else 1
    obj.show_wire = False
    obj.show_all_edges = False
    return obj


def _barycentric(point, triangle):
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d00 = float(np.dot(ab, ab))
    d01 = float(np.dot(ab, ac))
    d11 = float(np.dot(ac, ac))
    d20 = float(np.dot(ap, ab))
    d21 = float(np.dot(ap, ac))
    denominator = d00 * d11 - d01 * d01
    if denominator <= 1e-16:
        raise ValueError('picked triangle is degenerate')
    second = (d11 * d20 - d01 * d21) / denominator
    third = (d00 * d21 - d01 * d20) / denominator
    weights = np.array([1.0 - second - third, second, third], dtype=np.float64)
    if not np.isfinite(weights).all() or weights.min() < -1e-5 or weights.max() > 1 + 1e-5:
        raise ValueError('ray hit is outside the picked triangle')
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def _current_marker(scene):
    if scene.marker_pick_face < 0:
        raise LayoutError('pick a valid leg-skin center first')
    texture = bpy.path.abspath(scene.marker_texture)
    if not texture or not os.path.isfile(texture):
        raise LayoutError(f'marker texture file does not exist: {scene.marker_texture}')
    return {
        'id': int(scene.marker_id),
        'label': scene.marker_label.strip(),
        'size_m': float(scene.marker_size_m),
        'rotation_deg': float(scene.marker_rotation_deg),
        'texture': texture,
        'anchor': {
            'face': int(scene.marker_pick_face),
            'bary': [float(value) for value in scene.marker_pick_bary],
        },
    }


def _entry_marker(entry):
    return {
        'id': int(entry.marker_id),
        'label': entry.label,
        'size_m': float(entry.size_m),
        'rotation_deg': float(entry.rotation_deg),
        'texture': entry.texture,
        'anchor': {'face': int(entry.face), 'bary': [float(value) for value in entry.bary]},
    }


def _preview_material(texture_path, marker_id):
    image = bpy.data.images.load(texture_path, check_existing=True)
    material = bpy.data.materials.new(f'Marker {marker_id} preview material')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    shader = nodes.new('ShaderNodeBsdfPrincipled')
    image_node = nodes.new('ShaderNodeTexImage')
    image_node.image = image
    material.node_tree.links.new(image_node.outputs['Color'], shader.inputs['Base Color'])
    if 'Alpha' in image_node.outputs and 'Alpha' in shader.inputs:
        material.node_tree.links.new(image_node.outputs['Alpha'], shader.inputs['Alpha'])
    material.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    material.diffuse_color = (1, 1, 1, 1)
    return material


def _update_preview(scene, marker):
    face = marker['anchor']['face']
    triangle_ids = SKIN_FACES[face]
    triangle = SKIN_VERTS_BLENDER[REFERENCE_POSE, triangle_ids]
    bary = np.asarray(marker['anchor']['bary'], dtype=np.float64)
    center = bary @ triangle
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-12:
        raise LayoutError('picked triangle is degenerate; cannot preview this marker')
    normal /= normal_length
    tangent = triangle[1] - triangle[0]
    tangent -= np.dot(tangent, normal) * normal
    tangent_length = np.linalg.norm(tangent)
    if tangent_length <= 1e-12:
        raise LayoutError('picked triangle has no stable tangent for preview')
    tangent /= tangent_length
    bitangent = np.cross(normal, tangent)
    angle = math.radians(marker['rotation_deg'])
    rotated_tangent = math.cos(angle) * tangent + math.sin(angle) * bitangent
    rotated_bitangent = -math.sin(angle) * tangent + math.cos(angle) * bitangent
    half_size = marker['size_m'] / 2
    offset = normal * 0.0001
    corners = [
        center - half_size * rotated_tangent - half_size * rotated_bitangent + offset,
        center + half_size * rotated_tangent - half_size * rotated_bitangent + offset,
        center + half_size * rotated_tangent + half_size * rotated_bitangent + offset,
        center - half_size * rotated_tangent + half_size * rotated_bitangent + offset,
    ]

    old = bpy.data.objects.get('MarkerPreview')
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new('MarkerPreviewMesh')
    mesh.from_pydata([point.tolist() for point in corners], [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name='MarkerUV')
    for loop_index, uv in zip(mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv_layer.data[loop_index].uv = uv
    preview = bpy.data.objects.new('MarkerPreview', mesh)
    bpy.context.scene.collection.objects.link(preview)
    mesh.materials.append(_preview_material(marker['texture'], marker['id']))
    preview['marker_id'] = marker['id']
    preview.show_in_front = True
    return preview


class MarkerEntry(bpy.types.PropertyGroup):
    marker_id: IntProperty(name='ID', min=0)
    label: StringProperty(name='Label')
    size_m: FloatProperty(name='Size (m)', min=0.0001, default=0.03)
    rotation_deg: FloatProperty(name='Rotation (deg)')
    texture: StringProperty(name='Texture', subtype='FILE_PATH')
    face: IntProperty(name='Face', min=0)
    bary: FloatVectorProperty(name='Barycentric', size=3, default=(1 / 3, 1 / 3, 1 / 3))


class MARKER_UL_entries(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f'{item.marker_id}: {item.label}')
        row.label(text=f'{item.size_m:.3f} m')


class VIEW3D_OT_pick_marker_anchor(bpy.types.Operator):
    bl_idname = 'view3d.pick_marker_anchor'
    bl_label = 'Pick Marker Center'
    bl_description = 'Click a leg-only skin face to record its exact barycentric center'

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        self._preview = bpy.data.objects.get('MarkerPreview')
        if self._preview is not None:
            self._preview.hide_set(True)
        self.report({'INFO'}, 'Click a leg-skin face; press Esc to cancel')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _restore_preview(self):
        if self._preview is not None and self._preview.name in bpy.data.objects:
            self._preview.hide_set(False)

    def modal(self, context, event):
        if event.type == 'ESC':
            self._restore_preview()
            return {'CANCELLED'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}
        window_region = next((region for region in context.area.regions if region.type == 'WINDOW'), None)
        if window_region is None:
            self.report({'ERROR'}, 'Cannot find the 3D viewport region')
            self._restore_preview()
            return {'CANCELLED'}
        coordinate = (event.mouse_x - window_region.x, event.mouse_y - window_region.y)
        if not (0 <= coordinate[0] < window_region.width and 0 <= coordinate[1] < window_region.height):
            return {'PASS_THROUGH'}
        space = context.area.spaces.active
        origin = view3d_utils.region_2d_to_origin_3d(window_region, space.region_3d, coordinate)
        direction = view3d_utils.region_2d_to_vector_3d(window_region, space.region_3d, coordinate)
        hit, location, _normal, face_index, obj, matrix = context.scene.ray_cast(
            context.evaluated_depsgraph_get(), origin, direction,
        )
        if not hit or obj != SKIN_OBJECT or face_index < 0 or face_index >= len(SKIN_FACES):
            self.report({'WARNING'}, 'Click the visible Skin mesh to pick a marker center')
            return {'RUNNING_MODAL'}
        triangle_ids = SKIN_FACES[face_index]
        side = str(SOURCE['side'])
        labels = SKIN_LABELS[triangle_ids]
        if not all(int(label) in LEG_BONES[side] for label in labels):
            self.report({'WARNING'}, 'That face includes pelvis skin; choose a face entirely on the selected leg')
            return {'RUNNING_MODAL'}
        local_point = matrix.inverted_safe() @ location
        triangle = SKIN_VERTS_BLENDER[REFERENCE_POSE, triangle_ids]
        try:
            bary = _barycentric(np.asarray(local_point, dtype=np.float64), triangle)
        except ValueError as error:
            self.report({'WARNING'}, str(error))
            return {'RUNNING_MODAL'}
        context.scene.marker_pick_face = face_index
        context.scene.marker_pick_bary = tuple(float(value) for value in bary)
        context.scene.marker_pick_valid = True
        self.report({'INFO'}, f'Picked leg face {face_index}, barycentric {bary.round(4).tolist()}')
        self._restore_preview()
        return {'FINISHED'}


class MARKER_OT_set_reference_pose(bpy.types.Operator):
    bl_idname = 'marker.set_reference_pose'
    bl_label = 'Set Reference Pose'

    def execute(self, context):
        global REFERENCE_POSE
        pose = context.scene.marker_reference_pose
        if not 0 <= pose < len(SKIN_VERTS_BLENDER):
            self.report({'ERROR'}, 'Reference pose is outside the source NPZ')
            return {'CANCELLED'}
        REFERENCE_POSE = pose
        mesh = SKIN_OBJECT.data
        mesh.vertices.foreach_set('co', SKIN_VERTS_BLENDER[pose].astype(np.float32).ravel())
        mesh.update()
        context.scene.frame_set(pose + 1)
        if context.scene.marker_pick_valid:
            try:
                _update_preview(context.scene, _current_marker(context.scene))
            except (LayoutError, RuntimeError) as error:
                self.report({'WARNING'}, str(error))
        self.report({'INFO'}, f'Reference pose set to {pose}')
        return {'FINISHED'}


class MARKER_OT_preview(bpy.types.Operator):
    bl_idname = 'marker.preview'
    bl_label = 'Preview Marker'

    def execute(self, context):
        try:
            marker = _current_marker(context.scene)
            create_layout(SOURCE, [marker], REFERENCE_POSE, layout_path=context.scene.marker_output_path)
            _update_preview(context.scene, marker)
        except (LayoutError, RuntimeError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class MARKER_OT_add(bpy.types.Operator):
    bl_idname = 'marker.add'
    bl_label = 'Add Marker to Layout'

    def execute(self, context):
        scene = context.scene
        try:
            marker = _current_marker(scene)
            markers = [_entry_marker(entry) for entry in scene.marker_entries] + [marker]
            create_layout(SOURCE, markers, REFERENCE_POSE, layout_path=scene.marker_output_path)
            _update_preview(scene, marker)
            entry = scene.marker_entries.add()
            entry.marker_id = marker['id']
            entry.label = marker['label']
            entry.size_m = marker['size_m']
            entry.rotation_deg = marker['rotation_deg']
            entry.texture = marker['texture']
            entry.face = marker['anchor']['face']
            entry.bary = marker['anchor']['bary']
            scene.marker_entry_index = len(scene.marker_entries) - 1
        except (LayoutError, RuntimeError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class MARKER_OT_remove(bpy.types.Operator):
    bl_idname = 'marker.remove'
    bl_label = 'Remove Marker'

    def execute(self, context):
        scene = context.scene
        index = scene.marker_entry_index
        if 0 <= index < len(scene.marker_entries):
            scene.marker_entries.remove(index)
            scene.marker_entry_index = min(index, len(scene.marker_entries) - 1)
        return {'FINISHED'}


class MARKER_OT_save(bpy.types.Operator):
    bl_idname = 'marker.save_layout'
    bl_label = 'Save Marker Layout'

    def execute(self, context):
        scene = context.scene
        path = bpy.path.abspath(scene.marker_output_path)
        try:
            layout = create_layout(
                SOURCE,
                [_entry_marker(entry) for entry in scene.marker_entries],
                REFERENCE_POSE,
                layout_path=path,
            )
            save_layout(path, layout, source_npz=SOURCE)
        except (LayoutError, OSError, RuntimeError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Saved {len(layout["markers"])} markers to {path}')
        return {'FINISHED'}


class VIEW3D_PT_marker_placement(bpy.types.Panel):
    bl_label = 'Skin Marker Placement'
    bl_idname = 'VIEW3D_PT_marker_placement'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Markers'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text=f'{SOURCE["gender"]} {SOURCE["side"]} leg')
        layout.prop(scene, 'marker_reference_pose')
        layout.operator('marker.set_reference_pose')
        layout.operator('view3d.pick_marker_anchor', icon='EYEDROPPER')
        if scene.marker_pick_valid:
            layout.label(text=f'Face {scene.marker_pick_face}; bary {tuple(round(v, 3) for v in scene.marker_pick_bary)}')
        box = layout.box()
        box.prop(scene, 'marker_id')
        box.prop(scene, 'marker_label')
        box.prop(scene, 'marker_size_m')
        box.prop(scene, 'marker_rotation_deg')
        box.prop(scene, 'marker_texture')
        row = box.row(align=True)
        row.operator('marker.preview', icon='IMAGE_DATA')
        row.operator('marker.add', icon='ADD')
        layout.template_list('MARKER_UL_entries', '', scene, 'marker_entries', scene, 'marker_entry_index', rows=3)
        layout.operator('marker.remove', icon='REMOVE')
        layout.prop(scene, 'marker_output_path')
        layout.operator('marker.save_layout', icon='FILE_TICK')


CLASSES = (
    MarkerEntry,
    MARKER_UL_entries,
    VIEW3D_OT_pick_marker_anchor,
    MARKER_OT_set_reference_pose,
    MARKER_OT_preview,
    MARKER_OT_add,
    MARKER_OT_remove,
    MARKER_OT_save,
    VIEW3D_PT_marker_placement,
)


def register(output_path, pose, pose_count):
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.marker_output_path = StringProperty(name='Layout JSON', default=output_path, subtype='FILE_PATH')
    bpy.types.Scene.marker_reference_pose = IntProperty(name='Reference pose', min=0, max=pose_count - 1, default=pose)
    bpy.types.Scene.marker_id = IntProperty(name='Marker ID', min=0, default=0)
    bpy.types.Scene.marker_label = StringProperty(name='Label', default='marker_0')
    bpy.types.Scene.marker_size_m = FloatProperty(name='Size (m)', min=0.0001, default=0.03)
    bpy.types.Scene.marker_rotation_deg = FloatProperty(name='Rotation (deg)', default=0.0)
    bpy.types.Scene.marker_texture = StringProperty(name='Texture image', subtype='FILE_PATH')
    bpy.types.Scene.marker_pick_face = IntProperty(name='Picked face', default=-1)
    bpy.types.Scene.marker_pick_bary = FloatVectorProperty(name='Picked barycentric', size=3, default=(1 / 3, 1 / 3, 1 / 3))
    bpy.types.Scene.marker_pick_valid = bpy.props.BoolProperty(name='Has picked center', default=False)
    bpy.types.Scene.marker_entries = CollectionProperty(type=MarkerEntry)
    bpy.types.Scene.marker_entry_index = IntProperty(default=0)


def unregister():
    for name in (
        'marker_output_path', 'marker_reference_pose', 'marker_id', 'marker_label', 'marker_size_m',
        'marker_rotation_deg', 'marker_texture', 'marker_pick_face', 'marker_pick_bary',
        'marker_pick_valid', 'marker_entries', 'marker_entry_index',
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def main():
    global SOURCE, SOURCE_PATH, SKIN_OBJECT, SKIN_VERTS_BLENDER, SKIN_FACES, SKIN_LABELS, REFERENCE_POSE
    args = read_arguments()
    SOURCE_PATH = os.path.abspath(args.npz)
    if not os.path.isfile(SOURCE_PATH):
        raise SystemExit(f'pelvis-leg NPZ does not exist: {SOURCE_PATH}')
    with np.load(SOURCE_PATH, allow_pickle=False) as archive:
        SOURCE = {key: archive[key] for key in archive.files}
    side = str(SOURCE.get('side', ''))
    if side not in LEG_BONES:
        raise SystemExit(f'invalid pelvis-leg source side: {side!r}')
    verts = np.asarray(SOURCE['skin_verts'])
    try:
        source_metadata(SOURCE, args.pose)
    except LayoutError as error:
        raise SystemExit(f'invalid pelvis-leg NPZ: {error}') from error
    SOURCE_PATH = os.path.abspath(args.npz)
    REFERENCE_POSE = args.pose
    SKIN_VERTS_BLENDER = to_blender(verts)
    SKIN_FACES = np.asarray(SOURCE['skin_faces'], dtype=np.int32)
    SKIN_LABELS = np.asarray(SOURCE['skin_label'], dtype=np.int32)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(
        os.path.dirname(SOURCE_PATH), f'marker_layout_{SOURCE["gender"]}_{side}.json',
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    SKIN_OBJECT = _make_skin(SOURCE, REFERENCE_POSE)
    scene.frame_start = 1
    scene.frame_end = len(verts)
    scene.frame_set(REFERENCE_POSE + 1)
    center = SKIN_VERTS_BLENDER[REFERENCE_POSE].mean(axis=0)
    distance = float(np.ptp(SKIN_VERTS_BLENDER[REFERENCE_POSE], axis=0).max()) * 2.5
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_location = center
                area.spaces.active.region_3d.view_distance = max(distance, 0.5)
                area.spaces.active.shading.type = 'MATERIAL'
                area.spaces.active.shading.color_type = 'MATERIAL'
    register(output_path, REFERENCE_POSE, len(verts))
    print(f'Marker picker ready: {SOURCE_PATH}; pose {REFERENCE_POSE}; save layout to {output_path}')


if __name__ == '__main__':
    main()