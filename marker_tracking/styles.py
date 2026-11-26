"""Load and resolve reusable marker appearance presets."""
import json
import math
import numbers
import os
import struct

STYLE_SCHEMA_VERSION = 1
SUPPORTED_SHAPES = ('square', 'circle', 'texture_alpha')
PAPER_SITE_SUFFIXES = (
    'ASI', 'PSI', 'THI', 'KNE', 'TIB', 'ANK', 'HEE', 'TOE', 'MED', 'MFE',
    'MTI', 'KNM', 'LTC', 'MTC', 'KNA',
)
PAPER_SITE_CODES = frozenset(
    f'{letter}{suffix}' for letter in ('L', 'R') for suffix in PAPER_SITE_SUFFIXES
)
STYLE_FIELDS = ('family', 'shape', 'texture', 'size_m', 'rotation_deg')
REQUIRED_DEFAULT_FIELDS = frozenset(('family', 'shape', 'texture', 'size_m'))
DEFAULT_STYLE_PRESET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'presets', 'mikneesota_marker_styles.json',
)


class StylePresetError(ValueError):
    """Raised when a marker style preset is malformed or references invalid assets."""


def _png_has_transparency(image_file):
    while True:
        header = image_file.read(8)
        if len(header) != 8:
            return False
        chunk_length = struct.unpack('>I', header[:4])[0]
        chunk_type = header[4:]
        if chunk_type == b'tRNS':
            return True
        if chunk_type in (b'IDAT', b'IEND'):
            return False
        image_file.seek(chunk_length + 4, os.SEEK_CUR)


def _validate_texture_file(path, prefix, shape):
    try:
        with open(path, 'rb') as image_file:
            signature = image_file.read(8)
            image_file.seek(0, os.SEEK_END)
            file_size = image_file.tell()
            if file_size == 0:
                raise StylePresetError(f'{prefix}.texture image is empty: {path}')
            if os.path.splitext(path)[1].lower() == '.png' and signature != b'\x89PNG\r\n\x1a\n':
                raise StylePresetError(f'{prefix}.texture is not a valid PNG image: {path}')
            if signature == b'\x89PNG\r\n\x1a\n':
                image_file.seek(8)
                header = image_file.read(25)
                if len(header) != 25:
                    raise StylePresetError(f'{prefix}.texture is a truncated PNG image: {path}')
                chunk_length, chunk_type = struct.unpack('>I4s', header[:8])
                width, height, _depth, color_type, _compression, _filter, _interlace = struct.unpack(
                    '>IIBBBBB', header[8:21],
                )
                if chunk_length != 13 or chunk_type != b'IHDR' or width == 0 or height == 0:
                    raise StylePresetError(f'{prefix}.texture has an invalid PNG header: {path}')
                if shape == 'texture_alpha' and color_type not in (4, 6) and not _png_has_transparency(image_file):
                    raise StylePresetError(f'{prefix}.texture must contain alpha for texture_alpha shape: {path}')
            elif signature.startswith(b'\xff\xd8') or os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
                if signature[:2] != b'\xff\xd8' or file_size < 4:
                    raise StylePresetError(f'{prefix}.texture is not a valid JPEG image: {path}')
                image_file.seek(-2, os.SEEK_END)
                if image_file.read(2) != b'\xff\xd9':
                    raise StylePresetError(f'{prefix}.texture is a truncated JPEG image: {path}')
                if shape == 'texture_alpha':
                    raise StylePresetError(f'{prefix}.texture must contain alpha for texture_alpha shape: {path}')
    except OSError as error:
        raise StylePresetError(f'{prefix}.texture cannot be read: {path}: {error}') from error


def _validate_style_fields(value, prefix, preset_dir, defaults=False, fallback_shape='texture_alpha'):
    if not isinstance(value, dict):
        raise StylePresetError(f'{prefix} must be an object')
    unknown = sorted(set(value) - set(STYLE_FIELDS))
    if unknown:
        raise StylePresetError(f'{prefix} has unsupported field(s): {", ".join(unknown)}')
    missing = sorted(REQUIRED_DEFAULT_FIELDS - set(value)) if defaults else []
    if missing:
        raise StylePresetError(f'{prefix} is missing required field(s): {", ".join(missing)}')
    if not value:
        raise StylePresetError(f'{prefix} must define at least one style field')

    normalized = {}
    if 'family' in value:
        family = value['family']
        if not isinstance(family, str) or not family.strip():
            raise StylePresetError(f'{prefix}.family must be a non-empty string')
        normalized['family'] = family.strip()
    if 'shape' in value:
        shape = value['shape']
        if not isinstance(shape, str) or shape not in SUPPORTED_SHAPES:
            raise StylePresetError(
                f'{prefix}.shape must be one of {", ".join(SUPPORTED_SHAPES)}'
            )
        normalized['shape'] = shape
    if 'texture' in value:
        texture = value['texture']
        if not isinstance(texture, str) or not texture.strip() or '\0' in texture:
            raise StylePresetError(f'{prefix}.texture must be a non-empty file path')
        texture = os.path.expanduser(texture.strip())
        if not os.path.isabs(texture):
            texture = os.path.join(preset_dir, texture)
        texture = os.path.abspath(texture)
        if not os.path.isfile(texture):
            raise StylePresetError(f'{prefix}.texture file does not exist: {value["texture"]}')
        _validate_texture_file(texture, prefix, value.get('shape', fallback_shape))
        normalized['texture'] = texture
    if 'size_m' in value:
        size = value['size_m']
        if isinstance(size, bool) or not isinstance(size, numbers.Real) or not math.isfinite(float(size)):
            raise StylePresetError(f'{prefix}.size_m must be a finite number')
        if float(size) <= 0:
            raise StylePresetError(f'{prefix}.size_m must be positive')
        normalized['size_m'] = float(size)
    if 'rotation_deg' in value:
        rotation = value['rotation_deg']
        if isinstance(rotation, bool) or not isinstance(rotation, numbers.Real) or not math.isfinite(float(rotation)):
            raise StylePresetError(f'{prefix}.rotation_deg must be a finite number')
        normalized['rotation_deg'] = float(rotation)
    return normalized


def load_style_preset(path):
    """Read and validate a JSON preset, resolving its image paths relative to the file."""
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        with open(path, encoding='utf-8') as preset_file:
            preset = json.load(preset_file)
    except (OSError, json.JSONDecodeError) as error:
        raise StylePresetError(f'cannot read style preset {path}: {error}') from error
    if not isinstance(preset, dict):
        raise StylePresetError('style preset root must be an object')
    unknown = sorted(set(preset) - {'schema_version', 'defaults', 'sites'})
    if unknown:
        raise StylePresetError(f'style preset has unsupported field(s): {", ".join(unknown)}')
    version = preset.get('schema_version')
    if isinstance(version, bool) or version != STYLE_SCHEMA_VERSION:
        raise StylePresetError(
            f'unsupported style schema_version {version!r}; expected {STYLE_SCHEMA_VERSION}'
        )
    if 'defaults' not in preset or 'sites' not in preset:
        raise StylePresetError('style preset must contain defaults and sites objects')

    preset_dir = os.path.dirname(path)
    defaults = _validate_style_fields(preset['defaults'], 'defaults', preset_dir, defaults=True)
    if 'rotation_deg' not in defaults:
        defaults['rotation_deg'] = 0.0
    sites = preset['sites']
    if not isinstance(sites, dict):
        raise StylePresetError('sites must be an object keyed by MiKneeSoTA paper site code')
    normalized_sites = {}
    for code, style in sites.items():
        if not isinstance(code, str) or code not in PAPER_SITE_CODES:
            raise StylePresetError(f'unknown MiKneeSoTA paper site code: {code!r}')
        normalized_sites[code] = _validate_style_fields(
            style, f'sites.{code}', preset_dir, fallback_shape=defaults['shape'],
        )
    return {
        'schema_version': STYLE_SCHEMA_VERSION,
        'preset_path': path,
        'defaults': defaults,
        'sites': normalized_sites,
    }


def resolve_site_style(style_preset, paper_code):
    """Resolve one site's complete style without changing its layout anchor."""
    if paper_code not in PAPER_SITE_CODES:
        raise StylePresetError(f'unknown MiKneeSoTA paper site code: {paper_code!r}')
    if not isinstance(style_preset, dict) or 'defaults' not in style_preset or 'sites' not in style_preset:
        raise StylePresetError('style_preset must be loaded with load_style_preset()')
    style = dict(style_preset['defaults'])
    style.update(style_preset['sites'].get(paper_code, {}))
    return style


def style_preset_metadata(style_preset):
    """Return JSON-ready resolved preset provenance for a generated layout."""
    return {
        'schema_version': style_preset['schema_version'],
        'preset_path': style_preset['preset_path'],
        'defaults': dict(style_preset['defaults']),
        'sites': {code: dict(style) for code, style in style_preset['sites'].items()},
    }