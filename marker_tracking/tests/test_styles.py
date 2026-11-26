import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_layout import write_disc_texture
from styles import StylePresetError, load_style_preset, resolve_site_style


class TestMarkerStylePresets(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.default_texture = self.directory / 'default.png'
        write_disc_texture(self.default_texture)
        self.site_texture = self.directory / 'site.png'
        self.site_texture.write_bytes(self.default_texture.read_bytes())

    def write_preset(self, defaults=None, sites=None):
        preset = {
            'schema_version': 1,
            'defaults': defaults or {
                'family': 'aruco', 'shape': 'square', 'texture': 'default.png', 'size_m': 0.04,
            },
            'sites': sites or {},
        }
        path = self.directory / 'styles.json'
        path.write_text(json.dumps(preset), encoding='utf-8')
        return path

    def test_resolves_relative_textures_and_partial_site_overrides(self):
        preset_path = self.write_preset(sites={
            'LASI': {
                'family': 'stag_hd11', 'shape': 'texture_alpha',
                'texture': 'site.png', 'size_m': 0.025, 'rotation_deg': -12,
            },
        })

        preset = load_style_preset(preset_path)
        default = resolve_site_style(preset, 'RASI')
        override = resolve_site_style(preset, 'LASI')

        self.assertEqual(default['family'], 'aruco')
        self.assertEqual(default['shape'], 'square')
        self.assertEqual(default['texture'], str(self.default_texture.resolve()))
        self.assertEqual(default['rotation_deg'], 0.0)
        self.assertEqual(override['family'], 'stag_hd11')
        self.assertEqual(override['texture'], str(self.site_texture.resolve()))
        self.assertEqual(override['size_m'], 0.025)
        self.assertEqual(override['rotation_deg'], -12.0)

    def test_rejects_unknown_sites_unsupported_shapes_and_missing_textures(self):
        cases = (
            ({'family': 'custom', 'shape': 'hexagon', 'texture': 'default.png', 'size_m': 0.02}, {}, 'shape'),
            ({'family': 'custom', 'shape': 'square', 'texture': 'missing.png', 'size_m': 0.02}, {}, 'texture'),
            ({'family': 'custom', 'shape': 'square', 'texture': 'default.png', 'size_m': 0.02},
             {'UNKNOWN': {'size_m': 0.02}}, 'unknown'),
        )
        for defaults, sites, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(StylePresetError, message):
                    load_style_preset(self.write_preset(defaults, sites))

    def test_rejects_invalid_png_and_texture_alpha_without_alpha_channel(self):
        invalid_png = self.directory / 'invalid.png'
        invalid_png.write_bytes(b'not a png')
        with self.assertRaisesRegex(StylePresetError, 'PNG'):
            load_style_preset(self.write_preset({
                'family': 'custom', 'shape': 'square', 'texture': 'invalid.png', 'size_m': 0.02,
            }))

        opaque_png = self.directory / 'opaque.png'
        opaque_png.write_bytes(bytes.fromhex(
            '89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de'
            '0000000b49444154789c636000020000050001a5f645400000000049454e44ae426082'
        ))
        with self.assertRaisesRegex(StylePresetError, 'alpha'):
            load_style_preset(self.write_preset({
                'family': 'custom', 'shape': 'texture_alpha', 'texture': 'opaque.png', 'size_m': 0.02,
            }))

    def test_rejects_invalid_dimensions_and_non_finite_rotations(self):
        for size in (0, -0.01, float('nan'), float('inf')):
            with self.subTest(size=size):
                defaults = {
                    'family': 'custom', 'shape': 'circle', 'texture': 'default.png', 'size_m': size,
                }
                with self.assertRaisesRegex(StylePresetError, 'size_m'):
                    load_style_preset(self.write_preset(defaults))
        with self.assertRaisesRegex(StylePresetError, 'rotation_deg'):
            load_style_preset(self.write_preset(
                sites={'LASI': {'rotation_deg': float('nan')}},
            ))

    def test_rejects_malformed_preset_version_and_fields(self):
        path = self.write_preset()
        value = json.loads(path.read_text(encoding='utf-8'))
        value['schema_version'] = 2
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaisesRegex(StylePresetError, 'schema_version'):
            load_style_preset(path)

        value['schema_version'] = 1
        value['defaults']['unsupported'] = True
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaisesRegex(StylePresetError, 'unsupported field'):
            load_style_preset(path)


if __name__ == '__main__':
    unittest.main()