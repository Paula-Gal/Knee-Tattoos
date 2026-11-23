# Marker tracking

Video of a body with markers in, marker positions (camera frame, metres) out.
Work in progress.

## Layout

- `marker_tracking/`: the package, one step per file (`io`, `detect`, `lift`, `table`, `evaluate`, `run`)
- `blender/export_take.py`: renders a take from the Blender knee scene (virtual ground truth)
- `tests/`: pytest tests

## Tests

From this folder:

```bash
pytest
```
