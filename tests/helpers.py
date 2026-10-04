import numpy as np
import matplotlib.pyplot
from pathlib import Path
import os 

GOLDEN_DIR = Path(f'tests{os.sep}golden{os.sep}')

def snapshot(ax) -> dict[str, np.ndarray]:
    arrays = {}
    for i, line in enumerate(ax.lines):
        arrays[f'line{i}'] = np.column_stack(line.get_data())
    for i, patch in enumerate(ax.patches):
        arrays[f'patch{i}'] = patch.get_xy()
    return arrays

def save_golden(name: str, fig, ax) -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **snapshot(ax))
    fig.savefig(GOLDEN_DIR / f"{name}.png", dpi=100)

def assert_matches_golden(name: str, ax) -> None:
    got = snapshot(ax)
    golden = np.load(GOLDEN_DIR / f"{name}.npz")
    assert set(got) == set(golden.files)
    for key in got:
        np.testing.assert_allclose(got[key], golden[key], atol=1e-9)