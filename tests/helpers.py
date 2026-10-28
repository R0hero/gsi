import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 
from dataclasses import dataclass
from typing import Callable

GOLDEN_DIR = Path(__file__).parent / "golden"

@dataclass
class Scene:
    draw: Callable
    act: Callable
    xlim: tuple = (0, 100)
    ylim: tuple = (-10, 110)

def render(scene: Scene):
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
    ax.set_xlim(scene.xlim)
    ax.set_ylim(scene.ylim)
    ax.set_aspect('equal')
    ax.axis('off')

    scene.draw(ax)
    n_lines, n_patches = len(ax.lines), len(ax.patches)
    scene.act(ax)

    snap = {}
    for i, line in enumerate(ax.lines[n_lines:]):
        snap[f'line{i}'] = np.column_stack(line.get_data())
        snap[f'line{i}_style'] = np.array([str(line.get_color()), str(line.get_linestyle()), str(line.get_linewidth()), str(line.get_zorder())])
    for i, patch in enumerate(ax.patches[n_patches:]):
        snap[f'patch{i}'] = patch.get_xy()
    return fig, ax, snap

def save_golden(name: str, fig, snap) -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **snap)
    fig.savefig(GOLDEN_DIR / f"{name}.png", dpi=100)

def assert_matches_golden(name: str, snap) -> None:
    golden = np.load(GOLDEN_DIR / f"{name}.npz")
    assert set(snap) == set(golden.files)
    for key in snap:
        if snap[key].dtype.kind == "U":
            np.testing.assert_array_equal(snap[key], golden[key], err_msg=key)
        else:
            np.testing.assert_allclose(snap[key], golden[key], atol=1e-9, err_msg=key)