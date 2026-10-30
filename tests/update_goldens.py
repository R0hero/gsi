import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .helpers import render, save_golden
from .scenarios import SCENARIOS

for name in sys.argv[1:] or SCENARIOS:
    fig, ax, snap = render(SCENARIOS[name])
    save_golden(name, fig, snap)
    plt.close(fig)
    print(f"wrote {name}: {len(snap)} artists")
