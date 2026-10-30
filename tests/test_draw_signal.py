import matplotlib.pyplot as plt
import pytest

from .helpers import assert_matches_golden, render
from .scenarios import SCENARIOS


@pytest.mark.parametrize("name", SCENARIOS)
def test_draw_signal_matches_golden(name):
    fig, ax, snap = render(SCENARIOS[name])
    plt.close(fig)
    assert_matches_golden(name, snap)
