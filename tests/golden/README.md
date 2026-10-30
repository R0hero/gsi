# Golden files for `Satellite.draw_signal`

Each scenario in `tests/scenarios.py` has two files here:

- `<name>.npz`: coordinates of every line/polygon added by the call under test
  (the assertion data; compared with `atol=1e-9`)
- `<name>.png`: the rendered scene, for humans only (never compared)

Regenerate after an *intentional* behaviour change, and review the PNG diff:

    python -m tests.update_goldens                # all scenarios
    python -m tests.update_goldens signal_rx_only # just one

Scenarios that pin a known bug (see the PR / issue list) should be regenerated in the
same `fix:` commit that fixes the bug.
