from gsi import Satellite, Receiver, Building, Ground, Blob, Drone
from .helpers import Scene

SCENARIOS: dict[str, Scene] = {}

def register(name, scene): 
    assert name not in SCENARIOS, f'duplicate scenario {name}'
    SCENARIOS[name] = scene

def signal_scene(sat_pos, *targets, **kw):
    sat = Satellite(sat_pos)

    def draw(ax):
        Ground().plot(ax)
        sat.plot(ax)
        for t in targets:
            if not isinstance(t, Ground):
                t.plot(ax)
    def act(ax):
        sat.draw_signal(ax, *targets, **kw)

    return Scene(draw, act)