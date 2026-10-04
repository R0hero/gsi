from gsi import Satellite, Receiver, Building, Ground, Blob, Drone
from .helpers import Scene

SCENARIOS: dict[str, Scene] = {}

def register(name, scene): 
    assert name not in SCENARIOS, f'duplicate scenario {name}'
    SCENARIOS[name] = scene