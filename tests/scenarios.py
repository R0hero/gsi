from gsi import Satellite, Receiver, Building, Ground, Blob, Drone, Tree, TerrainFeature
from .helpers import Scene

SCENARIOS: dict[str, Scene] = {}

def register(name, scene): 
    assert name not in SCENARIOS, f'duplicate scenario {name}'
    SCENARIOS[name] = scene

def signal_scene(sat_pos, *targets, plot_kw=None, **kw):
    sat = Satellite(sat_pos)
    plot_kw = plot_kw or {}

    def draw(ax):
        for t in targets:
            t.plot(ax, **plot_kw.get(t, {}))
        if not any(isinstance(t, Ground) for t in targets):
            Ground().plot(ax)
        sat.plot(ax)
    def act(ax):
        sat.draw_signal(ax, *targets, **kw)

    return Scene(draw, act)

# statics
def rx(x=45, y=2, **k):
    return Receiver((x,y), 2, **k)

def bld(x=20, h=30, w=15, **k):
    return Building((x,0), h, w, **k)

def left_bld(**k):
    return Building((10, 0), k.pop("h", 20), 12, **k)

def right_bld(**k):
    return Building((70, 0), k.pop("h", 35), 15, **k)

def tree(x=10, y=0, h=15, **k):
    return Tree((x,y), h, **k)

def drone(x=45, y=20, **k):
    return Drone((x,y), **k)

def hill(x=50, h=15, w=30, shape="Hill", **k):
    return TerrainFeature(x, h, w, shape=shape, **k)

def terrain(*features, **k):
    return Ground(groundtype="Terrain", features=list(features), **k)

def rx_on(ground, x, h=2, **k):
    return Receiver((x, ground.height_at(x) + h), 2, **k)

# <--- single receiver with environment --->
register("signal_rx_only", signal_scene((10, 80), rx()))
register("signal_rx_tree_attenuated", signal_scene((10, 80), rx(), tree(x=30, h=25)))
register("signal_rx_tree_missed", signal_scene((90, 80), rx(), tree()))
register("signal_drone_only", signal_scene((10, 80), drone()))
_helix = rx(antennatype="Helix")
register("signal_rx_helix_tilted", signal_scene((10, 80), _helix, plot_kw={_helix: {"tilt_angle": 25}}))
register("signal_rx_drone_rotated", signal_scene((10, 80), drone(rotation=15)))

# <--- reflection on building --->
register("signal_bld_rx_specular", signal_scene((80, 70), bld(), rx()))
register("signal_bld_rx_specular_left", signal_scene((20, 70), bld(x=60), rx(x=45)))
register("signal_bld_rx_specular_no_hit", signal_scene((80, 70), bld(h=10), rx()))
register("signal_bld_rx_diffuse", signal_scene((80, 70), bld(reflection="Diffuse"), rx()))
register("signal_bld_rx_diffuse_left", signal_scene((10, 70), bld(reflection="Diffuse"), rx()))
register("signal_bld_rx_diffuse_custom_rays", signal_scene((80, 70), bld(reflection="Diffuse"), rx(), num_rays=6, line_length=5))
for d in (5, 10, 20):
    register(f"signal_bld_rx_specular_dist{d}", signal_scene((80, 70), bld(), rx(35+d)))

# <--- blockage of signal from building --->
register("signal_rx_bld_clear", signal_scene((10, 60), rx(), bld(h=10)))
register("signal_rx_bld_corner", signal_scene((10, 60), rx(), bld(h=20)))
register("signal_rx_bld_blocked", signal_scene((10, 60), rx(), bld(h=30)))
register("signal_rx_bld_blocked_rx_left", signal_scene((90, 60), rx(x=10), bld(x=40, h=30)))

# <--- ground to receiver reflections --->
register("signal_ground_common_rx", signal_scene((40, 100), Ground(), rx(x=10)))
register("signal_ground_common_drone", signal_scene((10, 90), Ground(), drone(x=60, y=20)))
register("signal_ground_common_rx_far", signal_scene((60, 100), Ground(), rx(x=90)))
register("signal_ground_river_hit", signal_scene((95, 90), Ground(groundtype="River", river_x=(50, 80)), drone()))
register("signal_ground_river_miss", signal_scene((95, 90), Ground(groundtype="River", river_x=(80, 90)), drone()))
register("signal_ground_coast_hit", signal_scene((10, 90), Ground(groundtype="Coast", coast_x=40), rx(x=80)))
register("signal_ground_terrain_no_reflection", signal_scene((10, 90), Ground(groundtype="Terrain", features=[TerrainFeature(50, 15, 30, shape="Hill")]), rx(80, y=2)))
register("signal_ground_rx_below_surface", signal_scene((40, 100), Ground(height=5), rx(10, y=2)))

# <--- 2 buildings to receiver --->

for refl in ("Specular", "Diffuse"):
    r = refl.lower()

    register(f"signal_2bld_sat_left_{r}", signal_scene((10, 50), left_bld(reflection=refl), right_bld(reflection=refl), rx()))
    register(f"signal_2bld_sat_left_swapped_{r}", signal_scene((10, 50), right_bld(reflection=refl), left_bld(reflection=refl), rx()))
    register(f"signal_2bld_sat_right_{r}", signal_scene((90, 50), left_bld(reflection=refl), right_bld(reflection=refl), rx()))
    register(f"signal_2bld_sat_right_swapped_{r}", signal_scene((90, 50), right_bld(reflection=refl), left_bld(reflection=refl), rx()))

register("signal_2bld_reflection_blocked", signal_scene((5, 62), left_bld(h=55), right_bld(), rx()))
register("signal_2bld_reflection_blocked_swapped", signal_scene((5, 62), right_bld(), left_bld(h=55), rx()))

register("signal_2bld_equal_distance", signal_scene((45, 90), bld(x=10, h=20, w=15), bld(x=65, h=20, w=15), rx()))