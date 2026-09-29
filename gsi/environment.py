from .enums import ReflectionType, GroundType, TreeType, TerrainShape

import matplotlib.patches as mpatches
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.affinity import scale
import numpy as np
       
class Building:
    def __init__(self, position : tuple, height : float, width : float,
                 plot_windows : bool = True, plot_door : bool = True,
                 reflection : str = 'Specular'):
        self.height = height
        self.width = width
        self.position = position

        # optional parameters
        self.plot_windows = plot_windows
        self.plot_door = plot_door
        
        # validate reflection type
        if reflection not in ReflectionType._value2member_map_:
            valid_types = ', '.join([f"'{r.value}'" for r in ReflectionType])
            raise ValueError(f"Unknown reflection type '{reflection}'. Use {valid_types}.")
        
        self.reflection = ReflectionType._value2member_map_[reflection]

    def plot(self, ax,
            door_height : float = 5, door_width : float = 3, 
            min_window_spacing : float = 1.4, window_height : float = 2.5, window_width : float = 2.5,
            alpha:float = 1):
        """Plots the building on the given position"""
        # make building outline
        building_outline = mpatches.Rectangle(self.position, self.width, self.height, facecolor='lightgray', edgecolor='black', zorder=2, alpha=alpha)

        if self.plot_door:
            # calculate door position to be in middle of building
            door_position = (self.position[0]+self.width/2-door_width/2, self.position[1])
            # create door patch
            door = mpatches.Rectangle(door_position, door_width, door_height, facecolor='gray', edgecolor='black', zorder=3, alpha=alpha)

        if self.plot_windows:
            # calculate available space for windows above the door
            available_height = self.height - door_height
            available_width = self.width

            # calculate amount of windows that fit in the available space
            num_windows_height = int((available_height - min_window_spacing) // (window_height + min_window_spacing))
            num_windows_width = int((available_width - min_window_spacing) // (window_width + min_window_spacing))

            # calculate total spacing needed to distribute evenly
            total_vertical_spacing = available_height - num_windows_height * window_height
            total_horizontal_spacing = available_width - num_windows_width * window_width

            # calculate the spacing between windows and the margins
            if num_windows_height > 1:
                adjusted_window_spacing_height = total_vertical_spacing / (num_windows_height + 1)
            else:
                adjusted_window_spacing_height = total_vertical_spacing / 2

            if num_windows_width > 1:
                adjusted_window_spacing_width = total_horizontal_spacing / (num_windows_width + 1)
            else:
                adjusted_window_spacing_width = total_horizontal_spacing / 2

            # initialize list for windows
            windows = []
            # loop through available amount of windows
            for i in range(num_windows_height):
                for j in range(num_windows_width):
                    # calculate position based on iterations
                    window_position = (self.position[0] + adjusted_window_spacing_width + j * (window_width + adjusted_window_spacing_width), self.position[1] + door_height + adjusted_window_spacing_height + i * (window_height + adjusted_window_spacing_height))
                    window = mpatches.Rectangle(window_position, window_width, window_height, facecolor='lightgray', edgecolor='black', zorder=3, alpha=alpha)
                    # store for later plotting
                    windows.append(window)

                    # calculate middle of windows
                    middle_window_left = (window_position[0], window_position[1] + window_height/2)
                    middle_window_right = (window_position[0] + window_width, window_position[1] + window_height/2)
                    middle_window_bottom = (window_position[0] + window_width/2, window_position[1])
                    middle_window_top = (window_position[0] + window_width/2, window_position[1] + window_height)

                    # plot lines on windows
                    ax.plot([middle_window_left[0], middle_window_right[0]], [middle_window_left[1], middle_window_right[1]], color='black', linewidth=0.3, zorder=4, alpha=alpha)
                    ax.plot([middle_window_bottom[0], middle_window_top[0]], [middle_window_bottom[1], middle_window_top[1]], color='black', linewidth=0.3, zorder=4, alpha=alpha)

        # add components to plot
        ax.add_patch(building_outline)
        if self.plot_door:
            ax.add_patch(door)
        if self.plot_windows:
            for window in windows:
                ax.add_patch(window)

class TerrainFeature:
    def __init__(self, position: float, height: float, width: float, 
                 shape: str = 'Hill', roughness: float = None, seed: int = 0):
        if shape not in TerrainShape._value2member_map_:
            valid_types = ', '.join([f"'{r.value}'" for r in TerrainShape])
            raise ValueError(f"Unknown terrain shape type '{shape}'. Use {valid_types}.")
        if height <= 0:
            raise ValueError("height must be positive. Use 'Valley' to carve downward")
        if width <= 0:
            raise ValueError("width must be positive.")
        
        self.position = position
        self.height = height
        self.width = width
        self.shape = TerrainShape._value2member_map_[shape]
        self.seed = seed

        if roughness is None:
            roughness = 0.12 if self.shape == TerrainShape.MOUNTAIN else 0.0
        if not 0 <= roughness < 1:
            raise ValueError("roughness must be within [0, 1].")
        self.roughness = roughness

        rng = np.random.default_rng(seed)
        self._noise_cycles = np.array([3.0, 7.0, 13.0])
        self._noise_amps = np.array([1.0, 0.5, 0.25])
        self._noise_phases = rng.uniform(0, 2 * np.pi, len(self._noise_cycles))

    def profile(self, x):
        """returns the height added to the ground by this feature at x"""
        x = np.asarray(x, dtype=float)

        u = np.abs(x - self.position) / (self.width / 2)
        inside = u < 1
        z = np.zeros_like(u)

        if self.shape == TerrainShape.MOUNTAIN:
            z[inside] = (1 - u[inside])**1.5
        else:
            z[inside] = 0.5 * (1 + np.cos(np.pi * u[inside]))

        if self.roughness > 0:
            t = (x - self.position) / self.width
            noise = sum(a * np.sin(2 * np.pi * c * t + ph) for a, c, ph in zip(self._noise_amps, self._noise_cycles, self._noise_phases))
            noise /= self._noise_amps.sum()

            z = z * np.clip(1 + self.roughness * noise, 0, None)
        sign = -1 if self.shape == TerrainShape.VALLEY else 1
        return sign * self.height * z

class Ground:
    def __init__(self, height : float = 0, groundtype : str = 'Common', 
                 river_x: tuple = None, coast_x: float = None, sea_side: str = 'right',
                 depth: float = 6, bank_slope: float = 0.25, water_offset: float = 1.5,
                 features: list = None, terrain_resolution: int = 800):
        self.height = height

        # validate reflection type
        if groundtype not in GroundType._value2member_map_:
            valid_types = ', '.join([f"'{r.value}'" for r in GroundType])
            raise ValueError(f"Unknown ground type type '{groundtype}'. Use {valid_types}.")
        
        self.groundtype = GroundType._value2member_map_[groundtype]

        self.depth = depth
        self.water_offset = water_offset
        self.water_level = height - water_offset

        # profile parameters only relevant if not Common type
        self.river_x = None
        self.coast_x = None
        self.sea_side = sea_side.lower()
        self._bank_dx = 0
        self._water_span = None

        self.features = []
        self.terrain_resolution = terrain_resolution
        if features and self.groundtype == GroundType.COMMON:
            raise ValueError("features can only be given for 'Terrain', 'River', or 'Coast'.")
        for feature in (features or []):
            self.add_feature(feature)

        if self.groundtype == GroundType.RIVER:
            if river_x is None or len(river_x) != 2 or river_x[0] >= river_x[1]:
                raise ValueError("River ground requires river_x=(x_left, x_right) with x_left < x_right.")
            if not 0 <= bank_slope <= 0.5:
                raise ValueError("For a river, bank_slope is a fraction of the river width per bank and must be in [0, 0.5].")
            self._validate_depths()
            self.river_x = tuple(river_x)
            self._bank_dx = bank_slope * (river_x[1] - river_x[0])

            # calculate where water surface meets banks
            shore = self._bank_dx * water_offset / depth
            self._water_span = (river_x[0] + shore, river_x[1] - shore)

        elif self.groundtype == GroundType.COAST:
            if coast_x is None:
                raise ValueError("Coast ground requires a value for coast_x.")
            if sea_side not in ('right', 'left'):
                raise ValueError("sea_side must be 'left' or 'right'.")
            if bank_slope < 0:
                raise ValueError("For a coast, bank_slope is a width in x-units, and must be positive.")
            self._validate_depths()
            self.coast_x = coast_x
            self._bank_dx = bank_slope
            shore = bank_slope * water_offset / depth
            if sea_side == 'right':
                self._water_span = (coast_x + shore, np.inf)
            else:
                self._water_span = (-np.inf, coast_x - shore)

    @staticmethod
    def _clip_profile(points, x_min, x_max):
        """clips a left-to-right polyline based on x_min to x_max, interpolating the end points"""
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        new_xs = np.concatenate(([x_min], xs[(xs > x_min) & (xs < x_max)], [x_max]))
        return list(zip(new_xs, np.interp(new_xs, xs, ys)))

    @property
    def reflective_surface(self):
        if self.groundtype == GroundType.COMMON:
            return (self.height, -np.inf, np.inf)
        if self.groundtype == GroundType.TERRAIN:
            return None
        return (self.water_level, *self._water_span)

    def _validate_depths(self):
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if not 0 <= self.water_offset < self.depth:
            raise ValueError("water_offset must be >= 0 and smaller than depth.")

    def _terrain_height(self, x):
        """base height plus contribution of placed features"""
        x = np.asarray(x, dtype=float)
        return self.height + sum((f.profile(x) for f in self.features), np.zeros_like(x))

    def _terrain_profile(self, x_min, x_max):
        """function to derive vertices of the land surface from x_min to x_max"""
        low, high = -1e9, 1e9
        if self.groundtype == GroundType.COMMON:
            return [(x_min, self.height), (x_max, self.height)]
        if self.groundtype in (GroundType.TERRAIN, GroundType.RIVER, GroundType.COAST):
            centres = [f.position for f in self.features if x_min < f.position < x_max]
            xs = np.unique(np.concatenate((np.linspace(x_min, x_max, self.terrain_resolution), centres)))
            return list(zip(xs, self._terrain_height(xs)))
        if self.groundtype == GroundType.RIVER:
            x0, x1 = self.river_x
            points = [(low, self.height), (x0, self.height), (x0 + self._bank_dx, self.height - self.depth), (x1 - self._bank_dx, self.height - self.depth), (x1, self.height), (high, self.height)]
        elif self.sea_side == 'right':
            points =  [(low, self.height), (self.coast_x, self.height), (self.coast_x + self._bank_dx, self.height - self.depth), (high, self.height - self.depth)]
        else: 
            points = [(low, self.height - self.depth), (self.coast_x - self._bank_dx, self.height - self.depth), (self.coast_x, self.height), (high, self.height)]
        return self._clip_profile(points, x_min, x_max)
     
    def _surface_profile(self, x_min, x_max):
        """function to dervice vertices of the surface profile from x_min to x_max"""
        low, high = -1e9, 1e9
        if self.groundtype in [GroundType.COMMON, GroundType.TERRAIN]:
            return self._terrain_profile(x_min, x_max)
        s0, s1 = self._water_span
        if self.groundtype == GroundType.RIVER:
            points = [(low, self.height), (self.river_x[0], self.height), (s0, self.water_level), (s1, self.water_level), (self.river_x[1], self.height), (high, self.height)]
        elif self.sea_side == 'right':
            points = [(low, self.height), (self.coast_x, self.height), (s0, self.water_level), (high, self.water_level)]
        else:
            points = [(low, self.water_level), (s1, self.water_level), (self.coast_x, self.height), (high, self.height)]
        return self._clip_profile(points, x_min, x_max)

    def _water_polygon(self, x_min, x_max):
        """function to derive vertices of water surface from x_min to x_max"""
        s0, s1 = self._water_span
        if self.groundtype == GroundType.RIVER:
            x0, x1 = self.river_x
            return [(s0, self.water_level), (x0 + self._bank_dx, self.height - self.depth), (x1 - self._bank_dx, self.height - self.depth), (s1, self.water_level)]
        elif self.sea_side == 'right':
            return [(s0, self.water_level), (self.coast_x + self._bank_dx, self.height - self.depth), (x_max, self.height - self.depth), (x_max, self.water_level)]
        return [(x_min, self.water_level), (x_min, self.height - self.depth), (self.coast_x - self._bank_dx, self.height - self.depth), (s1, self.water_level)]

    def height_at(self, x):
        """returns height of the land surface at x"""
        x = np.asarray(x, dtype=float)
        if self.groundtype == GroundType.COMMON:
            y = np.full_like(x, self.height)
        elif self.groundtype == GroundType.TERRAIN:
            y = self._terrain_height(x)
        else:
            profile = self._terrain_profile(np.min(x) - 1, np.max(x) + 1)
            y = np.interp(x, [p[0] for p in profile], [p[1] for p in profile])
        return float(y) if y.ndim == 0 else y

    def add_feature(self, feature: 'TerrainFeature'):
        """places a terrain feature on a Terrain ground. Returns the Ground, in order to be able to chain calls."""
        if self.groundtype == GroundType.COMMON:
            raise ValueError("Terrain features can not be added to 'Common' ground type.")
        if not isinstance(feature, TerrainFeature):
            raise TypeError("Only TerrainFeature objects can be added to the Terrain.")
        self.features.append(feature)
        return self
        
    def plot(self, ax, 
             x_limits: tuple = None,
             color : str = 'black', linewidth : float = 1, linestyle : str = '-',
             water_color: str = 'lightblue', water_alpha: float = 0.6, 
             surface_color: str = 'lightblue', surface_linewidth: float = 1, surface_linestyle: str = '-',
             n_waves: int = 0, wave_length: float = 3, wave_height: float = 0.4, wave_cycles: float = 0.3, n_rows: int = 1,
             terrain_fill_color: str = 'lightgray', terrain_fill_alpha: float = 0.5, terrain_fill_depth: float = 5):
        """Draws the ground"""
        if self.groundtype == GroundType.COMMON:
            # plot the ground as a simple line
            ax.axhline(y=self.height,color=color, linewidth=linewidth, linestyle=linestyle)
            return

        if x_limits is None:
            x_limits = ax.get_xlim()
        x_min, x_max = min(x_limits), max(x_limits)

        terrain = self._terrain_profile(x_min, x_max)

        if self.groundtype == GroundType.TERRAIN:
            xs, ys = zip(*terrain)

            if terrain_fill_color is not None:
                ax.fill_between(xs, self.height - terrain_fill_depth - max(0, self.height - min(ys)), ys, 
                                color=terrain_fill_color, alpha=terrain_fill_alpha, linewidth=0, zorder=0.6)
                ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, zorder=2)
                return 

        water = self._water_polygon(x_min, x_max)

        # water surface
        ax.fill(*zip(*water), color=water_color, zorder=0.6, alpha=water_alpha, linewidth=0)

        # terrain surface
        ax.plot(*zip(*terrain), color=color, linewidth=linewidth, linestyle=linestyle, zorder=2)

        # water surface line
        s0, s1 = max(self._water_span[0], x_min), min(self._water_span[1], x_max)

        surface_x = [s0]
        surface_y = [self.water_level]

        # draw waves, if prompted
        if n_waves > 0:
            k = 2 * np.pi * wave_cycles / wave_length
            amplitude = min(wave_height, 0.9 / k)

            theta = np.pi + np.linspace(0, 2 * np.pi * wave_cycles, 30 * max(int(np.ceil(wave_cycles)), 1))
            mark_x = theta / k - amplitude * np.sin(theta)
            mark_x -= mark_x.mean()
            mark_z = amplitude * np.cos(theta)

            depth_available = self.water_level - (self.height - self.depth)
            row_spacing = min(2.5 * wave_height, depth_available / n_rows)
            spacing = (s1 - s0) / n_waves

            for row in range(n_rows):
                shift = 0.5 * spacing if row % 2 else 0
                row_y = self.water_level + amplitude - row * row_spacing
                for i in range(n_waves):
                    xc = s0 + (i + 0.5) * spacing + shift
                    x_start, x_end = xc + mark_x[0], xc + mark_x[-1]
                    if x_start < s0 or x_end > s1:
                        continue
                    if row == 0:
                        # skip if it overlaps the previous one
                        if x_start < surface_x[-1]:
                            continue
                        surface_x += list(xc + mark_x)
                        surface_y += list(row_y + mark_z)
                    else:
                        ax.plot(xc + mark_x, row_y + mark_z, color=surface_color, linewidth=surface_linewidth * 0.8, zorder=2, solid_capstyle='round')

        surface_x.append(s1)
        surface_y.append(self.water_level)

        ax.plot(surface_x, surface_y, color=surface_color, 
                linewidth=surface_linewidth, linestyle=surface_linestyle, zorder=2)
        ax.fill_between(surface_x, self.water_level, surface_y, color=water_color, alpha=water_alpha, linewidth=0, zorder=0.6)

class Tree:
    def __init__(self, position: tuple, height: float, 
                 trunk_width: float = None, canopy_width: float = None, 
                 treetype: str = 'Deciduous'):
        self.position = position
        self.height = height

        self.trunk_width = trunk_width if trunk_width != None else 0.08*height
        self.canopy_width = canopy_width if canopy_width != None else 0.75*height

        if treetype not in TreeType._value2member_map_:
            valid_types = ', '.join([f"'{r.value}'" for r in TreeType])
            raise ValueError(f"Unknown tree type '{treetype}'. Use {valid_types}.")

        self.treetype = TreeType._value2member_map_[treetype]
        self.geometry = None

    def _plot_trunk(self, ax, trunk_height, alpha, facecolor='gray'):
        """plots the trunk"""
        trunk_position = (self.position[0] - self.trunk_width/2, self.position[1])
        trunk = mpatches.Rectangle(trunk_position, self.trunk_width, trunk_height, 
                                   facecolor=facecolor, edgecolor='black', zorder=2, alpha=alpha)
        ax.add_patch(trunk)
        return self.position[1] + trunk_height

    def _plot_deciduous(self, ax, alpha, n_lobes, lobe_radius_ratio, trunk_canopy_overlap, seed):
        """plots a round, organic canopy"""
        trunk_height = self.height * 0.35
        canopy_base_y = self.position[1] + trunk_height

        canopy_radius = (self.height - trunk_height) / 2
        canopy_center = (self.position[0], canopy_base_y + canopy_radius)

        visual_trunk_height = trunk_height + canopy_radius * trunk_canopy_overlap
        self._plot_trunk(ax, visual_trunk_height, alpha)

        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2*np.pi, n_lobes, endpoint=False) + rng.uniform(-0.2, 0.2, n_lobes)
        radial_jitter = rng.uniform(0.85, 1.15, n_lobes)

        circles = [Point(canopy_center).buffer(canopy_radius * 0.72)]
        for angle, jitter in zip(angles, radial_jitter):
            offset = (canopy_radius * 0.5 * jitter * np.cos(angle),
                      canopy_radius * 0.5 * jitter * np.sin(angle))
            lobe_center = tuple(c + o for c, o in zip(canopy_center, offset))
            circles.append(Point(lobe_center).buffer(canopy_radius * lobe_radius_ratio * jitter))

        union_shape = unary_union(circles)
        x, y = union_shape.exterior.xy
        ax.fill(x, y, facecolor='lightgray', edgecolor='black', linewidth=1, zorder=3, alpha=alpha)

        trunk_rect = box(self.position[0] - self.trunk_width/2, self.position[1], self.position[0] + self.trunk_width/2, self.position[1] + visual_trunk_height)
        self.geometry = unary_union([trunk_rect, union_shape])

    def _plot_conifer(self, ax, alpha, n_tiers, tier_overlap):
        """plots a stacked-triangle canopy"""
        trunk_height = self.height * 0.35
        base_y = self._plot_trunk(ax, trunk_height, alpha)

        remaining_height = self.height - trunk_height
        tier_height = remaining_height / (n_tiers - (n_tiers - 1) * tier_overlap)

        y = base_y
        tier_polygons = []
        for i in range(n_tiers):
            tier_width = self.canopy_width * (1 - i / n_tiers*0.6)
            apex_y = y + tier_height
            vertices = [
                (self.position[0] - tier_width / 2, y),
                (self.position[0] + tier_width / 2, y),
                (self.position[0], apex_y)
            ]
            triangle = mpatches.Polygon(vertices, closed=True, facecolor='lightgray', edgecolor='black', zorder=3+i, alpha=alpha)
            ax.add_patch(triangle)
            y += tier_height * (1 - tier_overlap)
            tier_polygons.append(Polygon(vertices))

        trunk_rect = box(self.position[0] - self.trunk_width/2, self.position[1], self.position[0] + self.trunk_width/2, self.position[1] + trunk_height)
        self.geometry = unary_union([trunk_rect] + tier_polygons)


    def _plot_poplar(self, ax, alpha, trunk_canopy_overlap):
        """plots a tall, narrow canopy"""
        trunk_height = self.height * 0.35        
        canopy_base_y = self.position[1] + trunk_height

        canopy_radius = (self.height - trunk_height) / 2
        canopy_center = (self.position[0], canopy_base_y + canopy_radius)

        visual_trunk_height = trunk_height + canopy_radius * trunk_canopy_overlap
        self._plot_trunk(ax, visual_trunk_height, alpha)

        canopy_height = self.height - trunk_height
        canopy_center = (self.position[0], canopy_base_y + canopy_height / 2)
        canopy = mpatches.Ellipse(canopy_center, width=self.canopy_width * 0.5, height=canopy_height, 
                                  facecolor='lightgray', edgecolor='black', zorder=3, alpha=alpha)
        ax.add_patch(canopy)

        trunk_rect = box(self.position[0] - self.trunk_width/2, self.position[1], self.position[0] + self.trunk_width/2, self.position[1] + visual_trunk_height)
        canopy_shape = scale(Point(canopy_center).buffer(1), xfact=self.canopy_width * 0.25, yfact=canopy_height * 0.5, origin=canopy_center)
        self.geometry = unary_union([trunk_rect, canopy_shape])

    def plot(self, ax, 
             alpha: float = 1,
             n_lobes: int = 6, lobe_radius_ratio: float = 0.5, trunk_canopy_overlap: float = 0.5, seed: int = 0, 
             n_tiers: int = 3, tier_overlap: float = 0.35):
        """plots the tree at given position based on the selected tree type"""
        if self.treetype == TreeType.DECIDUOUS:
            self._plot_deciduous(ax, alpha, n_lobes, lobe_radius_ratio, trunk_canopy_overlap, seed)
        elif self.treetype == TreeType.CONIFER:
            self._plot_conifer(ax, alpha, n_tiers, tier_overlap)
        elif self.treetype == TreeType.POPLAR:
            self._plot_poplar(ax, alpha, trunk_canopy_overlap)

    def get_geometry(self, ):
        """function to return silhouette of tree. Note, plot() must be called on tree before this is available."""
        if self.geometry == None:
            raise RuntimeError("Tree geometry is not available untill tree has been plotted (using .plot()).")
        return self.geometry