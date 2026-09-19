from .enums import ReflectionType, GroundType, TreeType

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

class Ground:
    def __init__(self, height : float = 0, groundtype : str = 'Common', 
                 river_x: tuple = None, coast_x: float = None, sea_side: str = 'right',
                 depth: float = 6, bank_slope: float = 0.25, water_offset: float = 1.5):
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

        if self.groundtype == GroundType.RIVER:
            if river_x is None or len(river_x) != 2 or river_x[0] >= river_x[1]:
                raise ValueError("River ground requires river_x=(x_left, x_right) with x_left < x_right.")
            if not 0 >= bank_slope <= 0.5:
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

    def _validate_depths(self):
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if not 0 <= self.water_offset < self.depth:
            raise ValueError("water_offset must be >= 0 and smaller than depth.")

    def _terrain_profile(self, x_min, x_max):
        """function to derive vertices of the land surface from x_min to x_max"""
        if self.groundtype == GroundType.RIVER:
            x0, x1 = self.river_x
            return [(x_min, self.height), (x0, self.height), (x0 + self._bank_dx, self.height - self.depth), (x1 - self._bank_dx, self.height - self.depth), (x1, self.height), (x_max, self.height)]
        if self.sea_side == 'right':
            return [(x_min, self.height), (self.coast_x, self.height), (self.coast_x + self._bank_dx, self.height - self.depth), (x_max, self.height - self.depth)]
        return [(x_min, self.height - self.depth), (self.coast_x - self._bank_dx, self.height - self.depth), (self.coast_x, self.height), (x_max, self.height)]

    def _water_polygon(self, x_min, x_max):
        """function to derive vertices of water surface from x_min to x_max"""
        s0, s1 = self._water_span
        if self.groundtype == GroundType.RIVER:
            x0, x1 = self.river_x
            return [(s0, self.water_level), (x0 + self._bank_dx, self.height - self.depth), (x1 - self._bank_dx, self.height - self.depth), (s1, self.water_level)]
        if self.sea_side == 'right':
            return [(s0, self.water_level), (self.coast_x + self._bank_dx, self.height - self.depth), (x_max, self.height - self.depth), (x_max, self.water_level)]
        return [(x_min, self.water_level), (x_min, self.height - self.depth), (self.coast_x - self._bank_dx, self.height - self.depth), (s1, self.water_level)]
        
    def plot(self, ax, 
             color : str = 'black', linewidth : float = 1, linestyle : str = '-'):
        """Draws the ground"""
        if self.groundtype == GroundType.COMMON:
            # plot the ground as a simple line
            ax.axhline(y=self.height,color=color, linewidth=linewidth, linestyle=linestyle)
        

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