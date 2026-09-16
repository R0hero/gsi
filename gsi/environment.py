from .enums import ReflectionType, GroundType, TreeType

import matplotlib.patches as mpatches
       
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
    def __init__(self, height : float = 0, groundtype : str = 'Common'):
        self.height = height

        # validate reflection type
        if groundtype not in GroundType._value2member_map_:
            valid_types = ', '.join([f"'{r.value}'" for r in GroundType])
            raise ValueError(f"Unknown ground type type '{groundtype}'. Use {valid_types}.")
        
        self.groundtype = GroundType._value2member_map_[groundtype]

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

    def _plot_deciduous(self, ax, alpha, n_lobes, lobe_radius_ratio, seed):
        """plots a round, organic canopy"""
        trunk_height = self.height * 0.35
        

    def _plot_conifer(self, ax, alpha, n_tiers, tier_overlap):
        """plots a stacked-triangle canopy"""
        trunk_height = self.height * 0.35

    def _plot_poplar(self, ax, alpha):
        """plots a tall, narrow canopy"""
        trunk_height = self.height * 0.35

    def plot(self, ax, 
             alpha: float = 1,
             n_lobes: int = 6, lobe_radius_ratio: float = 0.5, seed: int = 0, 
             n_tiers: int = 3, tier_overlap: float = 0.35):
        """plots the tree at given position based on the selected tree type"""
        if self.treetype == TreeType.DECIDUOUS:
            self._plot_deciduous(ax, alpha, n_lobes, lobe_radius_ratio, seed)
        elif self.TreeType == TreeType.CONIFER:
            self._plot_conifer(ax, alpha, n_tiers, tier_overlap)
        elif self.TreeType == TreeType.POPLAR:
            self._plot_poplar(ax, alpha)