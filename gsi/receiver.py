from .enums import AntennaType
from .environment import Building

import numpy as np
import matplotlib.patches as mpatches

class Receiver:
    def __init__(self, position : tuple, antenna_height : float, 
                 antennatype : str = 'Geodetic'):
        self.position = position
        self.antenna_height = antenna_height

        self.tilt_angle = None
        self.helix_height = None

        self.antennatype = AntennaType._value2member_map_[antennatype]

    def _rotate_point(self, point, tilt_angle, reference=None):
        """Rotate a point around a reference point by tilt_angle."""
        if reference is None:
            reference = np.array(self.position)  # default: rotate around receiver position
        
        R = np.array([[np.cos(tilt_angle), -np.sin(tilt_angle)],
                    [np.sin(tilt_angle),  np.cos(tilt_angle)]])
        
        point = np.array(point)
        reference = np.array(reference)
        
        return reference + R @ (point - reference)
    
    def _get_signal_endpoint(self):
        """Return the end-point of the antenna where the signal should be drawn to."""
        if self.antennatype == AntennaType.HELIX and self.helix_height is not None:
            # base of the helix
            base = np.array([self.position[0], self.position[1] - self.helix_height])
            tip = np.array([self.position[0], self.position[1]])
            
            if self.tilt_angle is not None and self.tilt_angle != 0:
                ref = base
                angle_rad = np.deg2rad(self.tilt_angle)
                return self._rotate_point(tip, angle_rad, reference=ref)
            else:
                return tip
        else:
            # for other types, signal ends at the top of the stick (unrotated)
            return np.array([self.position[0], self.position[1]])
        
    def plot(self, ax,
             radius : float = 1.3, start_angle : float = 0, end_angle : float = 180,
             n_turns : float = 1.5, helix_height : float = 1.2, n_points : int = 100,
             tilt_angle : float = 0,
             text_string : str = None, text_offset : tuple = (2,2), text_size : int = 10):
        """Plots the receiver on the given position"""

        if self.antennatype in [AntennaType.GEODETIC, AntennaType.PATCH]:
            # plot antenna height as a line
            ax.plot([self.position[0], self.position[0]],[self.position[1], self.position[1]-self.antenna_height], linewidth=2, color='black')

            # create antenna patch and add to plot
            antenna = mpatches.Wedge(self.position, radius, start_angle, end_angle, facecolor='lightgray', edgecolor='black', zorder=2)
            ax.add_patch(antenna)
        elif self.antennatype == AntennaType.HELIX:

            self.tilt_angle = tilt_angle
            self.helix_height = helix_height
            
            # plot antenna height as a line
            ax.plot([self.position[0], self.position[0]],[self.position[1]-self.helix_height, self.position[1]-self.antenna_height], linewidth=2, color='black')

            # define helical antenna
            theta = np.linspace(0, 2*np.pi * n_turns, n_points)
            x = self.position[0] + radius * np.cos(theta)
            z = np.linspace(self.position[1]-self.helix_height, self.antenna_height, len(theta))

            # check if helix antenna should be rotated
            helix = np.array([x, z]).T
            baseline_left = np.array([self.position[0] - radius, self.position[1] - self.helix_height])
            baseline_right = np.array([self.position[0] + radius, self.position[1] - self.helix_height])

            # apply rotation if needed
            if self.tilt_angle != 0:
                baseline_left = self._rotate_point(baseline_left, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height))
                baseline_right = self._rotate_point(baseline_right, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height))
                helix = np.array([self._rotate_point(pt, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height)) for pt in helix])

            # add to plot
            ax.plot(helix[:, 0], helix[:, 1], linewidth=2, color='gray', zorder=2)
            ax.plot([baseline_left[0], baseline_right[0]], [baseline_left[1], baseline_right[1]], linewidth=2, color='black')

        # add text if added
        if not text_string is None:
            ax.annotate(xy=(self.position[0] + text_offset[0], self.position[1] + text_offset[1]), text=text_string, size=text_size)

    def draw_distance(self, ax, building : 'Building',
                      adjusted_y : float = 2, end_line_length : float = 0.5, text_offset : float = 3, distance_string : str = None, fontsize : int = 10):
        """Plots the distance between the receiver and a given building"""
        if not isinstance(building, Building):
            raise TypeError(f'It is not possible to draw a distance between the receiver and anything else than a building.')

        # find the wall closest to the receiver
        if self.position[0] < building.position[0]:
            wall_x = building.position[0]  # left wall
        elif self.position[0] > building.position[0] + building.width:
            wall_x = building.position[0] + building.width  # right wall
        else:
            wall_x = self.position[0]  # directly above the receiver

        # find y-coordinate of line
        y_pos_line = self.position[1]-self.antenna_height-adjusted_y

        # draw distance lines onto plot
        ax.plot([self.position[0], wall_x], [y_pos_line, y_pos_line], linestyle='--', linewidth=1, color='black', zorder=2)
        ax.plot([self.position[0], self.position[0]],[y_pos_line-end_line_length, y_pos_line+end_line_length], linestyle='-', linewidth=1, color='black', zorder=2)
        ax.plot([wall_x, wall_x],[y_pos_line-end_line_length, y_pos_line+end_line_length], linestyle='-', linewidth=1, color='black', zorder=2)

        # find midpoint to place text
        midpoint_x = (self.position[0] + wall_x) / 2
        midpoint_y = y_pos_line

        # calculate distance between receiver and nearest wall
        distance = abs(self.position[0] - wall_x)

        # check if user has given a string to put in
        if not distance_string:
            distance_string = f'{distance:.2f} m'

        # draw text onto image in the middle of distance line
        ax.text(midpoint_x, midpoint_y-text_offset, distance_string,
                ha='center', va='bottom', fontsize=fontsize, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=1)

    def draw_radiation(self, ax,
                    field_of_view: float = 45, length: float = 20,
                    linestyle: str = '--', linewidth: float = 1.5, linecolor: str = 'green', line_alpha: float = 0.6,
                    fill_color: str = 'green', fill_alpha: float = 0.3):
        """Plots the radiation pattern for the receiver, accounting for tilt."""
        
        # define radiation pattern using field of view
        thetas = np.deg2rad([90 - field_of_view, 90 + field_of_view])  # initial angles (relative to upright)

        # calculate unrotated end points for the radiation pattern
        p1 = np.array([self.position[0] + length * np.cos(thetas[0]), 
                    self.position[1] + length * np.sin(thetas[0])])
        p2 = np.array([self.position[0] + length * np.cos(thetas[1]), 
                    self.position[1] + length * np.sin(thetas[1])])

        # apply tilt rotation if necessary
        if self.tilt_angle:
            if self.antennatype is AntennaType.HELIX:
                # rotate the receiver position (starting point of the beam)
                rotated_start = self._rotate_point(np.array(self.position), np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height))
                # rotate the radiation endpoints
                p1 = self._rotate_point(p1, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height))
                p2 = self._rotate_point(p2, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]-self.helix_height))
            elif self.antennatype in [AntennaType.GEODETIC, AntennaType.PATCH]:
                # rotate the receiver position (starting point of the beam)
                rotated_start = self._rotate_point(np.array(self.position), np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]))
                # rotate the radiation endpoints
                p1 = self._rotate_point(p1, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]))
                p2 = self._rotate_point(p2, np.deg2rad(self.tilt_angle), reference=(self.position[0], self.position[1]))

        else:
            rotated_start = np.array(self.position)  # no rotation needed

        # draw the radiation pattern lines
        ax.plot([rotated_start[0], p1[0]], [rotated_start[1], p1[1]], 
                linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=line_alpha, zorder=1)
        ax.plot([rotated_start[0], p2[0]], [rotated_start[1], p2[1]], 
                linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=line_alpha, zorder=1)

        # Define vertices for filled area and rotate them
        vertices = np.array([rotated_start, p1, p2])
        
        # Draw the filled radiation pattern
        ax.fill(vertices[:, 0], vertices[:, 1], color=fill_color, alpha=fill_alpha, zorder=0)

class Drone(Receiver):
    def __init__(self, position : tuple, rotation : float = 0):
        super().__init__(position=position, antenna_height=0)
        self.position = position
        self.rotation = rotation

        self.tilt_angle = self.rotation

    def _draw_propeller(self, ax, endpoint, 
                        propeller_width=0.65, propeller_height=0.15, propeller_angle=0,
                        propeller_holder_height=0.35, propeller_holder_width=0.15,
                        propeller_middle_radius=0.05, propeller_offset=0.15):
        """draws propellers with propeller holders"""
        # adjust with scale
        propeller_height *= self.scale
        propeller_width *= self.scale
        propeller_holder_height *= self.scale
        propeller_holder_width *= self.scale
        propeller_middle_radius *= self.scale
        propeller_offset *= self.scale

        propeller_angle += self.rotation

        # Define local offsets for propeller parts (centered on endpoint)
        prop_middle_offset = np.array([0, propeller_holder_height/2 + propeller_offset])
        left_offset = np.array([-propeller_width/2 - propeller_middle_radius, propeller_holder_height/2 + propeller_offset])
        right_offset = np.array([ propeller_width/2 + propeller_middle_radius, propeller_holder_height/2 + propeller_offset])
        holder_bottom_left = np.array([-propeller_holder_width/2, -propeller_holder_height/2])
        line_start_offset = np.array([0, propeller_holder_height/2])
        line_end_offset = np.array([0, propeller_holder_height/2 + propeller_offset])

        # Rotate all offsets by propeller_angle (drone rotation)
        prop_middle_pos = self._rotate_point(prop_middle_offset, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint
        left_pos = self._rotate_point(left_offset, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint
        right_pos = self._rotate_point(right_offset, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint
        holder_pos = self._rotate_point(holder_bottom_left, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint
        line_start = self._rotate_point(line_start_offset, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint
        line_end = self._rotate_point(line_end_offset, np.deg2rad(propeller_angle), reference=(0,0)) + endpoint

        # define patches
        propeller_middle = mpatches.Circle(prop_middle_pos, propeller_middle_radius,
                                        facecolor='lightgray', edgecolor='black', zorder=4)

        left_side_propeller = mpatches.Ellipse(left_pos, propeller_width, propeller_height, angle=propeller_angle,
                                            facecolor='lightgray', edgecolor='black', zorder=4)

        right_side_propeller = mpatches.Ellipse(right_pos, propeller_width, propeller_height, angle=propeller_angle,
                                                facecolor='lightgray', edgecolor='black', zorder=4)

        propeller_holder = mpatches.Rectangle(holder_pos, propeller_holder_width, propeller_holder_height, angle=propeller_angle,
                                            facecolor='lightgray', edgecolor='black', zorder=3)
        
        # add patches to plot
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], zorder=3, color='black', linewidth=3*self.scale/2)
        ax.add_patch(propeller_holder)
        ax.add_patch(propeller_middle)
        ax.add_patch(left_side_propeller)
        ax.add_patch(right_side_propeller)

    def _draw_straight_arm(self, ax, P0, P3, thickness=10, arm_angle_deg=0):
        """draws a straight arm between P0 (body) and P3 (rotor endpoint)"""
        P0 = np.array(P0)
        P3 = np.array(P3)

        cos_r, sin_r = np.cos(np.deg2rad(self.rotation)), np.sin(np.deg2rad(self.rotation))

        first_point_x_, first_point_y_ = self.a * np.cos(np.deg2rad(arm_angle_deg+thickness)), self.b * np.sin(np.deg2rad(arm_angle_deg+thickness))
        second_point_x_, second_point_y_ = self.a * np.cos(np.deg2rad(arm_angle_deg-thickness)), self.b * np.sin(np.deg2rad(arm_angle_deg-thickness))

        first_point_x = first_point_x_ * cos_r - first_point_y_ * sin_r + self.position[0]
        first_point_y = first_point_x_ * sin_r + first_point_y_ * cos_r + self.position[1]

        second_point_x = second_point_x_ * cos_r - second_point_y_ * sin_r + self.position[0]
        second_point_y = second_point_x_ * sin_r + second_point_y_ * cos_r + self.position[1]

        if arm_angle_deg > 360:
            arm_angle_deg -= 360

        if 180 <= arm_angle_deg <= 360:
            adjusted_length = 0.7
            adjust_tilt = 35
        else:
            adjusted_length = 0.8
            adjust_tilt = -10

        # direction unit vector
        direction = P3 - P0
        arm_length = np.linalg.norm(direction)
        direction_hat = direction / arm_length

        tilt_rad = np.deg2rad(adjust_tilt)

        u_hat = np.array([0,1])

        direction_tilted = np.cos(tilt_rad) * direction_hat + np.sin(tilt_rad) * u_hat
        direction_tilted /= np.linalg.norm(direction_tilted)

        direction_hat = direction_tilted

        # adjust length of arm
        P3 = P0 + direction_hat * arm_length * adjusted_length

        n_hat = np.array([-direction_hat[1], direction_hat[0]])

        P3_upper = P3 + 0.1 * n_hat
        P3_lower = P3 - 0.1 * n_hat

        # offsets for arm thickness
        P0_upper = np.array([first_point_x, first_point_y])
        P0_lower = np.array([second_point_x, second_point_y])

        # add arm polygon
        arm_poly = mpatches.Polygon([P0_upper,P3_upper,P3_lower,P0_lower],
                           closed=True, facecolor="lightgray", edgecolor="black", alpha=0.6, zorder=3)
        ax.add_patch(arm_poly)

        # draw propeller with handles
        self._draw_propeller(ax, P3)

    def plot(self, ax,
             number_of_arms: int = 4, arm_position_offset: float = 45,
             ellipse_width: float = 2, ellipse_height: float = 0.85,
             arm_height: float = 0.95, arm_tilt_angle: float = 0, arm_thickness:float = 10, 
             leg_length: float = 0.95, leg_angles: float = 15,
             scale: float = 1,
             text_string : str = None, text_offset : tuple = (2,2), text_size : int = 10):
        """Plots the drone on the given position"""
        # adjust with scale
        ellipse_width *= scale
        ellipse_height *= scale
        arm_height *= scale
        leg_length *= scale
        self.scale = scale

        # base of the drone 
        body = mpatches.Ellipse(self.position, width=ellipse_width, height=ellipse_height, angle=self.rotation, facecolor='lightgray', edgecolor='black', zorder=2)
        
        # define amount of arms
        arm_angles = np.linspace(360/number_of_arms, 360, number_of_arms) + arm_position_offset
        arm_rads = np.deg2rad(arm_angles)

        # define semiaxes of body
        self.a = ellipse_width / 2
        self.b = ellipse_height / 2

        # rotation matrix for ellipse orientation
        cos_r, sin_r = np.cos(np.deg2rad(self.rotation)), np.sin(np.deg2rad(self.rotation))

        # calculate corners of drone body
        for theta in arm_rads:
            # define points in ellipse
            x = self.a * np.cos(theta)
            y = self.b * np.sin(theta)

            # apply rotation to points
            x_rotated = x*cos_r - y*sin_r + self.position[0]
            y_rotated = x*sin_r + y*cos_r + self.position[1]

            # tilt arms
            unit_vector_x = np.cos(theta)
            unit_vector_y = np.sin(theta)
            x_tilt = unit_vector_x * np.cos(np.deg2rad(arm_tilt_angle)) - unit_vector_y * np.sin(np.deg2rad(arm_tilt_angle))
            y_tilt = unit_vector_x * np.sin(np.deg2rad(arm_tilt_angle)) + unit_vector_y * np.cos(np.deg2rad(arm_tilt_angle))

            # scale by arm height
            scaled_x_local = arm_height * x_tilt
            scaled_y_local = arm_height * y_tilt

            # rotate offset to world
            scaled_x_world = scaled_x_local * cos_r - scaled_y_local * sin_r
            scaled_y_world = scaled_x_local * sin_r + scaled_y_local * cos_r

            # define end points of arm
            arm_endpoint_x = x_rotated + scaled_x_world
            arm_endpoint_y = y_rotated + scaled_y_world

            # draw arms
            self._draw_straight_arm(ax, P0=(x_rotated, y_rotated), P3=(arm_endpoint_x, arm_endpoint_y), thickness=arm_thickness, arm_angle_deg=np.rad2deg(theta))

        # draw drone legs
        for angle in [-leg_angles, leg_angles]:
            angle += 135
            rad = np.deg2rad(angle)

            # tilt arms
            unit_vector_x = np.cos(rad)
            unit_vector_y = np.sin(rad)
            x_tilt = unit_vector_x * np.cos(rad) - unit_vector_y * np.sin(rad)
            y_tilt = unit_vector_x * np.sin(rad) + unit_vector_y * np.cos(rad)

            # scale by arm height
            scaled_x_local = leg_length * x_tilt
            scaled_y_local = leg_length * y_tilt

            # rotate offset to world
            scaled_x_world = scaled_x_local * cos_r - scaled_y_local * sin_r
            scaled_y_world = scaled_x_local * sin_r + scaled_y_local * cos_r

            # define end points of leg
            leg_endpoint_x = self.position[0] + scaled_x_world
            leg_endpoint_y = self.position[1] + scaled_y_world

            ax.plot([self.position[0], leg_endpoint_x], [self.position[1], leg_endpoint_y], color='black', linestyle='-', linewidth=2, zorder=1)

        # draw drone body
        ax.add_patch(body)

        # add text if added
        if not text_string is None:
            ax.annotate(xy=(self.position[0] + text_offset[0], self.position[1] + text_offset[1]), text=text_string, size=text_size)
        