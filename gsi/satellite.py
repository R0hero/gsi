from .enums import ReflectionType
from .receiver import Receiver, Drone
from .environment import Building, Ground, Tree
from .ionosphere import Blob

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

class Satellite:
    def __init__(self, position : tuple, rotation : float = 0):
        self.position = position
        self.rotation = rotation

    @staticmethod
    def _contiguous_runs(mask):
        """yields start and end index inclusive for each contiguous run of True in mask"""
        start = None
        for i, hidden in enumerate(mask):
            if hidden and start is None:
                start = i
            elif not hidden and start is not None:
                yield (start, i - 1)
                start = None
        if start is not None:
            yield (start, len(mask) - 1)

    def _rotate_element(self, ax, patch, angle, center):
        """Applies a rotation transformation to the given patch"""
        transform = mtransforms.Affine2D().rotate_deg_around(center[0], center[1], angle) + ax.transData
        patch.set_transform(transform)
        return patch
    
    def _find_intersection(self, line1_x, line1_y, line2_x, line2_y):
        """Finds the intersection of two lines given their endpoints"""
        # extract endpoints
        x1, x2 = line1_x
        y1, y2 = line1_y
        x3, x4 = line2_x
        y3, y4 = line2_y

        # calculate slope and y-intercept for line one
        denom1 = x2 - x1
        if denom1 != 0:
            m1 = (y2 - y1) / denom1
            b1 = y1 - m1 * x1
        else:
            m1, b1 = None, None  # line 1 is vertical

        # calculate slope and y-intercept for line two
        denom2 = x4 - x3
        if denom2 != 0:
            m2 = (y4 - y3) / denom2
            b2 = y3 - m2 * x3
        else:
            m2, b2 = None, None  # line 2 is vertical

        # check if lines are parallel
        if m1 == m2:
            return None  # no intersection (parallel lines or coincident)

        # find intersection point
        if m1 is not None and m2 is not None:
            # both lines are not vertical
            intersection_x = (b2 - b1) / (m1 - m2)
            intersection_y = m1 * intersection_x + b1
        elif m1 is None:  # line 1 is vertical
            intersection_x = x1
            intersection_y = m2 * intersection_x + b2
        elif m2 is None:  # line 2 is vertical
            intersection_x = x3
            intersection_y = m1 * intersection_x + b1

        # check if the intersection point is within both line segments
        if (min(x1, x2) <= intersection_x <= max(x1, x2) and
            min(y1, y2) <= intersection_y <= max(y1, y2) and
            min(x3, x4) <= intersection_x <= max(x3, x4) and
            min(y3, y4) <= intersection_y <= max(y3, y4)):
            return (intersection_x, intersection_y)

        return None # no intersection

    def _extend_to_ground(self, start, end, height, x_limits):
        """Find the intersection of a line with the ground level"""
        x1, y1 = start
        x2, y2 = end

        if y1 != y2:  # prevent division by zero for horizontal lines
            slope = (y2 - y1) / (x2 - x1)
            x_ground = x1 + (height - y1) / slope
            # constrain x_ground within x_limits
            if x_ground < x_limits[0]:
                x_ground = x_limits[0]
                height = y1 + slope * (x_ground - x1)
            elif x_ground > x_limits[1]:
                x_ground = x_limits[1]
                height = y1 + slope * (x_ground - x1)
            return (x_ground, height)
        return None
    
    def _angle_between_lines(self, point1, point2):
        """Find angle between points"""
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        angle = np.atan2(dy,dx)
        return angle

    def _generate_arc(self, start, end, control, n_points=50):
        """Generate an arc from start to end around a center with a given radius."""
        t = np.linspace(0,1,n_points)

        arc_x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
        arc_y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]

        return arc_x, arc_y

    def _round_corners(self, vertices_x, vertices_y, radius, n_points):
        """Round corners based on vertices using a Bezier curve"""

        # define points from vertices
        rounding_point1 = (vertices_x[0], vertices_y[0])
        rounding_point2 = (vertices_x[1], vertices_y[1])
        ground_point2 = (vertices_x[2], vertices_y[2])
        ground_point1 = (vertices_x[3], vertices_y[3])

        # compute angles of direction to place new center point for each corner
        left_angle = self._angle_between_lines(rounding_point1, ground_point2)
        right_angle = self._angle_between_lines(rounding_point2, ground_point1)

        # ensure correct angle
        if left_angle < 0:
            left_angle += np.pi
        if right_angle < 0:
            right_angle += np.pi
        
        # define new corners placed accordingly to given radius
        new_center_left = (rounding_point1[0] + (radius) * np.cos(left_angle),
                    rounding_point1[1] + (radius) * np.sin(right_angle))
        new_center_right = (rounding_point2[0] + (radius) * np.cos(right_angle),
                            rounding_point2[1] + (radius) * np.sin(right_angle))
        
        # calculate slope to find points on side
        left_vertical = (rounding_point1[0] - ground_point1[0]) == 0
        right_vertical = (rounding_point2[0] - ground_point2[0]) == 0
        if not left_vertical:
            slope_left = (rounding_point1[1] - ground_point1[1])/(rounding_point1[0] - ground_point1[0])
        else:
            slope_left = 0
        if not right_vertical:
            slope_right = (rounding_point2[1] - ground_point2[1])/(rounding_point2[0] - ground_point2[0])
        else:
            slope_right = 0

        # calculate starting point of the linear function
        starting_point_left = rounding_point1[1] - slope_left * rounding_point1[0]
        starting_point_right = rounding_point2[1] - slope_right * rounding_point2[0]
        
        # define points to start the rounded effect
        if left_vertical:
            start_rounding_left_x = rounding_point1[0]
        elif slope_left != 0:
            start_rounding_left_x = (new_center_left[1] - starting_point_left) / slope_left
        else:
            start_rounding_left_x = new_center_left[0]
        if right_vertical:
            start_rounding_right_x = rounding_point2[0]
        elif slope_right != 0:
            start_rounding_right_x = (new_center_right[1] - starting_point_right) / slope_right
        else:
            start_rounding_right_x = new_center_right[0]

        start_rounding_left_y = new_center_left[1]
        start_rounding_left = (start_rounding_left_x, start_rounding_left_y)
        start_rounding_right_y = new_center_right[1]
        start_rounding_right = (start_rounding_right_x, start_rounding_right_y)

        # define points to end the rounded effect
        end_rounding_left_x = new_center_left[0]
        end_rounding_left_y = rounding_point1[1]
        end_rounding_left = (end_rounding_left_x, end_rounding_left_y)
        end_rounding_right_x = new_center_right[0]
        end_rounding_right_y = rounding_point2[1]
        end_rounding_right = (end_rounding_right_x, end_rounding_right_y)
        
        # calculate bended corners using a Bezier curve
        arc_x_left, arc_y_left = self._generate_arc(start_rounding_left, end_rounding_left, rounding_point1, n_points)
        arc_x_right, arc_y_right = self._generate_arc(start_rounding_right, end_rounding_right, rounding_point2, n_points)

        return arc_x_left, arc_y_left, arc_x_right, arc_y_right

    def _terrain_stamp_polygon(self, seg_x, seg_ground_y, depth, radius, n_points, round_left=True, round_right=True):
        """helper function to mark footprint under a stretch of terrain with its bottom corners rounded by _round_corners"""
        seg_x = np.asarray(seg_x, dtype=float)
        seg_ground_y = np.asarray(seg_ground_y, dtype=float)
        x0, x1 = seg_x[0], seg_x[-1]

        radius = min(radius, np.hypot(x1 - x0, depth) / 2)

        arc_x_left, arc_y_left, arc_x_right, arc_y_right = self._round_corners([x0, x1, x1, x0], [-depth, -depth, 0, 0], radius, n_points)

        left = list(zip(arc_x_left, arc_y_left) if round_left else [(x0, -depth)])
        right = list(zip(arc_x_right, arc_y_right) if round_right else [(x1, -depth)])

        inner = seg_x[(seg_x > left[-1][0]) & (seg_x < right[0][0])]
        bottom = [(x, -depth) for x in inner]

        top = [(x, 0.0) for x in seg_x[::-1]]

        path = [(x0, 0.0)] + left + bottom + right + top
        xs = np.array([point[0] for point in path])
        ys = np.array([point[1] for point in path])

        return xs, ys + np.interp(xs, seg_x, seg_ground_y)

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Returns the minimum distance from a point to a line segment."""
        # ensure numpy arrays for easier computation
        point  = np.array(point)
        seg_start  = np.array(seg_start)
        seg_end  = np.array(seg_end)
        
        # calculate directions between relevant points
        direction_start_to_end = seg_end - seg_start
        direction_start_to_point = point - seg_start
        
        # calculate projection parameter
        projection_parameter = np.dot(direction_start_to_point, direction_start_to_end) / np.dot(direction_start_to_end, direction_start_to_end)
        projection_parameter = np.clip(projection_parameter, 0, 1) # clamp to segment
        
        closest = seg_start + projection_parameter * direction_start_to_end
        return np.linalg.norm(point - closest), closest

    def _plot_signal_line(self, ax, start, end, 
                          trees=None, color='black', linewidth=1, linestyle='-',
                          attenuated_color='darkolivegreen', attenuated_linestyle=None, zorder=1):
        """helper function to determining if a signal line should be attenuated when passing through a tree."""
        if attenuated_linestyle is None:
            attenuated_linestyle = linestyle
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        if not trees:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
            return 
        
        line = LineString([start, end])

        closest_distance = None
        for tree in trees:
            geometry = tree.get_geometry()
            if not line.intersects(geometry):
                continue
            intersection = line.intersection(geometry)

            candidate_points = []
            if intersection.geom_type == 'Point':
                candidate_points.append(intersection)
            elif intersection.geom_type == 'MultiPoint':
                candidate_points.extend(intersection.geoms)
            elif intersection.geom_type in ('LineString', 'MultiLineString'):
                geoms = [intersection] if intersection.geom_type == 'LineString' else intersection.geoms
                for geom in geoms:
                    candidate_points.extend(Point(c) for c in geom.coords)
            else:
                for geom in intersection.geoms:
                    candidate_points.extend(Point(c) for c in geom.coords)

            for point in candidate_points:
                distance = line.project(point)
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance

        if closest_distance is None:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
            return

        hit_point = np.array(line.interpolate(closest_distance).coords[0])

        ax.plot([start[0], hit_point[0]], [start[1], hit_point[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        ax.plot([hit_point[0], end[0]], [hit_point[1], end[1]], color=attenuated_color, linewidth=linewidth, linestyle=attenuated_linestyle, zorder=zorder)

    def _terrain_shadow_mask(self, xs, ys):
        """helper function to create a shadow mask behind terrain based on the satellites position"""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        dx = xs - self.position[0]
        dy = ys - self.position[1]

        with np.errstate(divide='ignore', invalid='ignore'):
            slope = dy / dx

        hidden = np.zeros(len(xs), dtype=bool)
        hiding_slope = np.full(len(xs), np.nan)

        right = np.where(dx > 0)[0]
        right = right[np.argsort(xs[right])]
        running_max = -np.inf
        for i in right:
            if slope[i] < running_max:
                hidden[i] = True
                hiding_slope[i] = running_max
            else:
                running_max = slope[i]

        left = np.where(dx < 0)[0]
        left = left[np.argsort(-xs[left])]
        running_min = np.inf
        for i in left:
            if slope[i] > running_min:
                hidden[i] = True
                hiding_slope[i] = running_min
            else:
                running_min = slope[i]

        return hidden, hiding_slope

    def plot(self, ax, 
             body_radius : float = 1, buffer_distance : float = 0.5, arm_width : float = 3, arm_height : float = 1.25, scale : float = 1):
        """Plots the satellite on the given position"""
        # adjust with scale
        body_radius = body_radius * scale
        buffer_distance = buffer_distance * scale
        arm_width = arm_width * scale
        arm_height = arm_height * scale

        # define body
        body = mpatches.Circle(self.position, radius=body_radius, facecolor='lightgray', edgecolor='black', zorder=2)

        # calculate arm positions
        right_arm_position = (self.position[0]+body_radius+buffer_distance, self.position[1]-arm_height/2)
        left_arm_position = (self.position[0]-arm_width-buffer_distance-body_radius, self.position[1]-arm_height/2)

        # define arms
        right_arm = mpatches.Rectangle(right_arm_position, arm_width, arm_height, facecolor='lightgray', edgecolor='black', zorder=2)
        left_arm = mpatches.Rectangle(left_arm_position, arm_width, arm_height, facecolor='lightgray', edgecolor='black', zorder=2)

        # calculate connection points between body and arms
        connecting_point_right_arm = (right_arm_position[0], right_arm_position[1]+arm_height/2)
        connecting_point_left_arm = (left_arm_position[0]+arm_width, left_arm_position[1]+arm_height/2)
        connecting_point_right_body = (self.position[0]+body_radius, self.position[1])
        connecting_point_left_body = (self.position[0]-body_radius, self.position[1])

        # add satellite elements with rotation to plot
        ax.add_patch(self._rotate_element(ax, body, angle=self.rotation, center=self.position))
        ax.add_patch(self._rotate_element(ax, right_arm, angle=self.rotation, center=self.position))
        ax.add_patch(self._rotate_element(ax, left_arm, angle=self.rotation, center=self.position))

        # calculate rotation matrix for connection lines
        angle_rad = np.deg2rad(self.rotation)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

        # define lines with rotation and add to plot
        for line in [
            (connecting_point_right_arm, connecting_point_right_body),
            (connecting_point_left_arm, connecting_point_left_body)
        ]:
            rotated_line = [np.dot(rotation_matrix, np.array([x - self.position[0], y - self.position[1]])) + np.array(self.position) for x, y in line]
            ax.plot([rotated_line[0][0], rotated_line[1][0]], [rotated_line[0][1], rotated_line[1][1]], color='black', zorder=2)

    def draw_signal(self, ax, *targets, 
                    color: str = 'black', linewidth: float = 1, linestyle: str = '-', 
                    num_rays: int = 12, buffer_distance: float = 1, line_length: float = 3,
                    corner_radius: float = 5,
                    attenuated_color: str = 'darkolivegreen', attenuated_linestyle: str = '-', zorder_signal_to_tree: int = 1):
        """Draws a signal between the satellite and specified targets (Building or Receiver)"""
        trees = [target for target in targets if isinstance(target, Tree)]
        targets = [target for target in targets if not isinstance(target, Tree)]

        # if only Receiver is given as input, plot a line between the satellite and the receiver
        if len(targets) == 1:
            target = targets[0]
            if isinstance(target, Receiver|Drone):
                receiver_position = target._get_signal_endpoint()
                self._plot_signal_line(ax, self.position, receiver_position, trees=trees, color=color, linewidth=linewidth, linestyle=linestyle, attenuated_color=attenuated_color, attenuated_linestyle=attenuated_linestyle, zorder=zorder_signal_to_tree)
        # if two inputs are given, check which one is first
        elif len(targets) == 2:
            # if Building is first and Receiver is second, check if there is a reflection
            if isinstance(targets[0], Building) and isinstance(targets[1], Receiver):
                building = targets[0]
                receiver = targets[1]

                # find the wall closest to the receiver
                if receiver.position[0] < building.position[0]:
                    wall_x = building.position[0]  # left wall
                elif receiver.position[0] > building.position[0] + building.width:
                    wall_x = building.position[0] + building.width  # right wall
                else:
                    wall_x = receiver.position[0]  # directly above the receiver

                # define the wall point in which will be closest to the antenna
                wall_point = (wall_x, receiver.position[1])

                # calculate where the image antenna will be placed
                image_antenna = tuple(
                    rp + 2 * (wp - rp) 
                    for rp, wp in zip(receiver._get_signal_endpoint(), wall_point)
                )

                # define lines between satellite and image receiver
                line_sat_to_receiver_x = np.array([self.position[0], image_antenna[0]])
                line_sat_to_receiver_y = np.array([self.position[1], image_antenna[1]])

                # perform reflection computation based on if the reflection type is specular
                if building.reflection == ReflectionType.SPECULAR:
                    # define lines for the wall
                    line_wall_x = np.array([wall_x, wall_x])
                    line_wall_y = np.array([building.position[1], building.position[1] + building.height])

                    # check if the lines are intersecting, and if they are, calculate the point
                    intersection = self._find_intersection(line_sat_to_receiver_x, line_sat_to_receiver_y, line_wall_x, line_wall_y)
                    
                    # if intersection happens, plot the reflected signal
                    if intersection:
                        # line between satellite and wall
                        ax.plot([self.position[0], intersection[0]], [self.position[1], intersection[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        # line between wall and receiver
                        receiver_position = receiver._get_signal_endpoint()
                        ax.plot([intersection[0], receiver_position[0]], [intersection[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                
                # perform reflection computation based on if the reflection type is diffuse
                elif building.reflection == ReflectionType.DIFFUSE:

                    # check if receiver is on the correct side of the reflective wall
                    if self.position[0] < wall_x:

                        line_sat_to_receiver_x = np.array([self.position[0], receiver.position[0]])
                        line_sat_to_receiver_y = np.array([self.position[1], receiver.position[1]])
                
                    # define lines for the wall, with an increased y-endpoint as the actual point is not important
                    line_wall_x = np.array([wall_x, wall_x])
                    line_wall_y = np.array([building.position[1], building.position[1] + self.position[1]])

                    # check if the lines are intersecting, and if they are, calculate the point
                    intersection = self._find_intersection(line_sat_to_receiver_x, line_sat_to_receiver_y, line_wall_x, line_wall_y)

                    # if intersection happens, plot the reflected signal
                    if intersection:
                        # line between satellite and wall
                        ax.plot([self.position[0], intersection[0]], [self.position[1], building.position[1] + building.height], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        
                        # define the angles between the diffuse rays
                        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

                        # compute start and end points for each ray
                        start_points = [
                            (intersection[0] + buffer_distance * np.cos(angle),
                            building.position[1] + building.height + buffer_distance * np.sin(angle))
                            for angle in angles
                        ]
                        end_points = [
                            (intersection[0] + (buffer_distance + line_length) * np.cos(angle),
                            building.position[1] + building.height + (buffer_distance + line_length) * np.sin(angle))
                            for angle in angles
                        ]

                        for start, end in zip(start_points, end_points):
                            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)

            elif isinstance(targets[0], Ground) and isinstance(targets[1], Receiver):
                # if Ground is first and Receiver is second, plot a line between ground and receiver
                ground = targets[0]
                receiver = targets[1]

                # surface which can reflect (whole ground for Common, only the water for River/Coast)
                surface = ground.reflective_surface

                if surface is None:
                    return
                level, surface_x_start, surface_x_end = surface

                if self.position[1] > level and receiver.position[1] > level:
                    reflection_x = receiver.position[0] + (self.position[0] - receiver.position[0]) * (receiver.position[1] - level) / ((receiver.position[1] - level) + (self.position[1] - level))

                    if surface_x_start <= reflection_x <= surface_x_end:
                        intersection = (reflection_x, level)
                        # line between satellite and ground
                        ax.plot([self.position[0], intersection[0]], [self.position[1], intersection[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        # line between ground and receiver
                        receiver_position = receiver._get_signal_endpoint()
                        ax.plot([intersection[0], receiver_position[0]], [intersection[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)

            elif isinstance(targets[0], Receiver) and isinstance(targets[1], Building):
                # if Receiver is first and Building is second, check if the building blocks the signal
                receiver = targets[0]
                building = targets[1]

                # find the wall closest to the receiver
                if receiver.position[0] < building.position[0]:
                    wall_x = building.position[0]  # left wall
                elif receiver.position[0] > building.position[0] + building.width:
                    wall_x = building.position[0] + building.width  # right wall
                else:
                    wall_x = receiver.position[0]  # directly above the receiver

                # define lines between satellite and receiver
                line_sat_to_receiver_x = np.array([self.position[0], receiver.position[0]])
                line_sat_to_receiver_y = np.array([self.position[1], receiver.position[1]])
                # define lines for the wall
                line_wall_x = np.array([wall_x, wall_x])
                line_wall_y = np.array([building.position[1], building.position[1] + building.height])

                # check if the lines are intersecting, and if they are, calculate the point
                intersection = self._find_intersection(line_sat_to_receiver_x, line_sat_to_receiver_y, line_wall_x, line_wall_y)
                if intersection:
                    corner = (wall_x, building.height)

                    corner_distance, closest_point_to_corner = self._point_to_segment_distance(corner, self.position, receiver.position)

                    if corner_distance < corner_radius:
                        corner_side_check = (
                            building.position[0] <= closest_point_to_corner[0] <= building.position[0] + building.width
                        )
                        if corner_side_check:
                            ax.plot([self.position[0], corner[0]], [self.position[1], corner[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                            receiver_position = receiver._get_signal_endpoint()
                            ax.plot([corner[0], receiver_position[0]], [corner[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        else:
                            # plot the signal between the satellite and the receiver
                            receiver_position = receiver._get_signal_endpoint()
                            ax.plot([self.position[0], receiver_position[0]], [self.position[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                    else:
                        # plot the signal between the satellite and blocking building
                        ax.plot([self.position[0], intersection[0]], [self.position[1], intersection[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)

                else:
                    # plot the signal between the satellite and the receiver
                    receiver_position = receiver._get_signal_endpoint()
                    ax.plot([self.position[0], receiver_position[0]], [self.position[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
        if len(targets) == 3:
            
            if isinstance(targets[0], Building) and isinstance(targets[1], Building) and isinstance(targets[2], Receiver):
                # if first two targets are Building and final is Receiver, check if the buildings blocks the signals for reflections
                building_one = targets[0]
                building_two = targets[1]
                receiver = targets[2]

                # find the wall for building one, which is closest to the receiver
                if receiver.position[0] < building_one.position[0]:
                    building_one_wall_x = building_one.position[0]
                elif receiver.position[0] > building_one.position[0] + building_one.width:
                    building_one_wall_x = building_one.position[0] + building_one.width
                else:
                    building_one_wall_x = receiver.position[0]
                # find the wall for building two, which is closest to the receiver
                if receiver.position[0] < building_two.position[0]:
                    building_two_wall_x = building_two.position[0]
                elif receiver.position[0] > building_two.position[0] + building_two.width:
                    building_two_wall_x = building_two.position[0] + building_two.width
                else:
                    building_two_wall_x = receiver.position[0]

                # define the wall point in which will be closest to the antenna
                wall_point_one = (building_one_wall_x, receiver.position[1])
                wall_point_two = (building_two_wall_x, receiver.position[1])

                # calculate where the image antenna will be placed
                image_antenna_one = tuple(
                    rp + 2 * (wp - rp) 
                    for rp, wp in zip(receiver._get_signal_endpoint(), wall_point_one)
                )
                image_antenna_two = tuple(
                    rp + 2 * (wp - rp) 
                    for rp, wp in zip(receiver._get_signal_endpoint(), wall_point_two)
                )

                # define lines between satellite and image receiver
                line_sat_to_receiver_x_one = np.array([self.position[0], image_antenna_one[0]])
                line_sat_to_receiver_y_one = np.array([self.position[1], image_antenna_one[1]])
                line_sat_to_receiver_x_two = np.array([self.position[0], image_antenna_two[0]])
                line_sat_to_receiver_y_two = np.array([self.position[1], image_antenna_two[1]])
                # define lines for the walls
                line_wall_x_one = np.array([building_one_wall_x, building_one_wall_x])
                line_wall_y_one = np.array([building_one.position[1], building_one.position[1] + building_one.height])
                line_wall_x_two = np.array([building_two_wall_x, building_two_wall_x])
                line_wall_y_two = np.array([building_two.position[1], building_two.position[1] + building_two.height])

                if abs(building_one_wall_x-self.position[0]) > abs(building_two_wall_x-self.position[0]):
                    blocked = self._find_intersection(line_sat_to_receiver_x_one, line_sat_to_receiver_y_one, line_wall_x_two, line_wall_y_two)
                    if not blocked:
                        # perform reflection computation based on if the reflection type is specular
                        if building_one.reflection == ReflectionType.SPECULAR:
                            # check if the lines are intersecting, and if they are, calculate the point
                            intersection = self._find_intersection(line_sat_to_receiver_x_one, line_sat_to_receiver_y_one, line_wall_x_one, line_wall_y_one)
                            
                            # if intersection happens, plot the reflected signal
                            if intersection:
                                # line between satellite and wall
                                ax.plot([self.position[0], intersection[0]], [self.position[1], intersection[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                                # line between wall and receiver
                                receiver_position = receiver._get_signal_endpoint()
                                ax.plot([intersection[0], receiver_position[0]], [intersection[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        # perform reflection computation based on if the reflection type is diffuse
                        elif building_one.reflection == ReflectionType.DIFFUSE:
                            # check if the lines are intersecting, and if they are, calculate the point
                            intersection = self._find_intersection(line_sat_to_receiver_x_one, line_sat_to_receiver_y_one, line_wall_x_one, line_wall_y_one)

                            # if intersection happens, plot the reflected signal
                            if intersection:
                                # line between satellite and wall
                                ax.plot([self.position[0], intersection[0]], [self.position[1], building_two.position[1] + building_two.height], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                                
                                # define the angles between the diffuse rays
                                angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

                                # ompute start and end points for each ray
                                start_points = [
                                    (intersection[0] + buffer_distance * np.cos(angle),
                                    building_two.position[1] + building_two.height + buffer_distance * np.sin(angle))
                                    for angle in angles
                                ]
                                end_points = [
                                    (intersection[0] + (buffer_distance + line_length) * np.cos(angle),
                                    building_two.position[1] + building_two.height + (buffer_distance + line_length) * np.sin(angle))
                                    for angle in angles
                                ]

                                for start, end in zip(start_points, end_points):
                                    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)

                if abs(building_one_wall_x-self.position[0]) < abs(building_two_wall_x-self.position[0]):
                    blocked = self._find_intersection(line_sat_to_receiver_x_two, line_sat_to_receiver_y_two, line_wall_x_one, line_wall_y_one)
                    if not blocked:
                        # perform reflection computation based on if the reflection type is specular
                        if building_two.reflection == ReflectionType.SPECULAR:
                            # check if the lines are intersecting, and if they are, calculate the point
                            intersection = self._find_intersection(line_sat_to_receiver_x_two, line_sat_to_receiver_y_two, line_wall_x_two, line_wall_y_two)
                            
                            # if intersection happens, plot the reflected signal
                            if intersection:
                                # line between satellite and wall
                                ax.plot([self.position[0], intersection[0]], [self.position[1], intersection[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                                # line between wall and receiver
                                receiver_position = receiver._get_signal_endpoint()
                                ax.plot([intersection[0], receiver_position[0]], [intersection[1], receiver_position[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                        # perform reflection computation based on if the reflection type is diffuse
                        elif building_two.reflection == ReflectionType.DIFFUSE:
                            # check if the lines are intersecting, and if they are, calculate the point
                            intersection = self._find_intersection(line_sat_to_receiver_x_two, line_sat_to_receiver_y_two, line_wall_x_two, line_wall_y_two)

                            # if intersection happens, plot the reflected signal
                            if intersection:
                                # line between satellite and wall
                                ax.plot([self.position[0], intersection[0]], [self.position[1], building_two.position[1] + building_two.height], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)
                                
                                # define the angles between the diffuse rays
                                angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

                                # ompute start and end points for each ray
                                start_points = [
                                    (intersection[0] + buffer_distance * np.cos(angle),
                                    building_one.position[1] + building_one.height + buffer_distance * np.sin(angle))
                                    for angle in angles
                                ]
                                end_points = [
                                    (intersection[0] + (buffer_distance + line_length) * np.cos(angle),
                                    building_one.position[1] + building_one.height + (buffer_distance + line_length) * np.sin(angle))
                                    for angle in angles
                                ]

                                for start, end in zip(start_points, end_points):
                                    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=1)

    def draw_footprint(self, ax, ground : Ground, *targets : Blob | Tree,
                        x_limits = None,
                        edge_color : str = 'black', edge_linestyle : str = '--', edge_linewidth : float = 1,
                        fill_color : str = 'lightblue', fill_alpha : float = 0.5,
                        plot_edge_lines : bool = True, shadow_linestyle : str = '--', shadow_linewidth : float = 0.8,
                        shadow_color : str = 'red', shadow_alpha : float = 0.3,
                        draw_ground : bool = True, ground_offset : float = 5, ground_footprint_offset : float = 3, corner_radius : float = 3, n_points_arcs : int = 15,
                        terrain_shadow: bool = True):
        """Draws the footprint of the satellite"""
        
        # extract plot limits from current plot
        if x_limits is None:
            x_limits = ax.get_xlim()
        difference_x = abs(x_limits[0] - x_limits[1])
        # define y limits
        y_ground_edgelines = ground.height + difference_x*0.2
        y_ground = ground.height

        # define vertices of polygon
        vertices_x = [x_limits[0], x_limits[0], self.position[0], x_limits[1], x_limits[1], x_limits[0]]
        vertices_y = [y_ground, y_ground_edgelines, self.position[1], y_ground_edgelines, y_ground, y_ground]
        
        # left edgeline for footprint
        ax.plot([self.position[0], x_limits[0]], [self.position[1], y_ground_edgelines], color=edge_color, linestyle=edge_linestyle, linewidth=edge_linewidth, zorder=1)
        # right edgeline for footprint
        ax.plot([self.position[0], x_limits[1]], [self.position[1], y_ground_edgelines], color=edge_color, linestyle=edge_linestyle, linewidth=edge_linewidth, zorder=1)
        # fill out footprint with a color
        surface = ground._surface_profile(x_limits[0], x_limits[1])
        fill_x = [x_limits[0], self.position[0], x_limits[1]] + [p[0] for p in surface[::-1]]
        fill_y = [y_ground_edgelines, self.position[1], y_ground_edgelines] + [p[1] for p in surface[::-1]]
        ax.fill(fill_x, fill_y, color=fill_color, alpha=fill_alpha, zorder=0)

        # create a footprint polygon
        footprint_polygon = Polygon(zip(vertices_x, vertices_y))

        # loop through all targets
        for target in targets:
            # check if targets are Blobs
            if not isinstance(target, Blob) and not isinstance(target, Tree):
                continue
            target_geometry = target.get_geometry()

            target_polygon = target_geometry.convex_hull

            # define the polygon and check if they intersect with the footprint
            intersection = footprint_polygon.intersection(target_polygon)

            # check if there is an intersection
            if intersection.is_empty:
                continue

            hull_coords = list(target_polygon.exterior.coords)[:-1]
            angles = [np.arctan2(y - self.position[1], x - self.position[0]) for x,y in hull_coords]
            idx_left_most = np.argmin(angles)
            idx_right_most = np.argmax(angles)
            left_point = hull_coords[idx_left_most]
            right_point = hull_coords[idx_right_most]
            
            # extend the points to the ground or to the x_limits
            left_extended = self._extend_to_ground(self.position, left_point, y_ground, x_limits)
            right_extended = self._extend_to_ground(self.position, right_point, y_ground, x_limits)

            # if plot edge lines are activated, plot them
            if plot_edge_lines:
                ax.plot([self.position[0], left_extended[0]], [self.position[1], left_extended[1]], color=shadow_color, linestyle=shadow_linestyle, alpha=shadow_alpha, linewidth=shadow_linewidth, zorder=1)
                ax.plot([self.position[0], right_extended[0]], [self.position[1], right_extended[1]], color=shadow_color, linestyle=shadow_linestyle, alpha=shadow_alpha, linewidth=shadow_linewidth, zorder=1)

            # check if ray hits corner and define vertices
            if left_extended[0] < x_limits[1] and right_extended[0] >= x_limits[1]:
                target_vertices_x = [right_point[0], left_point[0], left_extended[0], x_limits[1], right_extended[0]]
                target_vertices_y = [right_point[1], left_point[1], left_extended[1], y_ground, right_extended[1]]
            elif right_extended[0] > x_limits[0] and left_extended[0] <= x_limits[0]:
                target_vertices_x = [right_point[0], left_point[0], left_extended[0], x_limits[0], right_extended[0]]
                target_vertices_y = [right_point[1], left_point[1], left_extended[1], y_ground, right_extended[1]]
            else:
                target_vertices_x = [right_point[0], left_point[0], left_extended[0], right_extended[0]]
                target_vertices_y = [right_point[1], left_point[1], left_extended[1], right_extended[1]]

            quad_polygon = Polygon(zip(target_vertices_x, target_vertices_y))
            shadow_shape = unary_union([target_geometry, quad_polygon])
            if shadow_shape.geom_type != 'Polygon':
                shadow_shape = shadow_shape.convex_hull
            shadow_x, shadow_y = shadow_shape.exterior.xy
            
            # fill out with color
            ax.fill(shadow_x, shadow_y, color=shadow_color, alpha=shadow_alpha, zorder=0)

            # fill out a footprint color for the ground if draw_ground is triggered as True
            if draw_ground:
                # extend line to below ground according to offset
                left_extended_below_ground = self._extend_to_ground(self.position, left_point, y_ground-ground_footprint_offset, x_limits)
                right_extended_below_ground = self._extend_to_ground(self.position, right_point, y_ground-ground_footprint_offset, x_limits)

                # check if ray hits corner and define vertices
                if left_extended[0] < x_limits[1] and right_extended[0] >= x_limits[1]:
                    # define limits
                    footprint_below_ground_x = [left_extended_below_ground[0], x_limits[1],  x_limits[1], left_extended[0]]
                    footprint_below_ground_y = [left_extended_below_ground[1], y_ground-ground_footprint_offset, y_ground, left_extended[1]]

                    # # define points for rounded corners
                    arc_x_left, arc_y_left, arc_x_right, arc_y_right = self._round_corners(footprint_below_ground_x, footprint_below_ground_y, corner_radius, n_points_arcs)

                    # # # define new vertices
                    footprint_below_ground_x = [left_extended[0]] + list(arc_x_left) + [x_limits[1], x_limits[1]]
                    footprint_below_ground_y = [left_extended[1]] + list(arc_y_left) + [y_ground-ground_footprint_offset, y_ground]
                elif right_extended[0] > x_limits[0] and left_extended[0] <= x_limits[0]:
                    # define limits
                    footprint_below_ground_x = [x_limits[0], right_extended_below_ground[0], right_extended[0], x_limits[0]]
                    footprint_below_ground_y = [y_ground-ground_footprint_offset, right_extended_below_ground[1], right_extended[1], y_ground]

                    # define points for rounded corners
                    arc_x_left, arc_y_left, arc_x_right, arc_y_right = self._round_corners(footprint_below_ground_x, footprint_below_ground_y, corner_radius, n_points_arcs)

                    # # define new vertices
                    footprint_below_ground_x = [right_extended[0]] + list(arc_x_right) + [x_limits[0], x_limits[0]]
                    footprint_below_ground_y = [right_extended[1]] + list(arc_y_right) + [y_ground-ground_footprint_offset, y_ground]
                else:
                    # define limits
                    footprint_below_ground_x = [left_extended_below_ground[0], right_extended_below_ground[0], right_extended[0], left_extended[0]]
                    footprint_below_ground_y = [left_extended_below_ground[1], right_extended_below_ground[1], right_extended[1], left_extended[1]]

                    # # define points for rounded corners
                    arc_x_left, arc_y_left, arc_x_right, arc_y_right = self._round_corners(footprint_below_ground_x, footprint_below_ground_y, corner_radius, n_points_arcs)

                    # # # define new vertices
                    footprint_below_ground_x = [left_extended[0]] + list(arc_x_left) + list(arc_x_right[::-1]) + [right_extended[0]]
                    footprint_below_ground_y = [left_extended[1]] + list(arc_y_left) + list(arc_y_right[::-1]) + [right_extended[1]]

                # fill out with color
                ax.fill(footprint_below_ground_x, footprint_below_ground_y, color=shadow_color, alpha=shadow_alpha, zorder=0)

        # fill out a background color for the ground if draw_ground is triggered as True
        if draw_ground:
            terrain = ground._terrain_profile(x_limits[0], x_limits[1])
            below_ground_vertice_x = [p[0] for p in terrain] + [p[0] for p in terrain[::-1]]
            below_ground_vertice_y = [p[1] for p in terrain] + [p[1] - ground_offset for p in terrain[::-1]]

            # fill out with color
            ax.fill(below_ground_vertice_x, below_ground_vertice_y, color=fill_color, alpha=fill_alpha, zorder=0)

        if terrain_shadow and ground.features:
            terrain = ground._terrain_profile(x_limits[0], x_limits[1])
            xs = np.array([p[0] for p in terrain])
            ys = np.array([p[1] for p in terrain])

            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]

            hidden, hiding_slope = self._terrain_shadow_mask(xs, ys)

            for start, end in self._contiguous_runs(hidden):
                seg_x = xs[start:end + 1]
                seg_ground_y = ys[start:end + 1]

                ray_slope = hiding_slope[start]
                ray_y = self.position[1] + ray_slope * (seg_x - self.position[0])

                ax.fill_between(seg_x, seg_ground_y, ray_y, color=shadow_color, alpha=shadow_alpha, linewidth=0, zorder=0.7)
                if plot_edge_lines:
                    ax.plot([seg_x[0], seg_x[-1]], [ray_y[0], ray_y[-1]], color=shadow_color, alpha=shadow_alpha, linestyle=shadow_linestyle, linewidth=shadow_linewidth, zorder=1)
                if draw_ground:
                    run_length = seg_x[-1] - seg_x[0]
                    ramp = min(corner_radius, run_length / 2) if run_length > 0 else 0
                    if ramp > 0:
                        from_left = seg_x - seg_x[0]
                        from_right = seg_x[-1] - seg_x
                        ease_left = 0.5 * (1 - np.cos(np.pi * np.clip(from_left/ramp, 0, 1)))
                        ease_right = 0.5 * (1 - np.cos(np.pi * np.clip(from_right/ramp, 0, 1)))
                        depth = ground_footprint_offset * np.minimum(ease_left, ease_right)
                    else:
                        depth = np.full_like(seg_x, ground_footprint_offset)

                    ax.fill_between(seg_x, seg_ground_y - depth, seg_ground_y, color=shadow_color, alpha=shadow_alpha, linewidth=0, zorder=0.65)