from shapely.geometry import Point
from shapely.ops import unary_union

class Blob:
    def __init__(self, position : tuple, radius : float,
                 offsets_blobs : list[tuple] = None, radii_blobs: list = None):
        self.position = position
        self.radius = radius

        self.offsets_blobs = offsets_blobs
        self.radii_blobs = radii_blobs

        # calculate the union of the components
        self._calculate_union()

    def _calculate_union(self):
        """Calculates the union of the outer edge in the blob"""
        # create circles as Shapely objects
        circles = [Point(self.position).buffer(self.radius)]
        if not self.offsets_blobs is None and not self.radii_blobs is None:
            for offset, radius_blob in zip(self.offsets_blobs, self.radii_blobs):
                center_blob = tuple(c + o for c, o in zip(self.position, offset))
                circles.append(Point(center_blob).buffer(radius_blob))

        # compute the union of all circles
        union_shape = unary_union(circles)
        self.outer_x, self.outer_y = union_shape.exterior.xy

    def plot(self, ax,
             color_edge : str = 'black', linestyle_edge : str = '-', linewidth_edge : float = 1,
             color_fill : str = 'lightgray', alpha_fill : float = 1):
        """Plots the blob on the given position"""
        # plot the edge
        ax.plot(self.outer_x, self.outer_y, color=color_edge, linestyle=linestyle_edge, linewidth=linewidth_edge, zorder=2)

        # fill the blob
        ax.fill(self.outer_x, self.outer_y, color=color_fill, alpha=alpha_fill, zorder=1)
