from shapely.geometry import Polygon
import numpy as np
import trimesh

from .geometric_utils import get_square_horizon

def sprinkle(area_poly, count, sample_thresh):
    sample_area = area_poly.buffer(-1*(sample_thresh))
    xy = trimesh.path.polygons.sample(sample_area, count) # 2D
    yaw = np.random.random(count)*np.pi*2
    return xy, yaw

def sprinkle_cube(area_poly, count, cube_size, interior_thresh=0.1):
    xy, yaw = sprinkle(area_poly, count, cube_size+interior_thresh)
    return [Polygon(get_square_horizon(p, cube_size//2, r)) for p, r in zip(xy, yaw)]

    
