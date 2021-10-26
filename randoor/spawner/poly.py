import numpy as np
import trimesh

from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

from ..utils import get_square_horizon
from . import geom

def sprinkle_cube(area_poly, count, cube_size, interior_thresh=0):
    xy, yaw = geom.sample_sprinkle(area_poly, count, cube_size+interior_thresh)
    return xy, yaw, [Polygon(get_square_horizon(p, cube_size//2, r)) for p, r in zip(xy, yaw)]

def create_zones(polys, label):
    parray = np.array(polys)
    cls_count = max(label)+1

    zone_polys = [None]*cls_count
    zone_hulls = [None]*cls_count

    for i in range(cls_count):
        cluster = parray[label==i]
        vertices = [p.exterior.coords for p in cluster]
        vertices_all = np.concatenate(vertices)
        zone_polys[i] = unary_union(cluster)
        zone_hulls[i] = MultiPoint(vertices_all).convex_hull
    
    return zone_polys, zone_hulls

def get_clustered_zones(polys, thresh):
    label, _ = geom.get_cluster(polys, thresh)
    return create_zones(polys, label)

def random_triangulation(n_points=10, x_min=-10, x_max=10, y_min=-10, y_max=10):
    assert x_min < x_max, 'x_min must be lower than x_max'
    assert y_min < y_max, 'y_min must be lower than y_max'
    points = np.random.random([n_points, 2])
    points[:,0] = points[:,0] * (x_max - x_min) + x_min
    points[:,1] = points[:,1] * (y_max - y_min) + y_min
    return unary_union(triangulate(MultiPoint(points)))