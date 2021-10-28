from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster import DBSCAN
import numpy as np
from trimesh.path.polygons import sample

def sample_sprinkle(area_poly, count, sample_thresh):
    sample_area = area_poly.buffer(-1*(sample_thresh))
    xy = sample(sample_area, count) # 2D
    yaw = np.random.random(count)*np.pi*2
    return xy, yaw

def sample_sure(area_poly, count, supplemental_range=0.2):
    factor = count / supplemental_range
    sampled = []
    while not len(sampled) == count:
        sampled = sample(area_poly, count, factor)
    return sampled

def get_cluster(polys, thresh):
    """get_cluster

    Function return cluster of that calculated by clustering polygons with DBSCAN.

    Args:
        polys (MultiPolygon, List[Polygon]): Clustering target.
        thresh ([type]): Distance threshold for clustering.

    Returns:
        List[int]: Cluster labels.
        List[(float,float)]: points xy of polygons centroid.

    """
    points = np.array([p.centroid.coords[0] for p in polys])
    db = DBSCAN(eps=thresh, min_samples=1).fit(points)
    return db.labels_, points

def sample_from_faces(polys, count=1, face_size=0.2):
    """sample_each_cluster

    Function return sample position from faces of clusters that made by clustering polys with DBSCAN.

    Args:
        polys (MultiPolygon, List[Polygon]): Polygons for sampling coordinates from faces.
        thresh (float): Distance threshold for clustering.

    Returns:
        List[(float,float)]: Coordinates (x,y) by sampled.
    
    """
    results = np.empty([len(polys),count,2])
    for i, p in enumerate(polys):
        face = Polygon(p.buffer(face_size).exterior.coords, [p.exterior.coords])
        results[i] = sample_sure(face, count, face_size)

    return results