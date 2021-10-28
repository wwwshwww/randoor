from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import numpy as np

from ..spawner.poly import sprinkle_cube, simple_cube, get_clustered_zones
from ..spawner.geom import sample_from_faces
from .obstacle_room import ObstacleRoomConfig, ObstacleRoomGenerator

class SimpleSearchRoomConfig(ObstacleRoomConfig):
    def __init__(self, 
                 wall_shape, 
                 obstacle_shape, 
                 target_shape, 
                 obstacle_count, 
                 target_count, 
                 obstacle_hulls):

        super(SimpleSearchRoomConfig, self).__init__(wall_shape, obstacle_shape, obstacle_count)

        self.target_shape = target_shape
        self.target_count = target_count
        self.obstacle_hulls = obstacle_hulls

        self.tag_wall = 'wall'
        self.tag_obstacle = 'obstacle'
        self.tag_target = 'target'

    def prepare(self):
        self.register(self.tag_wall, self.wall_shape, 1)
        self.register(self.tag_obstacle, self.obstacle_shape, self.obstacle_count)
        self.register(self.tag_target, self.target_shape, self.target_count)

    def get_freezone_poly(self):
        return self.get_inner_poly(self.tag_wall, unary_union(self.obstacle_hulls))

class SimpleSearchRoomGenerator(ObstacleRoomGenerator):
    target_each_count = 1
    target_sample_face = 0.01

    def __init__(self, 
                 obstacle_count=10,
                 obstacle_size=0.7,
                 target_size=0.2,
                 obstacle_zone_thresh=1.5,
                 room_length_max=9,
                 room_wall_thickness=0.05,
                 wall_threshold=0.1):

        super(SimpleSearchRoomGenerator, self).__init__(obstacle_count, obstacle_size, room_length_max, room_wall_thickness, wall_threshold)

        self.obstacle_zone_thresh = obstacle_zone_thresh
        self.target_size = target_size

    def generate_new(self):
        ## ------------------------- same super's ------------------------------------
        wall_shape = self._create_wall_poly()
        wall_collision = True
        wall_pos = np.array([(0,0,0)])

        wall_interior = Polygon(wall_shape.interiors[0])
        obstacle_xy, obstacle_yaw, obstacle_polys = sprinkle_cube(
            area_poly=wall_interior, 
            count=self.obstacle_count, 
            cube_size=self.obstacle_size,
            interior_thresh=self.wall_threshold
        )
        obstacle_shape = simple_cube(self.obstacle_size)
        obstacle_collision = True
        obstacle_pos = np.empty([len(obstacle_xy), 3])
        obstacle_pos[:,:2] = obstacle_xy
        obstacle_pos[:,2] = obstacle_yaw
        ## ---------------------------------------------------------------------------

        zone_polys, zone_hull = get_clustered_zones(obstacle_polys, self.obstacle_zone_thresh)
        target_shape = simple_cube(self.target_size)
        target_collision = False
        target_pos = self._sample_target_pos(zone_hull)

        room = SimpleSearchRoomConfig(
            wall_shape=wall_shape,
            obstacle_shape=obstacle_shape,
            target_shape=target_shape,
            obstacle_count=self.obstacle_count,
            target_count=len(target_pos),
            obstacle_hulls=zone_hull
        )

        room.prepare()
        room.set_config(room.tag_wall, wall_collision, wall_pos)
        room.set_config(room.tag_obstacle, obstacle_collision, obstacle_pos)
        room.set_config(room.tag_target, target_collision, target_pos)
        room.set_polygons_auto(room.tag_wall)
        room.set_polygons_direct(room.tag_obstacle, obstacle_polys)
        room.set_polygons_auto(room.tag_target)

        return room

    def _sample_target_pos(self, hulls):
        target_placing_hull = [h.buffer(self.wall_threshold) for h in hulls]
        target_pos = np.empty([len(hulls), 3])
        target_pos[:,:2] = sample_from_faces(
            polys=target_placing_hull, 
            count=self.target_each_count, 
            face_size=self.target_sample_face
        )[:,0]
        target_pos[:,2] = 0.0
        return target_pos
    
    def reposition_target(self, room_conf):
        target_pos = self._sample_target_pos(room_conf.obstacle_hulls)
        room_conf.set_config_positions(room_conf.tag_target, target_pos)
        room_conf.set_polygons_auto(room_conf.tag_target)