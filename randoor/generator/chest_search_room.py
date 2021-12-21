from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import numpy as np

from randoor.spawner.poly import simple_cube

from ..spawner.geom import sample_sure
from .simple_search_room import SimpleSearchRoomGenerator, SimpleSearchRoomConfig

class ChestSearchRoomConfig(SimpleSearchRoomConfig):
    def __init__(self, 
                 wall_shape, 
                 obstacle_shape, 
                 target_shape, 
                 key_shape, 
                 obstacle_count, 
                 target_count, 
                 key_count, 
                 obstacle_hulls, 
                 key_placing_area):

        super(ChestSearchRoomConfig, self).__init__(wall_shape, obstacle_shape, target_shape, obstacle_count, target_count, obstacle_hulls)

        self.key_shape = key_shape
        self.key_count = key_count
        self.key_placing_area = key_placing_area

        self.tag_wall = 'wall'
        self.tag_obstacle = 'obstacle'
        self.tag_target = 'target'
        self.tag_key = 'key'

    def tweak_key_collision(self, index, is_col=True):
        self.tweak_collision(self.tag_key, index, is_col)

    def prepare(self):
        self.register(self.tag_wall, self.wall_shape, 1)
        self.register(self.tag_obstacle, self.obstacle_shape, self.obstacle_count)
        self.register(self.tag_target, self.target_shape, self.target_count)
        self.register(self.tag_key, self.key_shape, self.key_count)

    def get_freezone_poly(self):
        return self.get_inner_poly(self.tag_wall, unary_union(self.obstacle_hulls))

class ChestSearchRoomGenerator(SimpleSearchRoomGenerator):
    key_each_count = 1

    def __init__(self, 
                 obstacle_count=10,
                 obstacle_size=0.7,
                 target_size=0.2,
                 key_size=0.2,
                 obstacle_zone_thresh=1.5,
                 distance_key_placing=0.7,
                 range_key_placing=0.3,
                 room_length_max=9,
                 room_wall_thickness=0.05,
                 wall_threshold=0.1):

        super(ChestSearchRoomGenerator, self).__init__(obstacle_count, obstacle_size, target_size, obstacle_zone_thresh, room_length_max, room_wall_thickness, wall_threshold)

        self.key_size = key_size
        self.distance_key_placing = distance_key_placing
        self.range_key_placing = range_key_placing

    def generate_new(self):
        pre =  super(ChestSearchRoomGenerator, self).generate_new()
        freezone = pre.get_freezone_poly()
        zone_hull = pre.obstacle_hulls
        hull_buff = self.distance_key_placing + self.range_key_placing
        path_area = freezone.buffer(-self.distance_key_placing)
        key_placing_area = [path_area.intersection(h.buffer(hull_buff)) for h in zone_hull]

        key_shape = simple_cube(self.key_size)
        key_pos = self._sample_key_pos(key_placing_area)
        key_collision = [False for _ in range(len(key_pos))]

        room = ChestSearchRoomConfig(
            wall_shape=pre.wall_shape,
            obstacle_shape=pre.obstacle_shape,
            target_shape=pre.target_shape,
            key_shape=key_shape,
            obstacle_count=pre.obstacle_count,
            target_count=pre.target_count,
            key_count=len(key_pos),
            obstacle_hulls=pre.obstacle_hulls,
            key_placing_area=key_placing_area
        )

        room.prepare()
        self.merge_config(pre, room)
        room.set_config(room.tag_key, key_collision, key_pos)
        room.set_polygons_auto(room.tag_key)

        return room

    def _sample_key_pos(self, key_placing_areas):
        key_pos = np.empty([len(key_placing_areas), 3])
        key_pos[:,:2] = [sample_sure(a, self.key_each_count, self.range_key_placing)[0] for a in key_placing_areas]
        key_pos[:,2] = 0.0
        return key_pos

    def reposition_key(self, room_conf):
        key_pos = self._sample_key_pos(room_conf.key_placing_area)
        room_conf.set_config_positions(room_conf.tag_key, key_pos)
        room_conf.set_polygons_auto(room_conf.tag_key)