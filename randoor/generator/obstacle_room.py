from shapely.geometry.polygon import Polygon
import numpy as np

from ..spawner.poly import sprinkle_cube, simple_cube
from .empty_room import EmptyRoomConfig, EmptyRoomGenerator

class ObstacleRoomConfig(EmptyRoomConfig):
    def __init__(self, wall_shape, obstacle_shape, obstacle_count):
        super(ObstacleRoomConfig, self).__init__(wall_shape)

        self.obstacle_shape = obstacle_shape
        self.obstacle_count = obstacle_count

        self.tag_wall = 'wall'
        self.tag_obstacle = 'obstacle'

    def prepare(self):
        self.register(self.tag_wall, self.wall_shape, 1)
        self.register(self.tag_obstacle, self.obstacle_shape, self.obstacle_count)

class ObstacleRoomGenerator(EmptyRoomGenerator):
    def __init__(self, 
                 obstacle_count=10,
                 obstacle_size=0.7,
                 room_length_max=9,
                 room_wall_thickness=0.05,
                 wall_threshold=0.1):
        
        super(ObstacleRoomGenerator, self).__init__(room_length_max, room_wall_thickness)
        self.wall_threshold = wall_threshold
        self.obstacle_count = obstacle_count
        self.obstacle_size = obstacle_size

    def generate_new(self):
        wall_shape = self._create_wall_poly()
        wall_collision = [True]
        wall_pos = np.array([(0,0,0)])

        wall_interior = Polygon(wall_shape.interiors[0])
        xy, yaw, polys = sprinkle_cube(
            area_poly=wall_interior, 
            count=self.obstacle_count, 
            cube_size=self.obstacle_size,
            interior_thresh=self.wall_threshold
        )
        obstacle_shape = simple_cube(self.obstacle_size)
        obstacle_collision = [True for _ in range(self.obstacle_count)]
        obstacle_pos = np.empty([len(xy), 3])
        obstacle_pos[:,:2] = xy
        obstacle_pos[:,2] = yaw

        obs_room = ObstacleRoomConfig(
            wall_shape=wall_shape,
            obstacle_shape=obstacle_shape,
            obstacle_count=self.obstacle_count
        )
        obs_room.prepare()
        obs_room.set_config(obs_room.tag_wall, wall_collision, wall_pos)
        obs_room.set_config(obs_room.tag_obstacle, obstacle_collision, obstacle_pos)
        obs_room.set_polygons_auto(obs_room.tag_wall)
        obs_room.set_polygons_direct(obs_room.tag_obstacle, polys)

        return obs_room
