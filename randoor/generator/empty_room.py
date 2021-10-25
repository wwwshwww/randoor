import numpy as np
import shapely
from shapely.geometry.polygon import Polygon

from ..generator_factory import RoomConfig, RoomGeneratorFactory

class EmptyRoomConfig(RoomConfig):
    def __init__(self, wall_polygon, wall_thickness, wall_height):
        super(EmptyRoomConfig, self).__init__()
        self.wall_polygon = wall_polygon
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height

        self.wall_collision_polygon = None
        self.wall_interior_polygon = None
        self.wall_exterior_polygon = None

        self.tag4config_wall = 'wall'

    @property
    def wall_pose(self):
        return self.spawn_config[self.tag4config_wall]

    def prepare(self):
        self.initialize_config(self.tag4config_wall)

    def set_wall_poly(self):
        exterior = self.wall_polygon.buffer(self.wall_thickness/2, join_style=2) # mitre style
        self.wall_exterior_polygon = 

class EmptyRoomGenerator(RoomGeneratorFactory):
    def __init__(self):
        pass