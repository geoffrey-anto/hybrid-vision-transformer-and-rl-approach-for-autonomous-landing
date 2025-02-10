import airsim
from typing import NewType
import config

State = NewType("State", tuple)

class Agent:
    def __init__(self, opts, *args, **kwargs):
        self.opts = opts
        self.gravity = opts.get("gravity")
        self.mass = opts.get("mass")
        
        self.client = airsim.MultirotorClient(ip=config.IP, port=config.PORT)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def start(self, initial_position):
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(initial_position[0], initial_position[1], initial_position[2], config.VELOCITY).join()
        return True
        
    def move(self, position):
        self.client.moveToPositionAsync(position[0], position[1], position[2], config.VELOCITY).join()
        return True
    
    def get_state(self):
        return self.client.getMultirotorState()
    
    def get_image(self):
        return self.client.simGetImage("0", airsim.ImageType.Scene)
    
    def get_lidar_point_cloud(self):
        return self.client.getLidarData()
    
    def land_drone(self):
        state = self.get_state()
        current_state = state.landed_state
        
        if current_state == airsim.LandedState.Landed:
            self.client.armDisarm(False)
            return True
        else:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            return True