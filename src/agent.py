from typing import NewType

State = NewType("State", tuple)

class Agent:
    def __init__(self, opts, *args, **kwargs):
        # state = (x, y, z, vx, vy, vz, pitch, yaw, roll)
        if kwargs.get("state"):
            self.state = State(kwargs.get("state"))
        else:
            self.state = State((0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.opts = opts
        self.gravity = opts.get("gravity")
        self.mass = opts.get("mass")
        
    def move(self):
        pass
    
    def get_state(self):
        return self.state

    def act(self):
        pass

    def learn(self):
        pass

    def save(self):
        pass

    def load(self):
        pass