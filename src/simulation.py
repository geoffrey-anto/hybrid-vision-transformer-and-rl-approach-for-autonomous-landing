import src.agent as agent
import config

class Simulation:
    def __init__(self):
        self.agent = agent.Agent({
            "gravity": config.GRAVITY,
            "mass": config.MASS
        })
        
        print("Agent Initialized at State: ", self.agent.get_state())
    
    def run(self):
        pass