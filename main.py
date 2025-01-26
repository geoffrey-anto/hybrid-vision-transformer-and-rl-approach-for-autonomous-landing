#!/usr/bin/env python
from src.simulation import Simulation

def main():
    print("Starting simulation...")
    sim = Simulation()
    
    print("Running simulation...")
    sim.run()

if __name__ == "__main__":
    main()