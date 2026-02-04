import numpy as np

class OrbitalPhysics:
    def __init__(self, orbital_radius=6771000): # Radius in meters
    self.mu = 3.986004418e14  # Earth's gravitational parameter
    self.a = orbital_radius    # Distance from Earth center
    self.n = np.sqrt(self.mu / self.a**3) # Mean motion (angular velocity)

    def get_stm(self, dt):
        """
        Returns the 6x6 State Transition Matrix (Phi) for time step dt.
        This represents the natural drift of the spacecraft.
        """