import numpy as np

class OrbitalPhysics:
    def __init__(self, orbital_radius=6771000): # Earth center to orbit (m)
        self.mu = 3.986004418e14  
        self.a = orbital_radius    
        self.n = np.sqrt(self.mu / self.a**3) # Mean motion

    def get_stm(self, dt):
        """
        Returns the 6x6 State Transition Matrix (Phi) for time step dt
        based on the Clohessy-Wiltshire equations.
        """
        n = self.n
        nt = n * dt
        sn = np.sin(nt)
        cn = np.cos(nt)
        
        # 1. Position from Position (Phi_rr)
        phi_rr = np.array([
            [4 - 3*cn,      0,   0],
            [6*(sn - nt),   1,   0],
            [0,             0,  cn]
        ])

        # 2. Position from Velocity (Phi_rv)
        phi_rv = np.array([
            [sn/n,           (2/n)*(1-cn),          0],
            [(2/n)*(cn-1),   (4*sn - 3*nt)/n,       0],
            [0,              0,                  sn/n]
        ])

        # 3. Velocity from Position (Phi_vr)
        phi_vr = np.array([
            [3*n*sn,        0,   0],
            [6*n*(cn - 1),  0,   0],
            [0,             0, -n*sn]
        ])

        # 4. Velocity from Velocity (Phi_vv)
        phi_vv = np.array([
            [cn,        2*sn,     0],
            [-2*sn,     4*cn - 3, 0],
            [0,         0,       cn]
        ])

        # Assemble the 6x6 matrix
        top = np.hstack((phi_rr, phi_rv))
        bottom = np.hstack((phi_vr, phi_vv))
        stm = np.vstack((top, bottom))
        
        return stm