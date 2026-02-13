import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OrbitalPhysics:
    def __init__(self, orbital_radius=6771000): 
        self.mu = 3.986004418e14  
        self.a = orbital_radius    
        self.n = np.sqrt(self.mu / self.a**3) 

    def get_stm(self, dt):
        """Returns the 6x6 Clohessy-Wiltshire State Transition Matrix."""
        n = self.n
        nt = n * dt
        sn, cn = np.sin(nt), np.cos(nt)
        
        phi_rr = np.array([
            [4 - 3*cn,      0,   0],
            [6*(sn - nt),   1,   0],
            [0,             0,  cn]
        ])

        phi_rv = np.array([
            [sn/n,           (2/n)*(1-cn),    0],
            [(2/n)*(cn-1),   (4*sn - 3*nt)/n, 0],
            [0,              0,            sn/n]
        ])

        phi_vr = np.array([
            [3*n*sn,        0,   0],
            [6*n*(cn - 1),  0,   0],
            [0,             0, -n*sn]
        ])

        phi_vv = np.array([
            [cn,        2*sn,     0],
            [-2*sn,     4*cn - 3, 0],
            [0,         0,       cn]
        ])

        return np.block([[phi_rr, phi_rv], [phi_vr, phi_vv]])

class OrbitAlignEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.physics = OrbitalPhysics()
        
        # RL Hyperparameters
        self.dt = 1.0
        self.max_acceleration = 0.5  # m/s^2
        self.docking_threshold = 0.5 # meters
        self.vel_threshold = 0.1     # m/s
        self.max_range = 5000        # meters
        self.max_steps = 1000
        
        # Gymnasium spaces (Position x,y,z, Velocity vx,vy,vz)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial state within 1km
        self.state = np.random.uniform(-1000, 1000, size=(6,)).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        thrust = action * self.max_acceleration 
        
        Phi = self.physics.get_stm(self.dt)
        
        # Discrete Control Matrix (Gamma)
        dt = self.dt
        Gamma = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])
        
        # State Transition
        self.state = (Phi @ self.state) + (Gamma @ thrust)
        
        # Metrics
        dist = np.linalg.norm(self.state[:3])
        vel = np.linalg.norm(self.state[3:])
        
        # --- Reward Shaping ---
        # 1. Quadratic penalty for distance and velocity (keeps it smooth)
        reward = -(0.01 * dist**2 + 0.1 * vel**2)
        # 2. Fuel penalty
        reward -= 0.05 * np.linalg.norm(action)
        # 3. Large bonus for successful docking
        terminated = bool(dist < self.docking_threshold and vel < self.vel_threshold)
        if terminated:
            reward += 1000
        
        truncated = bool(self.steps >= self.max_steps or dist > self.max_range)
        
        return self.state, float(reward), terminated, truncated, {"dist": dist, "vel": vel}