import numpy as np

class WarmUpActionPolicy:
    """_summary_
    Just produce random actions for the environment for system identification. It has multiple functions.
    1. White Noise - Just pure random actions
    2. Pseudo-Random Binary Sequence (PRBS) - PRBS is a classic SysID signal that randomly jumps between a maximum and minimum value (e.g., -1 and 1) at discrete time intervals.
    3. Chirp Signal - A chirp is a sine wave that continuously increases (or decreases) in frequency over time.
    4. Multisine (Sum of Sines) - A combination of multiple sine waves at different frequencies and different phases
    5. Impulses - Short bursts of maximum amplitude followed by periods of no input
    I do 1 for each unique set of system parameters
    """
    SIGNAL_TYPES = ["noise", "prbs", "multisine", "impulse", "chirp_normal", "chirp_shifted"]
    NUM_SIGNAL_TYPES = len(SIGNAL_TYPES)
    def __init__(self, action_space, total_timesteps, dt, frame_skip):
        self.chosen_signal_type = None
        self.dt = dt
        self.frame_skip = frame_skip
        self.sampling_rate = self.dt * self.frame_skip  # Effective sampling rate after frame skipping
        self.total_timesteps = total_timesteps
        self.total_time = self.sampling_rate * total_timesteps  # Total time per episode
        self.t = np.linspace(0, self.total_time, self.total_timesteps)
        self.action_space = action_space
        self.action_dims = self.action_space.shape[0]
        
        # Precompute the global multisine signal attributes
        self.base_freq = 1 / (self.total_timesteps * self.sampling_rate)  # Fundamental frequency based on total episode length
        self.max_freq = 10.0  # According to physical bandwidth of STS3215 motor. The point where the motor's response drops off
        self.frequencies = np.linspace(self.base_freq, self.max_freq, num=8)  # 8 frequencies evenly spaced on a log scale
    
    def reset(self, variation_idx, seed=None):
        if variation_idx < 0 or variation_idx >= 10:
            raise ValueError("Invalid variation index. Choose from 0 to 9")
        
        # Map the 10 sequential variations back to the 6 physical SIGNAL_TYPES
        if variation_idx < 4:
            signal_type_idx = variation_idx
        elif variation_idx < 6:
            signal_type_idx = 4  # chirp_normal (sampled 2 times total)
        else:
            signal_type_idx = 5  # chirp_shifted (sampled 4 times total)
        self.chosen_signal_type = self.SIGNAL_TYPES[signal_type_idx]
        
        # Create a isolated local generator for perfect multi-process determinism
        rng = np.random.default_rng(seed)

        if self.chosen_signal_type == "multisine":
            self.multisine_function = np.zeros((self.total_timesteps, self.action_dims))
            for act_idx in range(self.action_dims):
                for freq in self.frequencies:
                    phase = rng.uniform(0, 2 * np.pi)  # Switched to local rng
                    self.multisine_function[:, act_idx] += np.sin(2 * np.pi * freq * self.t + phase)
                self.multisine_function /= len(self.frequencies)
                
        elif self.chosen_signal_type in ["chirp_normal", "chirp_shifted"]:
            self.chirp_function = np.zeros((self.total_timesteps, self.action_dims))
            for act_idx in range(self.action_dims):
                if self.chosen_signal_type == "chirp_normal":
                    frequencies = [(0, 10), (0, -10), (-10, 0), (10, 0)]
                    chosen_frequency_pair = frequencies[rng.choice(len(frequencies))]  # Switched to local rng
                    f1, f2 = chosen_frequency_pair
                elif self.chosen_signal_type == "chirp_shifted":
                    f1 = rng.uniform() * self.max_freq  # Switched to local rng
                    if f1 > 0.0:
                        f2 = rng.uniform(-1.0, 0.0) * self.max_freq  # Switched to local rng
                    else:
                        f2 = rng.uniform(0.0, 1.0) * self.max_freq  # Switched to local rng
                a = (f2 - f1)/(2 * self.total_time)
                chirp = np.sin(2 * np.pi * (f1 + a * self.t) * self.t)
                self.chirp_function[:, act_idx] = chirp
                
        elif self.chosen_signal_type == "prbs":
            prbs = rng.uniform(size=(self.total_timesteps, self.action_dims))  # Switched to local rng
            prbs = (prbs > 0.5).astype(float) * 2 - 1
            self.prbs_function = prbs
                
    def impulse_signal(self, timestep):
        half_max_steps = self.total_timesteps // 2
        if 5 <= timestep < 15:
            # Torque pulse positive at 100ms upto 300ms
            action = np.ones(self.action_space.shape)
        elif half_max_steps + 5 <= timestep < half_max_steps + 15:
            # Torque pulse negative at 6.1s upto 6.3s
            action = -np.ones(self.action_space.shape)
        else:
            # let it settle for about 10 seconds
            action = np.zeros(self.action_space.shape)
        return action
    
    def chirp_signal(self, timestep):
        return self.chirp_function[timestep, :]
    
    def prbs_signal(self, timestep):
        return self.prbs_function[timestep, :]
        
    def multisine_signal(self, timestep):
        return self.multisine_function[timestep, :]

    def act(self, timestep):
        """
        Args:
            timestep (_type_): Takes in the environment timestep
        """
        if self.chosen_signal_type == "noise":
            action = self.action_space.sample()
        elif self.chosen_signal_type == "prbs":
            action = self.prbs_signal(timestep)
        elif self.chosen_signal_type in ["chirp_normal", "chirp_shifted"]:
            action = self.chirp_signal(timestep)
        elif self.chosen_signal_type == "multisine":
            action = self.multisine_signal(timestep)
        elif self.chosen_signal_type == "impulse":
            action = self.impulse_signal(timestep)
        return action
