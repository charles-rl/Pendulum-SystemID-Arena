import mujoco
import mujoco.viewer
import gymnasium
import numpy as np


class SinglePendulumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    FRAME_SKIP = 10
    # We initialize it here so that no need to create a dummy env, we skip __init__
    MAX_EPISODE_STEPS = 300  # 3 seconds
    
    def __init__(self, render_mode=None, track_targets=True, print_info=False):
        self.render_mode = render_mode
        self.track_targets = track_targets
        self.print_info = print_info

        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path("./assets/scene.xml")
        self.data = mujoco.MjData(self.model)
        # Name of your joint
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor")
        # Also name of your joint (?)
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor")
        # Get the DOF address for this joint
        self.jnt_dof_adr = self.model.jnt_dofadr[self.joint_id]
        # Name of backlash joint as well
        self.backlash_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "backlash_joint")
        # Get the DOF address for the backlash joint
        self.backlash_dof_adr = self.model.jnt_dofadr[self.backlash_id]
        # To change pole stuff
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cpole")
        
        # Add these positional address registers
        self.jnt_qpos_adr = self.model.jnt_qposadr[self.joint_id]
        self.backlash_qpos_adr = self.model.jnt_qposadr[self.backlash_id]
        
        self.viewer = None
        self.renderer = None
        self.timesteps = 0
        
        # Environment
        # Scale to actual positions which in this case is -pi, pi
        self.actuator_scale = np.pi
        self.target_angle = None
        self.prev_action = np.zeros((1,), dtype=np.float64) # Added to track action smoothness
        # Let's keep it at 64 bits
        # raw theta (goes beyond -pi and pi), angular velocity
        observation_dims = 3 if self.track_targets else 2
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dims,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timesteps = 0
        if seed is not None:
            np.random.seed(seed)
        
        # If options are provided then update parameters otherwise use default
        if options is not None:
            p = options["params"]
            # TODO: Need a way to return to default if system parameter was passed
            self.model.actuator_gainprm[self.actuator_id, 0] = p["kp"]
            self.model.actuator_biasprm[self.actuator_id, 1] = -p["kp"]
            self.model.actuator_biasprm[self.actuator_id, 2] = -p["kv"]
            self.model.actuator_forcerange[self.actuator_id] = [-p["max_torque"], p["max_torque"]]
            
            self.model.dof_frictionloss[self.jnt_dof_adr] = p["frictionloss"]
            self.model.dof_damping[self.jnt_dof_adr] = p["damping"]
            self.model.dof_armature[self.jnt_dof_adr] = p["armature"]
            
            backlash_max_rad = np.deg2rad(p["backlash_max_deg"])
            self.model.jnt_range[self.backlash_id] = [-backlash_max_rad, backlash_max_rad]
            self.model.dof_damping[self.backlash_dof_adr] = p["backlash_damping"]
            self.model.dof_armature[self.backlash_dof_adr] = p["backlash_armature"]

        # Reset Physics
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.jnt_qpos_adr] = np.random.uniform(-np.pi, np.pi)  # random initial angle
        self.data.qvel[self.jnt_dof_adr] = np.random.uniform(-0.2, 0.2)  # small initial angular velocity
        
        if self.track_targets:
            # Set target angle
            self.target_angle = np.random.uniform(-np.pi, np.pi)
        # Set action to current position
        self.data.ctrl = self.data.qpos[self.jnt_qpos_adr]
        self.prev_action = self.data.ctrl.copy()
        if self.print_info:
            print(f"Resetting Environment: Target Angle = {self.target_angle:.3f} radians")
            print(f"Initial Pole Angle = {self.data.qpos[self.jnt_qpos_adr]:.3f} radians, Initial Velocity = {self.data.qvel[self.jnt_dof_adr]:.3f} rad/s")
            print(f"Action Control: {self.data.ctrl[0]:.3f}")
        
        return self._get_obs_info()
    
    def _get_obs_info(self):
        # Calculate TRUE mechanical positions by summing motor and backlash elements
        # For STS3215 motor, the encoder is mounted at the FINAL output shaft, so it measures the combined effect of the motor joint and the backlash joint
        pole_angle = self.data.qpos[self.jnt_qpos_adr] + self.data.qpos[self.backlash_qpos_adr]
        pole_velocity = self.data.qvel[self.jnt_dof_adr] + self.data.qvel[self.backlash_dof_adr]
        
        current_parameters = {
            "kp": self.model.actuator_gainprm[self.actuator_id, 0],
            "kv": self.model.actuator_biasprm[self.actuator_id, 2],
            "max_torque": self.model.actuator_forcerange[self.actuator_id, 1],
            "frictionloss": self.model.dof_frictionloss[self.jnt_dof_adr],
            "damping": self.model.dof_damping[self.jnt_dof_adr],
            "armature": self.model.dof_armature[self.jnt_dof_adr],
            "backlash_range": self.model.jnt_range[self.backlash_id],
            "backlash_armature": self.model.dof_armature[self.backlash_dof_adr],
            "backlash_damping": self.model.dof_damping[self.backlash_dof_adr]
        }
        
        info = {"pole_angle": pole_angle, "pole_velocity": pole_velocity, "parameters": current_parameters}
        # Scale angle to be between -1 and 1
        scaled_pole_angle = (pole_angle + np.pi) / (2 * np.pi) * 2 - 1
        # Scale velocity according to 45 RPM max speed of the motor (converted to rad/s)
        max_velocity = (45 / 60) * 2 * np.pi # Convert 45 RPM to rad/s
        scaled_pole_velocity = pole_velocity / max_velocity
        if self.track_targets:
            # Scale target angle as well
            scaled_target_angle = (self.target_angle + np.pi) / (2 * np.pi) * 2 - 1
            obs = np.array([scaled_pole_angle, scaled_pole_velocity, scaled_target_angle], dtype=np.float64)
        else:
            obs = np.array([scaled_pole_angle, scaled_pole_velocity], dtype=np.float64)
        
        return obs, info
    
    def step(self, action):
        self.data.ctrl = action * self.actuator_scale

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        
        self.timesteps += 1
        
        if self.track_targets and self.timesteps == (self.MAX_EPISODE_STEPS // 2):
            # Set target angle
            self.target_angle = np.random.uniform(-np.pi, np.pi)
        
        truncated = bool(self.timesteps >= self.MAX_EPISODE_STEPS)
        terminated = False
        obs, info = self._get_obs_info()
        
        if self.track_targets:
            # --- DR-SENSITIVE REWARD CALCULATION ---
            pole_angle = info["pole_angle"]
            pole_velocity = info["pole_velocity"]
            
            # 1. Normalized Position Error (handles angle wrapping bugs gracefully)
            angle_error = self.target_angle - pole_angle
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            r_pos = np.exp(-2.0 * np.abs(angle_error))
            
            # 2. Action Smoothness (Penalizes changes in command over time)
            r_smooth = -np.square(action[0] - self.prev_action[0])
            
            # 3. Energy Penalty (Reads absolute torque force applied by motor constraint)
            r_energy = -np.square(self.data.actuator_force[self.actuator_id])
            
            # 4. Stabilization Reward (Damps out velocity oscillation right at the goal)
            r_stability = -np.square(pole_velocity) if np.abs(angle_error) < 0.05 else 0.0
            
            r_pos = 1.0 * r_pos
            r_smooth = 0.2 * r_smooth
            r_energy = 0.01 * r_energy
            r_stability = 0.05 * r_stability
            
            # Composite reward configuration
            reward = r_pos + r_smooth + r_energy + r_stability
        else:
            reward = 0.0  # No reward if not tracking targets
        
        if self.print_info:
            print(f"Step: {self.timesteps}, Angle Error: {angle_error:.3f}, r_pos: {r_pos:.3f}, r_smooth: {r_smooth:.6f}, r_energy: {r_energy:.3f}, r_stability: {r_stability:.6f}, Total Reward: {reward:.3f}")
        
        # Store state for next cycle smoothness calculation
        self.prev_action = np.copy(action)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            # 1. Launch Viewer if it doesn't exist yet
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False)

            if self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                self.renderer.update_scene(self.data, camera="main_cam")
            else:
                self.renderer.update_scene(self.data, camera="main_cam")

            return self.renderer.render()
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()



if __name__ == "__main__":
    import time
    # Define test parameters dict matching the environment expected options schema
    test_params = {
        "kp": 998.22,
        "kv": 2.731,
        "max_torque": 2.94,          # Set nominal (2.94) or peak voltage performance (3.35)
        "frictionloss": 0.052,
        "damping": 0.6,
        "armature": 0.028,
        "backlash_max_deg": 0.87,
        "backlash_armature": 0.01,
        "backlash_damping": 0.01,
    }

    env = SinglePendulumEnv(render_mode="human", print_info=True)
    
    # Pass params via the matching string dictionary key 
    obs, info = env.reset(seed=42, options={"params": test_params})
    done = False

    while not done:
        action = np.array([env.target_angle / env.actuator_scale])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        time.sleep(env.model.opt.timestep * env.FRAME_SKIP)

    env.close()
