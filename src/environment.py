import mujoco
import mujoco.viewer
import gymnasium
import numpy as np

def simp_angle(a):
    """
    Wrap angle to [-pi, pi]
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

class SinglePendulumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    FRAME_SKIP = 20
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path("../assets/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None
        self.timesteps = 0
        self.max_episode_steps = 600  # 12 seconds
        
        # Environment
        self.actuator_scale = 3.0
        # Let's keep it at 64 bits
        # raw theta (goes beyond -pi and pi), angular velocity
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timesteps = 0
        if seed is not None:
            np.random.seed(seed)
        
        # If options are provided then update parameters otherwise use default
        if options is not None:
            self.model.dof_damping[0] = options["parameters"][0]
            self.model.dof_frictionloss[0] = options["parameters"][1]
            self.model.dof_armature[0] = options["parameters"][2]

        # Reset Physics
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.pi  # pole starts at the bottom (equilibrium)
        self.data.qvel[0] = 0.0
        
        return self._get_obs_info()
    
    def _get_obs_info(self):
        current_parameters = np.array([
            self.model.dof_damping[0], self.model.dof_frictionloss[0], self.model.dof_armature[0]
        ], dtype=np.float64)
        
        info = {"qpos": self.data.qpos[0], "qvel": self.data.qvel[0], "parameters": current_parameters}
        obs = np.array([self.data.qpos[0], self.data.qvel[0]], dtype=np.float64)
        
        return obs, info
    
    def step(self, action):
        self.data.ctrl = action * self.actuator_scale

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        
        self.timesteps += 1
        
        truncated = bool(self.timesteps >= self.max_episode_steps)
        # Reward is useless for this application
        reward = 0.0

        terminated = False

        obs, info = self._get_obs_info()
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
    # Script used to save video and the stroboscopic effect image for the report.
    # You need to install imageio for saving videos: pip install imageio[ffmpeg]
    # Change the scene to 'white_scene.xml' since it has "main_cam" and white background
    import imageio
    import numpy as np
    
    def create_stroboscopic_overlay(frames, start_idx=0, end_idx=500, step=5):
        """
        Creates a clean stroboscopic effect using Minimum Intensity Blending.
        """
        # 1. Slice the frame range you actually want
        selected_frames = frames[start_idx:end_idx:step]
        
        # 2. Start with the background (first frame)
        # Convert to float for math, then back to uint8 at the end
        result = selected_frames[0].astype(np.float32)
        
        for i in range(1, len(selected_frames)):
            frame = selected_frames[i].astype(np.float32)
            
            # --- THE FIX: MINIMUM BLENDING ---
            # On a white background, this keeps the darkest pixels (the pole) 
            # from every frame without color distortion.
            result = np.minimum(result, frame)
            
            # OPTIONAL: To make the "older" versions look fainter, 
            # you can slightly lighten the previous 'result' before the next minimum.
            result = result * 0.98 + 5 # Uncomment to experiment with fading
            
        return result.astype(np.uint8)
    
    # 1. Initialize env with rgb_array mode
    env = SinglePendulumEnv(render_mode="rgb_array")
    
    test_params = [2.0, 1.0, 0.4] 
    obs, info = env.reset(seed=42, options={"parameters": test_params})
    
    frames = []
    done = False
    print("Recording episode...")

    while not done:
        # --- CONTROL LOGIC ---
        if 10 <= env.timesteps < 20:
            action = np.array([1.0])
        elif 300 <= env.timesteps < 310:
            action = np.array([-1.0])
        else:
            action = np.array([0.0])

        # --- STEP ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- RENDER AND CAPTURE ---
        # Capture frame (returns a numpy array)
        frame = env.render()
        frames.append(frame)

    env.close()

    # 2. Save the video
    video_path = "../figures/pendulum_trajectory.mp4"
    print(f"Saving video to {video_path}...")
    
    # fps is 50 because your frame_skip=20 at 0.001s means 20ms steps (50Hz)
    imageio.mimsave(video_path, frames, fps=50)
    
    overlay_img = create_stroboscopic_overlay(frames)
    imageio.imwrite("../figures/stroboscopic_figure.png", overlay_img)
    print("Done!")
