import argparse
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

with open("./src/configs/sysid_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEFAULT_DATA_PATH = config["dataset"]["raw_path"]
FIGURES_PATH = config["dataset"]["figures_path"]
os.makedirs(FIGURES_PATH, exist_ok=True)


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)
    required_keys = ["trajectories", "actions", "parameters"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in dataset: {missing}")

    trajectories = data["trajectories"]
    actions = data["actions"]
    parameters = data["parameters"]

    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectories with shape (N, T, obs_dim), got {trajectories.shape}")
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape (N, T, act_dim), got {actions.shape}")
    if parameters.ndim != 2:
        raise ValueError(f"Expected parameters with shape (N, param_dim), got {parameters.shape}")

    n_episodes = trajectories.shape[0]
    if actions.shape[0] != n_episodes or parameters.shape[0] != n_episodes:
        raise ValueError(
            "Mismatch in episode dimension among trajectories/actions/parameters: "
            f"{trajectories.shape}, {actions.shape}, {parameters.shape}"
        )

    return trajectories, actions, parameters


def sample_episode_indices(n_episodes: int, n_samples: int, rng: np.random.Generator):
    n = min(n_samples, n_episodes)
    return rng.choice(n_episodes, size=n, replace=False)


def plot_sampled_episodes(
    trajectories: np.ndarray,
    actions: np.ndarray,
    parameters: np.ndarray,
    episode_indices: np.ndarray,
    param_names: list,
):
    obs_dim = trajectories.shape[2]
    act_dim = actions.shape[2]
    n_rows = len(episode_indices)

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, max(3.5 * n_rows, 4)), squeeze=False)
    fig.suptitle("Sampled Episodes: Observations, Actions, and Dynamics", fontsize=14)

    for row, ep in enumerate(episode_indices):
        obs = trajectories[ep]
        act = actions[ep]
        prm = parameters[ep]
        t = np.arange(obs.shape[0])

        ax_obs = axes[row, 0]
        for j in range(obs_dim):
            ax_obs.plot(t, obs[:, j], linewidth=1.2, label=f"obs[{j}]")
        ax_obs.set_title(f"Episode {ep}: Observations")
        ax_obs.set_xlabel("timestep")
        ax_obs.set_ylabel("value")
        ax_obs.grid(alpha=0.25)
        ax_obs.legend(loc="best", fontsize=8)

        ax_act = axes[row, 1]
        for j in range(act_dim):
            ax_act.plot(t, act[:, j], linewidth=1.2, label=f"act[{j}]")
        ax_act.set_title(f"Episode {ep}: Actions")
        ax_act.set_xlabel("timestep")
        ax_act.set_ylabel("value")
        ax_act.grid(alpha=0.25)
        ax_act.legend(loc="best", fontsize=8)

        ax_mix = axes[row, 2]
        if obs_dim >= 2 and act_dim >= 1:
            sc = ax_mix.scatter(obs[:, 0], obs[:, 1], c=act[:, 0], s=8, cmap="viridis", alpha=0.85)
            fig.colorbar(sc, ax=ax_mix, fraction=0.046, pad=0.04, label="act[0]")
            ax_mix.set_xlabel("obs[0]")
            ax_mix.set_ylabel("obs[1]")
            ax_mix.set_title(f"Episode {ep}: Phase Plot (color=act[0])")
        else:
            ax_mix.plot(t, obs[:, 0], linewidth=1.2, label="obs[0]")
            if act_dim >= 1:
                ax_mix.plot(t, act[:, 0], linewidth=1.0, label="act[0]")
            ax_mix.set_xlabel("timestep")
            ax_mix.set_ylabel("value")
            ax_mix.set_title(f"Episode {ep}: Quick Overview")
            ax_mix.legend(loc="best", fontsize=8)

        prm_text = "\n".join([f"{param_names[k]} = {v:.4f}" for k, v in enumerate(prm)])
        ax_mix.text(
            1.02,
            0.98,
            prm_text,
            transform=ax_mix.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )
        ax_mix.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/sampled_episodes.png", dpi=300, bbox_inches="tight")


def plot_parameter_distributions(
    parameters: np.ndarray,
    param_names: list,
    sysid_bounds: dict,
):
    """
    Plots a scientific Corner Plot (Pairplot) of the system parameters.
    - Diagonal: Marginal histograms checking the uniform density property of the Sobol sequence [3].
    - Lower Triangle: Space-filling pairwise projections colored by sequence index to show dispersion over time [3].
    """
    n_params = parameters.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(2.2 * n_params, 2.2 * n_params), squeeze=False)
    fig.suptitle("Sobol Sequence space-filling & Distribution Diagnostics", fontsize=14, y=0.98)
    
    # Generate sequential coloring index to visually verify the Sobol space-filling progress [3]
    sequence_idx = np.arange(len(parameters))

    for r in range(n_params):
        for c in range(n_params):
            ax = axes[r, c]
            
            # 1. Hide Upper Triangle
            if c > r:
                ax.axis("off")
                continue
                
            # 2. Diagonal Plots (Marginal Histograms)
            if r == c:
                p_data = parameters[:, r]
                counts, bins, _ = ax.hist(p_data, bins=20, alpha=0.75, color="teal", edgecolor="white", linewidth=0.5)
                ax.grid(alpha=0.2)
                
                # Calculate Coefficient of Variation (CV = std/mean) of bin counts [3]
                # A low CV indicates highly uniform space-filling properties [3]
                cv = np.std(counts) / (np.mean(counts) + 1e-12)
                
                # Check for bounds in config to draw reference limits [3]
                p_name = param_names[r]
                if p_name in sysid_bounds:
                    b_min, b_max = sysid_bounds[p_name]
                    ax.axvline(b_min, color="crimson", linestyle="--", alpha=0.7, linewidth=1, label="Min Bound")
                    ax.axvline(b_max, color="crimson", linestyle="--", alpha=0.7, linewidth=1, label="Max Bound")
                    ax.set_xlim(b_min - 0.05 * (b_max - b_min), b_max + 0.05 * (b_max - b_min))
                
                ax.set_title(f"CV: {cv:.3f}", fontsize=8, pad=2)
                
            # 3. Off-Diagonal Plots (Pairwise Scatter Projections)
            else:
                x_data = parameters[:, c]
                y_data = parameters[:, r]
                
                sc = ax.scatter(
                    x_data, 
                    y_data, 
                    c=sequence_idx, 
                    cmap="plasma", 
                    s=2.0, 
                    alpha=0.6, 
                    rasterized=True
                )
                ax.grid(alpha=0.2)
                
                # Set axes boundaries based on config if possible [3]
                x_name, y_name = param_names[c], param_names[r]
                if x_name in sysid_bounds:
                    ax.set_xlim(sysid_bounds[x_name][0], sysid_bounds[x_name][1])
                if y_name in sysid_bounds:
                    ax.set_ylim(sysid_bounds[y_name][0], sysid_bounds[y_name][1])

            # Label Cleanups (Seaborn-style matrix wrapping)
            if r == n_params - 1:
                ax.set_xlabel(param_names[c], fontsize=9)
            else:
                ax.set_xticklabels([])
                
            if c == 0 and r > 0:
                ax.set_ylabel(param_names[r], fontsize=9)
            elif c > 0 or r == 0:
                ax.set_yticklabels([])

    # Add sequence progression colorbar on the right side
    fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.15)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="plasma"), 
        cax=cbar_ax, 
        label="Sobol Sequence Sample Generation Progress"
    )
    
    fig.savefig(f"{FIGURES_PATH}/parameter_distributions.png", dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw pendulum system-ID dataset")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to raw dataset NPZ file")
    parser.add_argument("--num-episodes", type=int, default=4, help="Number of episodes to plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trajectories, actions, parameters = load_dataset(args.data)

    print("Dataset loaded successfully")
    print(f"  trajectories shape: {trajectories.shape}")
    print(f"  actions shape:      {actions.shape}")
    print(f"  parameters shape:   {parameters.shape}")

    # Extract Parameter Names directly from config to avoid generic labels [3]
    sysid_bounds = config.get("sysid_bounds", {})
    param_names = list(sysid_bounds.keys())
    
    # Fallback to index names if config mapping doesn't match data dimension [3]
    if len(param_names) != parameters.shape[1]:
        param_names = [f"param_{i}" for i in range(parameters.shape[1])]

    ep_ids = sample_episode_indices(trajectories.shape[0], args.num_episodes, rng)
    plot_sampled_episodes(trajectories, actions, parameters, ep_ids, param_names)

    # Replaced transition sampling with the space-filling Corner Plot diagnostic [3]
    plot_parameter_distributions(parameters, param_names, sysid_bounds)
    print(f"Diagnostics plots saved to {FIGURES_PATH}/")

    plt.show()


if __name__ == "__main__":
    main()