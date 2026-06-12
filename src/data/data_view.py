import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_PATH = "./dataset/raw_pendulum_sysid_dataset.npz"


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


def sample_transitions(
    trajectories: np.ndarray,
    actions: np.ndarray,
    parameters: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
):
    n_episodes, horizon, _ = trajectories.shape
    total_transitions = n_episodes * horizon
    n = min(n_samples, total_transitions)

    flat_ids = rng.choice(total_transitions, size=n, replace=False)
    ep_ids = flat_ids // horizon
    t_ids = flat_ids % horizon

    obs_samples = trajectories[ep_ids, t_ids, :]
    act_samples = actions[ep_ids, t_ids, :]
    param_samples = parameters[ep_ids, :]

    return ep_ids, t_ids, obs_samples, act_samples, param_samples


def plot_sampled_episodes(
    trajectories: np.ndarray,
    actions: np.ndarray,
    parameters: np.ndarray,
    episode_indices: np.ndarray,
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

        prm_text = "\n".join([f"param[{k}] = {v:.4f}" for k, v in enumerate(prm)])
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


def plot_transition_samples(
    ep_ids: np.ndarray,
    t_ids: np.ndarray,
    obs_samples: np.ndarray,
    act_samples: np.ndarray,
    param_samples: np.ndarray,
):
    obs_dim = obs_samples.shape[1]
    act_dim = act_samples.shape[1]
    prm_dim = param_samples.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Random Transition Samples", fontsize=14)

    ax1 = axes[0, 0]
    if obs_dim >= 2 and act_dim >= 1:
        sc = ax1.scatter(obs_samples[:, 0], obs_samples[:, 1], c=act_samples[:, 0], cmap="plasma", alpha=0.85)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="act[0]")
        ax1.set_xlabel("obs[0]")
        ax1.set_ylabel("obs[1]")
        ax1.set_title("obs[0] vs obs[1] (color=act[0])")
    else:
        ax1.scatter(np.arange(len(obs_samples)), obs_samples[:, 0], s=12, alpha=0.85)
        ax1.set_xlabel("sample index")
        ax1.set_ylabel("obs[0]")
        ax1.set_title("obs[0] sampled transitions")
    ax1.grid(alpha=0.25)

    ax2 = axes[0, 1]
    y_action = act_samples[:, 0] if act_dim >= 1 else np.zeros(len(act_samples))
    c_param = param_samples[:, 0] if prm_dim >= 1 else np.zeros(len(param_samples))
    sc = ax2.scatter(t_ids, y_action, c=c_param, cmap="viridis", alpha=0.85)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, label="param[0]")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("act[0]" if act_dim >= 1 else "action")
    ax2.set_title("Action over sampled timesteps (color=param[0])")
    ax2.grid(alpha=0.25)

    ax3 = axes[1, 0]
    x = obs_samples[:, 0]
    y = y_action
    c = param_samples[:, 1] if prm_dim >= 2 else c_param
    sc = ax3.scatter(x, y, c=c, cmap="cividis", alpha=0.85)
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="param[1]" if prm_dim >= 2 else "param[0]")
    ax3.set_xlabel("obs[0]")
    ax3.set_ylabel("act[0]" if act_dim >= 1 else "action")
    ax3.set_title("obs[0] vs action (color=system parameter)")
    ax3.grid(alpha=0.25)

    ax4 = axes[1, 1]
    if prm_dim >= 3:
        sizes = 20 + 80 * (param_samples[:, 2] - param_samples[:, 2].min()) / (
            (np.ptp(param_samples[:, 2]) + 1e-12)
        )
        ax4.scatter(param_samples[:, 0], param_samples[:, 1], s=sizes, alpha=0.75)
        ax4.set_xlabel("param[0]")
        ax4.set_ylabel("param[1]")
        ax4.set_title("param[0] vs param[1] (size=param[2])")
    elif prm_dim == 2:
        ax4.scatter(param_samples[:, 0], param_samples[:, 1], alpha=0.75)
        ax4.set_xlabel("param[0]")
        ax4.set_ylabel("param[1]")
        ax4.set_title("param[0] vs param[1]")
    else:
        ax4.hist(param_samples[:, 0], bins=25, alpha=0.8)
        ax4.set_xlabel("param[0]")
        ax4.set_ylabel("count")
        ax4.set_title("param[0] distribution")
    ax4.grid(alpha=0.25)

    # Annotate with sampled episode IDs for traceability.
    info = f"Unique sampled episodes: {len(np.unique(ep_ids))}\nTotal transitions shown: {len(ep_ids)}"
    ax4.text(
        1.02,
        0.02,
        info,
        transform=ax4.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Visualize raw pendulum system-ID dataset")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to raw dataset NPZ file")
    parser.add_argument("--num-episodes", type=int, default=4, help="Number of episodes to plot")
    parser.add_argument(
        "--num-transitions", type=int, default=2000, help="Number of random transitions to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trajectories, actions, parameters = load_dataset(args.data)

    print("Dataset loaded successfully")
    print(f"  trajectories shape: {trajectories.shape}")
    print(f"  actions shape:      {actions.shape}")
    print(f"  parameters shape:   {parameters.shape}")

    ep_ids = sample_episode_indices(trajectories.shape[0], args.num_episodes, rng)
    plot_sampled_episodes(trajectories, actions, parameters, ep_ids)

    transition_pack = sample_transitions(
        trajectories,
        actions,
        parameters,
        n_samples=args.num_transitions,
        rng=rng,
    )
    plot_transition_samples(*transition_pack)

    plt.show()


if __name__ == "__main__":
    main()
