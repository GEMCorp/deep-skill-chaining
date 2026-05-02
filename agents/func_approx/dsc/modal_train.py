import modal

# Define the image with your dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "numpy", 
    "scipy",
    "scikit-learn",
    "tensorboardX",
    "gymnasium",
    "matplotlib",
)

# Create the Modal app
app = modal.App("dsc-training", image=image)

# Mount your local code so Modal can access it
repo_mount = modal.Mount.from_local_dir(
    local_path="/home/elton-modestus/hrl_experiments/deep-skill-chaining",
    remote_path="/root/deep-skill-chaining",
)

@app.function(
    gpu="T4",  # or "A10G", "A100" for more power
    timeout=3600,  # 1 hour max
    mounts=[repo_mount],
)
def train_dsc(
    experiment_name: str,
    env: str = "Pendulum-v1",
    episodes: int = 100,
    steps: int = 500,
    use_ivf: bool = True,
    ivf_c: float = 0.5,
    ivf_step_penalty: float = 0.01,
    selection_mode: str = "categorical",
    seed: int = 0,
):
    """Single DSC training run on GPU."""
    import sys
    sys.path.insert(0, "/root/deep-skill-chaining")
    
    from simple_rl.tasks.gym.GymMDPClass import GymMDP
    from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Create MDP
    overall_mdp = GymMDP(env, render=False)
    overall_mdp.seed(seed)
    
    state_dim = overall_mdp.env.observation_space.shape[0]
    action_dim = overall_mdp.env.action_space.shape[0]
    
    # Create skill chaining agent
    chainer = SkillChaining(
        mdp=overall_mdp,
        max_steps=steps,
        lr_actor=1e-4,
        lr_critic=1e-3,
        ddpg_batch_size=64,
        device=device,
        max_num_options=5,
        subgoal_reward=0.0,
        enable_option_timeout=True,
        generate_plots=False,
        log_dir=f"/root/deep-skill-chaining/{experiment_name}",
        seed=seed,
        tensor_log=False,
        use_ivf=use_ivf,
        ivf_c=ivf_c,
        ivf_c1=0.2,
        ivf_c2=0.2,
        ivf_step_penalty=ivf_step_penalty,
        selection_mode=selection_mode,
    )
    
    # Train
    scores, durations = chainer.skill_chaining(episodes, steps)
    
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "final_score": float(scores[-1]) if scores else 0.0,
        "avg_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "episodes_completed": len(scores),
    }


@app.function(
    gpu="T4",
    timeout=7200,  # 2 hours for parallel runs
    mounts=[repo_mount],
)
def train_parallel_seeds(
    experiment_name: str,
    env: str = "Pendulum-v1",
    episodes: int = 100,
    steps: int = 500,
    seeds: list[int] = [0, 1, 2, 3, 4],
    use_ivf: bool = True,
):
    """Run multiple seeds in parallel using Modal's map."""
    results = []
    for seed in seeds:
        result = train_dsc.remote(
            experiment_name=f"{experiment_name}_seed{seed}",
            env=env,
            episodes=episodes,
            steps=steps,
            use_ivf=use_ivf,
            seed=seed,
        )
        results.append(result)
    return results


@app.local_entrypoint()
def main():
    """Local entrypoint to trigger Modal runs."""
    # Single run
    result = train_dsc.remote(
        experiment_name="ivf_modal_test",
        env="Pendulum-v1",
        episodes=5,
        steps=300,
        use_ivf=True,
    )
    print(f"Single run result: {result}")
    
    # Parallel seeds (uncomment to use)
    # results = train_parallel_seeds.remote(
    #     experiment_name="ivf_parallel",
    #     env="Pendulum-v1", 
    #     episodes=50,
    #     steps=500,
    #     seeds=[0, 1, 2, 3, 4],
    # )
    # print(f"Parallel results: {results}")