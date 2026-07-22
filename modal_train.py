import modal

# Define the image with your dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "/home/elton-modestus/hrl_experiments/deep-skill-chaining/requirements.txt"
).add_local_dir(
    "/home/elton-modestus/hrl_experiments/deep-skill-chaining",
    remote_path="/root/deep-skill-chaining",
    ignore=[
        ".git",
        "venv",
        "__pycache__",
        "runs",
        "value_function_plots",
        "initiation_set_plots",
        "*.pyc",
        "*.pyo",
    ],
)

# Create the Modal app
app = modal.App("dsc-training", image=image)

@app.function(
    gpu="T4",  # or "A10G", "A100" for more power
    timeout=3600,  # 1 hour max
)
def train_dsc(
    experiment_name: str,
    env: str = "Pendulum-v1",
    episodes: int = 100,
    steps: int = 500,
    ivf_c: float = 0.5,
    ivf_step_penalty: float = 0.01,
    seed: int = 0,
):
    """Single DSC training run on GPU."""
    import os
    import types
    import sys
    os.environ.setdefault("MPLBACKEND", "Agg")
    root = "/root/deep-skill-chaining"
    sys.path.insert(0, root)
    if "simple_rl" not in sys.modules:
        simple_rl_module = types.ModuleType("simple_rl")
        simple_rl_module.__package__ = "simple_rl"
        simple_rl_module.__path__ = [root]
        simple_rl_module.__file__ = f"{root}/__init__.py"
        sys.modules["simple_rl"] = simple_rl_module
    
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
        device=str(device),
        max_num_options=5,
        enable_option_timeout=True,
        generate_plots=False,
        log_dir=f"/root/deep-skill-chaining/{experiment_name}",
        seed=seed,
        tensor_log=False,
        ivf_c=ivf_c,
        ivf_c1=0.2,
        ivf_c2=0.2,
        ivf_step_penalty=ivf_step_penalty,
        experiment_name=experiment_name,
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
)
def train_parallel_seeds(
    experiment_name: str,
    env: str = "Pendulum-v1",
    episodes: int = 100,
    steps: int = 500,
    seeds: list[int] = [0, 1, 2, 3, 4],
):
    """Run multiple seeds in parallel using Modal's map."""
    results = []
    for seed in seeds:
        result = train_dsc.remote(
            experiment_name=f"{experiment_name}_seed{seed}",
            env=env,
            episodes=episodes,
            steps=steps,
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