from setuptools import setup

# Map the workspace layout to the simple_rl package namespace.
package_dir = {
    "simple_rl": ".",
    # Core subpackages
    "simple_rl.abstraction": "abstraction",
    "simple_rl.abstraction.action_abs": "abstraction/action_abs",
    "simple_rl.abstraction.state_abs": "abstraction/state_abs",
    "simple_rl.agents": "agents",
    "simple_rl.agents.bandits": "agents/bandits",
    "simple_rl.agents.func_approx": "agents/func_approx",
    "simple_rl.agents.func_approx.dqn": "agents/func_approx/dqn",
    "simple_rl.agents.func_approx.ddpg": "agents/func_approx/ddpg",
    "simple_rl.agents.func_approx.dsc": "agents/func_approx/dsc",
    "simple_rl.experiments": "experiments",
    "simple_rl.mdp": "mdp",
    "simple_rl.mdp.markov_game": "mdp/markov_game",
    "simple_rl.mdp.oomdp": "mdp/oomdp",
    "simple_rl.planning": "planning",
    "simple_rl.pomdp": "pomdp",
    "simple_rl.tasks": "tasks",
    # Commonly used tasks in this repo
    "simple_rl.tasks.gym": "tasks/gym",
    "simple_rl.tasks.point_env": "tasks/point_env",
    "simple_rl.tasks.point_maze": "tasks/point_maze",
    "simple_rl.tasks.dm_fixed_reacher": "tasks/dm_fixed_reacher",
}

setup(
    name="simple_rl",
    version="0.0.0+local",
    description="Local editable install of simple_rl from deep-skill-chaining workspace",
    packages=list(package_dir.keys()),
    package_dir=package_dir,
    include_package_data=True,
)
