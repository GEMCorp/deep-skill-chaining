# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

from collections import deque, defaultdict
from copy import deepcopy
import pdb
import argparse
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.utils import *
from simple_rl.agents.func_approx.ddpg.utils import *
from simple_rl.agents.func_approx.dqn.DQNAgentClass import DQNAgent


class SkillChaining(object):
	def __init__(self, mdp, max_steps, lr_actor, lr_critic, ddpg_batch_size, device, max_num_options=5,
				 enable_option_timeout=True,
				 init_q=None, generate_plots=False, use_full_smdp_update=False,
				 log_dir="", seed=0, tensor_log=False,
				 ivf_c=0.5, ivf_c1=0.2, ivf_c2=0.2, uncertainty_method="competence",
				 ivf_step_penalty=0.0):
		"""                                                                       4
		Args:
			mdp (MDP): Underlying domain we have to solve
			max_steps (int): Number of time steps allowed per episode of the MDP
			lr_actor (float): Learning rate for DDPG Actor
			lr_critic (float): Learning rate for DDPG Critic
			ddpg_batch_size (int): Batch size for DDPG agents
			device (str): torch device {cpu/cuda:0/cuda:1}
			enable_option_timeout (bool): whether or not the option times out after some number of steps
			init_q (float): If not none, we use this value to initialize the value of a new option
			generate_plots (bool): whether or not to produce plots in this run
			use_full_smdp_update (bool): sparse 0/1 reward or discounted SMDP reward for training policy over options
			log_dir (os.path): directory to store all the scores for this run
			seed (int): We are going to use the same random seed for all the DQN solvers
			tensor_log (bool): Tensorboard logging enable
		"""
		self.mdp = mdp
		self.original_actions = deepcopy(mdp.actions)
		self.max_steps = max_steps
		self.enable_option_timeout = enable_option_timeout
		self.init_q = init_q
		self.use_full_smdp_update = use_full_smdp_update
		self.generate_plots = generate_plots
		self.log_dir = log_dir
		self.seed = seed
		self.device = torch.device(device)
		self.max_num_options = max_num_options
		# IVF config
		self.ivf_c = ivf_c
		self.ivf_c1 = ivf_c1
		self.ivf_c2 = ivf_c2
		self.uncertainty_method = uncertainty_method
		self.ivf_step_penalty = ivf_step_penalty

		tensor_name = "runs/{}_{}".format(args.experiment_name, seed)
		self.writer = SummaryWriter(tensor_name) if tensor_log else None

		print("Initializing skill chaining with option_timeout={}, seed={}".format(self.enable_option_timeout, seed))

		random.seed(seed)
		np.random.seed(seed)

		self.validation_scores = []

		# This option has an initiation set that is true everywhere and is allowed to operate on atomic timescale only
		self.global_option = Option(overall_mdp=self.mdp, name="global_option", global_solver=None,
									lr_actor=lr_actor, lr_critic=lr_critic,
									ddpg_batch_size=ddpg_batch_size,
									seed=self.seed, max_steps=self.max_steps,
									enable_timeout=self.enable_option_timeout,
									generate_plots=self.generate_plots, writer=self.writer, device=self.device,
									c_threshold=self.ivf_c, uncertainty_method=self.uncertainty_method,
									ivf_step_penalty=self.ivf_step_penalty)
		self.trained_options = [self.global_option]

		# This is our first untrained option - one that gets us to the goal state from nearby the goal
		# We pick this option when self.agent_over_options thinks that we should
		# Once we pick this option, we will use its internal DDPG solver to take primitive actions until termination
		# Once we hit its termination condition N times, we will start learning its initiation set
		# Once we have learned its initiation set, we will create its child option
		goal_option = Option(overall_mdp=self.mdp, name='overall_goal_policy', global_solver=self.global_option.solver,
							 lr_actor=lr_actor, lr_critic=lr_critic,
							 ddpg_batch_size=ddpg_batch_size,
							 seed=self.seed, max_steps=self.max_steps,
							 enable_timeout=self.enable_option_timeout,
							 generate_plots=self.generate_plots, writer=self.writer, device=self.device,
							 c_threshold=self.ivf_c, uncertainty_method=self.uncertainty_method,
							 ivf_step_penalty=self.ivf_step_penalty)

		# This is our policy over options
		# We use (double-deep) (intra-option) Q-learning to learn the Q-values of *options* at any queried state Q(s, o)
		# We start with this DQN Agent only predicting Q-values for taking the global_option, but as we learn new
		# options, this agent will predict Q-values for them as well
		self.agent_over_options = DQNAgent(self.mdp.state_space_size(), 1, trained_options=self.trained_options,
										   seed=seed, lr=1e-4, name="GlobalDQN", eps_start=1.0, tensor_log=tensor_log,
										   use_double_dqn=True, writer=self.writer, device=self.device)

		# Pointer to the current option:
		# 1. This option has the termination set which defines our current goal trigger
		# 2. This option has an untrained initialization set and policy, which we need to train from experience
		self.untrained_option = goal_option

		# List of init states seen while running this algorithm
		self.init_states = []

		# Debug variables
		self.global_execution_states = []
		self.num_option_executions = defaultdict(lambda : [])
		self.option_rewards = defaultdict(lambda : [])
		self.option_qvalues = defaultdict(lambda : [])
		self.num_options_history = []

	def create_child_option(self, parent_option):
		# Create new option whose termination is the initiation of the option we just trained
		name = "option_{}".format(str(len(self.trained_options) - 1))
		print("Creating {}".format(name))

		old_untrained_option_id = id(parent_option)
		new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_option.solver,
									  lr_actor=parent_option.solver.actor_learning_rate,
									  lr_critic=parent_option.solver.critic_learning_rate,
									  ddpg_batch_size=parent_option.solver.batch_size,
									  seed=self.seed, parent=parent_option, max_steps=self.max_steps,
									  enable_timeout=self.enable_option_timeout,
									  writer=self.writer, device=self.device,
									  c_threshold=self.ivf_c, uncertainty_method=self.uncertainty_method,
									  ivf_step_penalty=self.ivf_step_penalty)
# Bring in assertions here
		assert new_untrained_option is not None, "Failed to create new untrained option"
		assert old_untrained_option_id != id(new_untrained_option), "Option creation failed: object of same parent option returned"
		assert new_untrained_option.parent is parent_option, "New option doesn't link to parent option"
		return new_untrained_option

	def make_off_policy_updates_for_options(self, state, action, reward, next_state):
		for option in self.trained_options: # type: Option
			option.off_policy_update(state, action, reward, next_state)\


	def make_smdp_update(self, state, action, total_discounted_reward, next_state, option_transitions):
		"""
		Use Intra-Option Learning for sample efficient learning of the option-value function Q(s, o)
		Args:
			state (State): state from which we started option execution
			action (int): option taken by the global solver
			total_discounted_reward (float): cumulative reward from the overall SMDP update
			next_state (State): state we landed in after executing the option
			option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
		"""
		def get_reward(transitions):
			gamma = self.global_option.solver.gamma
			raw_rewards = [tt[2] for tt in transitions]
			return sum([(gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])

		# TODO: Should we do intra-option learning only when the option was successful in reaching its subgoal?
		selected_option = self.trained_options[action]  # type: Option
		for i, transition in enumerate(option_transitions):
			start_state = transition[0]
			if selected_option.is_init_true(start_state):
				if self.use_full_smdp_update:
					sub_transitions = option_transitions[i:]
					option_reward = get_reward(sub_transitions)
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=len(sub_transitions))
				else:
					option_reward = 1.0 if selected_option.is_term_true(next_state) else -1. #maybe change scale for +1 / 0: pure success indicator or a full discounted option return:
					self.agent_over_options.step(start_state.features(), action, option_reward, next_state.features(),
												 next_state.is_terminal(), num_steps=1)

	def get_init_q_value_for_new_option(self, newly_trained_option):
		global_solver = self.agent_over_options  # type: DQNAgent
		state_option_pairs = newly_trained_option.final_transitions
		q_values = []
		for state, option_idx in state_option_pairs:
			q_value = global_solver.get_qvalue(state.features(), option_idx)
			q_values.append(q_value)
		return np.max(q_values)

	def _augment_agent_with_new_option(self, newly_trained_option, init_q_value):
		"""
		Train the current untrained option and initialize a new one to target.
		Add the newly_trained_option as a new node to the Q-function over options
		Args:
			newly_trained_option (Option)
			init_q_value (float): if given use this, else compute init_q optimistically
		"""
		# Add the trained option to the action set of the global solver
		if newly_trained_option not in self.trained_options:
			self.trained_options.append(newly_trained_option)

		# Augment the global DQN with the newly trained option
		num_actions = len(self.trained_options)
		new_global_agent = DQNAgent(self.agent_over_options.state_size, num_actions, self.trained_options,
									seed=self.seed, name=self.agent_over_options.name,
									eps_start=self.agent_over_options.epsilon,
									tensor_log=self.agent_over_options.tensor_log,
									use_double_dqn=self.agent_over_options.use_ddqn,
									lr=self.agent_over_options.learning_rate,
									writer=self.writer, device=self.device)
		new_global_agent.replay_buffer = self.agent_over_options.replay_buffer

		init_q = self.get_init_q_value_for_new_option(newly_trained_option) if init_q_value is None else init_q_value
		print("Initializing new option node with q value {}".format(init_q))
		new_global_agent.policy_network.initialize_with_smaller_network(self.agent_over_options.policy_network, init_q)
		new_global_agent.target_network.initialize_with_smaller_network(self.agent_over_options.target_network, init_q)

		self.agent_over_options = new_global_agent

	def act(self, state):
		"""
		Sample option o from:
			p(o|s) = [mu(o|s) * J_plus(s,o)] / sum_o' [mu(o'|s) * J_plus(s,o')]
		where mu(o|s) is derived from the policy-over-options logits (DQN q-values)
		via softmax over feasible options only.
		"""
		# Q-values from policy-over-options (shape: [1, num_options])
		q_values_t = self.agent_over_options.get_qvalues(state.features()).detach().cpu().numpy().reshape(-1)

		# Feasibility mask
		state_t = torch.from_numpy(state.features()).float().unsqueeze(0).to(self.device)
		impossible_idx = set(self.agent_over_options.get_impossible_option_idx(state_t))
		possible_idx = [i for i in range(len(self.trained_options)) if i not in impossible_idx]

		if len(possible_idx) == 0:
			# Defensive fallback (should be rare)
			option_idx = 0
			selected_option = self.trained_options[option_idx]
			return selected_option

		# mu(o|s): softmax over feasible q-values only
		feasible_q = q_values_t[possible_idx]
		feasible_q = feasible_q - np.max(feasible_q)  # numerical stability
		mu = np.exp(feasible_q)
		mu_sum = np.sum(mu)
		if mu_sum <= 0 or not np.isfinite(mu_sum):
			mu = np.ones_like(mu) / len(mu)
		else:
			mu = mu / mu_sum

		# J_plus(s,o) per feasible option
		j_plus = np.array([self.trained_options[i].J_plus(state) for i in possible_idx], dtype=np.float64)
		j_plus = np.clip(j_plus, 0.0, 1.0)

		# Final sampling weights: mu * J_plus
		weights = mu * j_plus
		weights_sum = np.sum(weights)
		if weights_sum <= 0 or not np.isfinite(weights_sum):
			probs = np.ones_like(weights) / len(weights)  # fallback uniform over feasible
		else:
			probs = weights / weights_sum

		# Categorical sample (Algorithm 1, line 8)
		local_choice = np.random.choice(len(possible_idx), p=probs)
		option_idx = possible_idx[local_choice]

		# Keep scheduler update for compatibility with existing DQN agent state
		self.agent_over_options.update_epsilon()

		selected_option = self.trained_options[option_idx]  # type: Option

		# Debug logging: If it was possible to take an option, did we take it?
		for option in self.trained_options:  # type: Option
			if option.is_init_true(state):
				option_taken = option.option_idx == selected_option.option_idx
				if option.writer is not None:
					option.writer.add_scalar("{}_taken".format(option.name), option_taken, option.n_taken_or_not)
					option.taken_or_not.append(option_taken)
					option.n_taken_or_not += 1

		return selected_option

	def take_action(self, state, step_number, episode_option_executions):
		"""
		Either take a primitive action from `state` or execute a closed-loop option policy.
		Args:
			state (State)
			step_number (int): which iteration of the control loop we are on
			episode_option_executions (defaultdict)

		Returns:
			experiences (list): list of (s, a, r, s') tuples
			reward (float): sum of all rewards accumulated while executing chosen action
			next_state (State): state we landed in after executing chosen action
		"""
		selected_option = self.act(state)

		option_transitions, discounted_reward = selected_option.execute_option_in_mdp(self.mdp, step_number)

		option_reward = self.get_reward_from_experiences(option_transitions)
		next_state = self.get_next_state_from_experiences(option_transitions)
		if next_state is None:
			# No transitions collected (e.g., immediate termination): keep current state
			next_state = state

		# If we triggered the untrained option's termination condition, add to its buffer of terminal transitions
		if self.untrained_option.is_term_true(next_state) and not self.untrained_option.is_term_true(state):
			self.untrained_option.final_transitions.append((state, selected_option.option_idx))

		# Add data to train Q(s, o)
		self.make_smdp_update(state, selected_option.option_idx, discounted_reward, next_state, option_transitions)

		# Debug logging
		episode_option_executions[selected_option.name] += 1
		self.option_rewards[selected_option.name].append(discounted_reward)

		sampled_q_value = self.sample_qvalue(selected_option)
		self.option_qvalues[selected_option.name].append(sampled_q_value)
		if self.writer is not None:
			self.writer.add_scalar("{}_q_value".format(selected_option.name),
								   sampled_q_value, selected_option.num_executions)

		return option_transitions, option_reward, next_state, len(option_transitions)

	def sample_qvalue(self, option):
		if len(option.solver.replay_buffer) > 500:
			sample_experiences = option.solver.replay_buffer.sample(batch_size=500)
			sample_states = torch.from_numpy(sample_experiences[0]).float().to(self.device)
			sample_actions = torch.from_numpy(sample_experiences[1]).float().to(self.device)
			sample_qvalues = option.solver.get_qvalues(sample_states, sample_actions)
			return sample_qvalues.mean().item()
		return 0.0

	@staticmethod
	def get_next_state_from_experiences(experiences):
		if len(experiences) == 0:
			return None
		return experiences[-1][-1]

	@staticmethod
	def get_reward_from_experiences(experiences):
		total_reward = 0.
		for experience in experiences:
			reward = experience[2]
			total_reward += reward
		return total_reward

	def should_create_more_options(self):
		# IVF creation rule: if for all options, E[J(s0,o)] < c1 and E[U(s0,o)] < c2 over s0~rho0
		local_options = self.trained_options[1:]
		if len(local_options) == 0:
			return True
		for option in local_options:  # type: Option
			j_vals = []
			u_vals = []
			for s0 in self.init_states:
				j_vals.append(option.J_value(s0))
				u_vals.append(option.U_value(s0))
			if (np.mean(j_vals) >= self.ivf_c1) or (np.mean(u_vals) >= self.ivf_c2):
				return False
		return True
		# return len(self.trained_options) < self.max_num_options

	def skill_chaining(self, num_episodes, num_steps):

		# For logging purposes
		per_episode_scores = []
		per_episode_durations = []
		last_10_scores = deque(maxlen=10)
		last_10_durations = deque(maxlen=10)

		for episode in range(num_episodes):

			self.mdp.reset()
			score = 0.
			step_number = 0
			uo_episode_terminated = False
			state = deepcopy(self.mdp.init_state)
			self.init_states.append(deepcopy(state))
			experience_buffer = []
			state_buffer = []
			episode_option_executions = defaultdict(lambda : 0)

			while step_number < num_steps:
				experiences, reward, state, steps = self.take_action(state, step_number, episode_option_executions)
				score += reward
				step_number += steps
				for experience in experiences:
					experience_buffer.append(experience)
					state_buffer.append(experience[0])

				# Don't forget to add the last s' to the buffer
				if state.is_terminal() or (step_number == num_steps - 1):
					state_buffer.append(state)

				if self.untrained_option.is_term_true(state) and (not uo_episode_terminated) and\
						self.max_num_options > 0 and self.untrained_option not in self.trained_options:
					uo_episode_terminated = True
					self._augment_agent_with_new_option(self.untrained_option, init_q_value=self.init_q)
					if self.should_create_more_options():
						new_option = self.create_child_option(self.untrained_option)
						self.untrained_option = new_option

				if state.is_terminal():
					break

			last_10_scores.append(score)
			last_10_durations.append(step_number)
			per_episode_scores.append(score)
			per_episode_durations.append(step_number)

			self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)

		return per_episode_scores, per_episode_durations

	def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):

		print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
			episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon), end="")

		self.num_options_history.append(len(self.trained_options))

		if self.writer is not None:
			self.writer.add_scalar("Episodic scores", last_10_scores[-1], episode)

		if episode % 10 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tGO Eps: {:.2f}'.format(
				episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_option.solver.epsilon))

		if episode > 0 and episode % 100 == 0:
			eval_score = self.trained_forward_pass(render=False)
			self.validation_scores.append(eval_score)
			print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

		if self.generate_plots and episode % 10 == 0:
			render_sampled_value_function(self.global_option.solver, episode, args.experiment_name)

		for trained_option in self.trained_options:  # type: Option
			self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
			if self.writer is not None:
				self.writer.add_scalar("{}_executions".format(trained_option.name),
									   episode_option_executions[trained_option.name], episode)

	def save_all_models(self):
		for option in self.trained_options: # type: Option
			save_model(option.solver, args.episodes, best=False)

	def save_all_scores(self, pretrained, scores, durations):
		print("\rSaving training and validation scores..")
		training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
		training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
		validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)
		num_option_history_file_name = "sc_pretrained_{}_num_options_per_epsiode_{}.pkl".format(pretrained, self.seed)

		if self.log_dir:
			training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
			training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
			validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)
			num_option_history_file_name = os.path.join(self.log_dir, num_option_history_file_name)

		with open(training_scores_file_name, "wb+") as _f:
			pickle.dump(scores, _f)
		with open(training_durations_file_name, "wb+") as _f:
			pickle.dump(durations, _f)
		with open(validation_scores_file_name, "wb+") as _f:
			pickle.dump(self.validation_scores, _f)
		with open(num_option_history_file_name, "wb+") as _f:
			pickle.dump(self.num_options_history, _f)

	def perform_experiments(self):
		for option in self.trained_options:
			visualize_dqn_replay_buffer(option.solver, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(self.option_qvalues[o.name])
			plt.title(o.name)
		plt.savefig("value_function_plots/{}/sampled_q_so_{}.png".format(args.experiment_name, self.seed))
		plt.close()

		for option in self.trained_options:
			visualize_next_state_reward_heat_map(option.solver, args.episodes, args.experiment_name)

		for i, o in enumerate(self.trained_options):
			plt.subplot(1, len(self.trained_options), i + 1)
			plt.plot(o.taken_or_not)
			plt.title(o.name)
		plt.savefig("value_function_plots/{}_taken_or_not_{}.png".format(args.experiment_name, self.seed))
		plt.close()

	def trained_forward_pass(self, render=True):
		"""
		Called when skill chaining has finished training: execute options when possible and then atomic actions
		Returns:
			overall_reward (float): score accumulated over the course of the episode.
		"""
		self.mdp.reset()
		state = deepcopy(self.mdp.init_state)
		overall_reward = 0.
		self.mdp.render = render
		num_steps = 0
		option_trajectories = []

		while not state.is_terminal() and num_steps < self.max_steps:
			selected_option = self.act(state)

			option_reward, next_state, num_steps, option_state_trajectory = selected_option.trained_option_execution(self.mdp, num_steps)
			overall_reward += option_reward

			# option_state_trajectory is a list of (o, s) tuples
			option_trajectories.append(option_state_trajectory)

			state = next_state

		return overall_reward, option_trajectories

def create_log_dir(experiment_name):
	path = os.path.join(os.getcwd(), experiment_name)
	try:
		os.mkdir(path)
	except OSError:
		print("Creation of the directory %s failed" % path)
	else:
		print("Successfully created the directory %s " % path)
	return path


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment_name", type=str, help="Experiment Name")
	parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
	parser.add_argument("--env", type=str, help="name of gym environment", default="Pendulum-v0")
	parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
	parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
	parser.add_argument("--episodes", type=int, help="# episodes", default=200)
	parser.add_argument("--steps", type=int, help="# steps", default=1000)
	parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
	parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
	parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
	parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
	parser.add_argument("--option_timeout", type=bool, help="Whether option times out at 200 steps", default=False)
	parser.add_argument("--generate_plots", type=bool, help="Whether or not to generate plots", default=False)
	parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
	parser.add_argument("--control_cost", type=bool, help="Penalize high actuation solutions", default=False)
	parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
	parser.add_argument("--init_q", type=str, help="compute/zero", default="zero")
	parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
	# IVF thresholds
	parser.add_argument("--ivf_c", type=float, help="IVF initiation threshold c", default=0.5)
	parser.add_argument("--ivf_c1", type=float, help="Creation rule threshold E[J] < c1", default=0.2)
	parser.add_argument("--ivf_c2", type=float, help="Creation rule threshold E[U] < c2", default=0.2)
	parser.add_argument("--uncertainty_method", type=str, help="competence/count uncertainty method", default="competence")
	parser.add_argument("--ivf_step_penalty", type=float, help="Per-step penalty for non-terminal steps", default=0.0)
	args = parser.parse_args()

	if "reacher" in args.env.lower():
		from simple_rl.tasks.dm_fixed_reacher.FixedReacherMDPClass import FixedReacherMDP
		overall_mdp = FixedReacherMDP(seed=args.seed, difficulty=args.difficulty, render=args.render)
		state_dim = overall_mdp.init_state.features().shape[0]
		action_dim = overall_mdp.env.action_spec().minimum.shape[0]
	elif "maze" in args.env.lower():
		from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
		overall_mdp = PointMazeMDP(dense_reward=False, seed=args.seed, render=args.render)
		state_dim = 6
		action_dim = 2
	elif "point" in args.env.lower():
		from simple_rl.tasks.point_env.PointEnvMDPClass import PointEnvMDP
		overall_mdp = PointEnvMDP(control_cost=args.control_cost, render=args.render)
		state_dim = 4
		action_dim = 2
	else:
		from simple_rl.tasks.gym.GymMDPClass import GymMDP
		overall_mdp = GymMDP(args.env, render=args.render)
		state_dim = overall_mdp.env.observation_space.shape[0]
		action_dim = overall_mdp.env.action_space.shape[0]
		# Gymnasium-compatible seeding: use the MDP wrapper's seed method
		overall_mdp.seed(args.seed)

	# Create folders for saving various things
	logdir = create_log_dir(args.experiment_name)
	create_log_dir("saved_runs")
	create_log_dir("value_function_plots")
	create_log_dir("initiation_set_plots")
	create_log_dir("value_function_plots/{}".format(args.experiment_name))
	create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

	print("Training skill chaining agent from scratch")
	print("MDP InitState = ", overall_mdp.init_state)

	q0 = 0. if args.init_q == "zero" else None

	chainer = SkillChaining(overall_mdp, args.steps, args.lr_a, args.lr_c, args.ddpg_batch_size,
							device=args.device, seed=args.seed,
							log_dir=logdir,
							enable_option_timeout=args.option_timeout, init_q=q0, use_full_smdp_update=args.use_smdp_update,
							generate_plots=args.generate_plots, tensor_log=args.tensor_log,
							ivf_c=args.ivf_c, ivf_c1=args.ivf_c1, ivf_c2=args.ivf_c2,
							uncertainty_method=args.uncertainty_method,
							ivf_step_penalty=args.ivf_step_penalty)
	episodic_scores, episodic_durations = chainer.skill_chaining(args.episodes, args.steps)

	# Log performance metrics
	chainer.save_all_models()
	chainer.perform_experiments()
	chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
