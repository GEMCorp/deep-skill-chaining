# Python imports.
from __future__ import print_function
import random
import numpy as np
import pdb
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent

class Option(object):
	'''
	Represents a single DSC skill (Option) with its own low-level policy (DDPGAgent from func_approx.ddpg.DDPGAgentClass),
	  initiation classifier, timeout, and bookkeeping for training and execution.

	Attributes:
		Args in __init__
	'''

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size,
				 max_steps=20000, seed=0, parent=None,
				 enable_timeout=True, timeout=100,
				 generate_plots=False, device=torch.device("cpu"), writer=None,
				 c_threshold=0.5, uncertainty_method="competence", c_u=0.1,
				 ivf_step_penalty=0.0):
		"""
		Args:
			overall_mdp (MDP): The environment where this option will be used.
			name (str): Identifier for the option; special names are global_option and overall_goal_policy.
			global_solver (DDPGAgent): Global DDPG agent whose weights seed new options and share experience.
			lr_actor (float): Learning rate for the option's DDPG actor.
			lr_critic (float): Learning rate for the option's DDPG critic.
			ddpg_batch_size (int): Batch size for the option's DDPG policy.
			max_steps (int): Episode budget upper bound for option execution.
			seed (int): Random seed for reproducibility.
			parent (Option): Parent option whose initiation set defines this option's termination set.
			enable_timeout (bool): Whether to timeout option execution.
			timeout (int): Max steps per option execution when enable_timeout is True.
			generate_plots (bool): Whether to generate debug plots.
			device (torch.device): Device for all tensor operations.
			writer (SummaryWriter): Optional TensorBoard writer.
			c_threshold (float): IVF initiation threshold — is_init_true iff J(s,o) > c_threshold.
			uncertainty_method (str): "competence" or "count" uncertainty estimation.
			c_u (float): Coefficient for count-based uncertainty bonus.
			ivf_step_penalty (float): Per-step penalty for non-terminal option transitions.
		"""
		self.name = name
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.enable_timeout = enable_timeout
		self.generate_plots = generate_plots
		self.writer = writer
		self.c_threshold = c_threshold
		self.uncertainty_method = uncertainty_method
		self.c_u = c_u
		self.ivf_step_penalty = ivf_step_penalty

		self.timeout = np.inf
		# Global option operates on atomic timescale (per time step); child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		if self.name == "global_option":
			self.option_idx = 0
		elif self.name == "overall_goal_policy": # Goal option i.e. option that leads to the overall goal
			self.option_idx = 1
		else:
			self.option_idx = self.parent.option_idx + 1

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		self.device = device
		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		solver_name = "{}_ddpg_agent".format(self.name)
		self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, name=solver_name) if name == "global_option" else global_solver
		self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, tensor_log=(writer is not None), writer=writer, name=solver_name)

		# IVF networks (small MLP, sigmoid output) — always initialised
		self.ivf_net = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		).to(self.device)
		self.ivf_target_net = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		).to(self.device)
		for tparam,  param in zip(self.ivf_target_net.parameters(), self.ivf_net.parameters()):
			tparam.data.copy_(param.data)
		self.ivf_optimizer = optim.Adam(self.ivf_net.parameters(), lr=1e-3)
		self.ivf_target_tau = 1e-3
		self._counts = defaultdict(int)

		self.overall_mdp = overall_mdp
		self.final_transitions = []

		# Debug
		self.num_executions = 0
		self.taken_or_not = []
		self.n_taken_or_not = 0

		# IVF options are initialized immediately on creation.
		if self.name != "global_option":
			self.initialize_option_policy()

	# Methods for identity and comparisons by name:
	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other):
		if not isinstance(other, Option):
			return False
		return str(self) == str(other)

	def __ne__(self, other):
		return not self == other

	def initialize_with_global_ddpg(self):
		""" 
		Copy weights from global DDPG solver to local option DDPG solver for initializing the option's low-level policy.
		Copies actor/critic weights from global_solver and replays compatible experiences into local solver if inside initiation set.
		"""
		for my_param, global_param in zip(self.solver.actor.parameters(), self.global_solver.actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.critic.parameters(), self.global_solver.critic.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_actor.parameters(), self.global_solver.target_actor.parameters()):
			my_param.data.copy_(global_param.data)
		for my_param, global_param in zip(self.solver.target_critic.parameters(), self.global_solver.target_critic.parameters()):
			my_param.data.copy_(global_param.data)

		# Replay global buffer into local solver with IVF rewards
		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, 1.0, next_state, True)
				else:
					self.solver.step(state, action, -self.ivf_step_penalty, next_state, done)

	def batched_is_init_true(self, state_matrix):
		""" Check initiation set membership for a batch of states via IVF. """
		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))
		vals = [float(self._ivf_forward(self.ivf_net, state_matrix[i, :])) for i in range(state_matrix.shape[0])]
		return (np.array(vals) > self.c_threshold).astype(np.int32)

	def is_init_true(self, ground_state):
		"""
		Check if the given state is in this option's initiation set.
		Always True for global_option; otherwise J(s, o) > c_threshold via IVF.
		"""
		if self.name == "global_option":
			return True
		return float(self._ivf_forward(self.ivf_net, self._state_features(ground_state))) > self.c_threshold

	def is_term_true(self, ground_state):
		""" 
		Check if the given ground state is in the option's termination set. 
		For non-global options, termination set is defined by parent's initiation set (when parent.is_init_true).
		For global option and goal option, termination set is the goal states of the overall MDP.
		"""
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		# If option does not have a parent, it must be the goal option or the global option
		assert self.name == "overall_goal_policy" or self.name == "global_option", "{}".format(self.name)
		return self.overall_mdp.is_goal_state(ground_state)

	def initialize_option_policy(self):
		"""
		Initialize the local DDPG solver from the global option's weights and sync epsilon.
		"""
		self.initialize_with_global_ddpg()
		self.solver.epsilon = self.global_solver.epsilon

	def get_subgoal_reward(self, state):
		"""
		Return the per-step shaping reward for non-terminal option transitions.
		Success (termination) rewards are handled in update_option_solver.
		"""
		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return 0.0
		return -float(self.ivf_step_penalty)

	def off_policy_update(self, state, action, reward, next_state):
		"""
		Make off-policy updates to the option's DDPG solver using external transitions,
		only when inside the initiation set and not already at termination.
		"""
		assert self.overall_mdp.is_primitive_action(action), "option should be markov: {}".format(action)
		assert not state.is_terminal(), "Terminal state did not terminate at some point"
		if self.is_term_true(state):
			print("[off_policy_update] Warning: called updater on {} term states: {}".format(self.name, state))
			return
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, 1.0, next_state.features(), True)
		elif self.is_init_true(state):
			self.solver.step(state.features(), action, -self.ivf_step_penalty, next_state.features(), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		"""
		Make on-policy updates to the option's DDPG solver during execution.
		Applies success reward (1.0) at termination, step penalty otherwise, and runs IVF TD update.
		"""
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"
		if self.is_term_true(s):
			print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return
		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			self.solver.step(s.features(), a, 1.0, s_prime.features(), True)
		elif s_prime.is_terminal():
			print("[{}]: {} is_terminal() but not term_true()".format(self.name, s))
			self.solver.step(s.features(), a, 1.0, s_prime.features(), True)
		else:
			self.solver.step(s.features(), a, -self.ivf_step_penalty, s_prime.features(), False)
		self.update_ivf(s, s_prime)

	def execute_option_in_mdp(self, mdp, step_number):
		"""
		Option main control loop. Executes until termination, terminal state, max steps, or timeout.
		Returns:
			option_transitions (list): list of (s, a, r, s') tuples
			total_reward (float): cumulative reward obtained
		"""
		state = mdp.cur_state

		if self.is_init_true(state):
			option_transitions = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), evaluation_mode=False)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				self.update_option_solver(state, action, reward, next_state)

				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option":
					self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
					self.global_solver.update_epsilon()

				self.solver.update_epsilon()
				option_transitions.append((state, action, reward, next_state))

				total_reward += reward
				state = next_state
				step_number += 1
				num_steps += 1

			if self.writer is not None:
				self.writer.add_scalar("{}_ExecutionLength".format(self.name), len(option_transitions), self.num_executions)

			return option_transitions, total_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	# -------------------------
	# IVF helpers and updates
	# -------------------------
	def _state_features(self, ground_state):
		"""
		Return a plain numpy feature vector for IVF networks.
		Args:
			ground_state (State or np.array): simple_rl `State` or raw feature array.
		Returns:
			np.array: feature vector used as input to `ivf_net` / `ivf_target_net`.
		"""
		return ground_state.features() if isinstance(ground_state, State) else ground_state

	def _ivf_forward(self, net, features_np):
		"""
		Compute a single forward pass through the given IVF network.
		Args:
			net (nn.Module): IVF network (`ivf_net` or `ivf_target_net`).
			features_np (State or np.array): input features or `State` to evaluate.
		Returns:
			float: scalar IVF value in [0, 1].
		"""
		if isinstance(features_np, State):
			features_np = features_np.features()
		x = torch.from_numpy(np.array(features_np)).float().unsqueeze(0).to(self.device)
		net.eval()
		with torch.no_grad():
			y = net(x)
		net.train()
		return y[0][0].item()

	def J_value(self, ground_state):
		""" Return the current IVF estimate J(s, o) in [0, 1]. Always 1.0 for global_option. """
		if self.name == "global_option":
			return 1.0
		return float(self._ivf_forward(self.ivf_net, self._state_features(ground_state)))

	def J_target_value(self, ground_state):
		""" Return the target-network IVF estimate J_target(s, o). Always 1.0 for global_option. """
		if self.name == "global_option":
			return 1.0
		return float(self._ivf_forward(self.ivf_target_net, self._state_features(ground_state)))

	def U_value(self, ground_state):
		"""
		Estimate uncertainty U(s, o) about the IVF prediction.
		- competence: |J(s,o) - J_target(s,o)|
		- count: c_u / sqrt(N(s, o)) using coarse position bins.
		"""
		if self.uncertainty_method == "competence":
			return abs(self.J_value(ground_state) - self.J_target_value(ground_state))
		feats = self._state_features(ground_state)
		pos = feats[:2]
		key = (int(round(pos[0]*10)), int(round(pos[1]*10)), self.option_idx)
		n = max(1, self._counts[key])
		return float(self.c_u / np.sqrt(n))

	def J_plus(self, ground_state):
		""" Compute optimistic executability J⁺(s, o) = clip(J + U, 0, 1). """
		return float(np.clip(self.J_value(ground_state) + self.U_value(ground_state), 0.0, 1.0))

	def _soft_update_ivf(self):
		"""
		Soft-update the target IVF network parameters with coefficient `ivf_target_tau`.
		"""
		for tparam, param in zip(self.ivf_target_net.parameters(), self.ivf_net.parameters()):
			tparam.data.copy_(self.ivf_target_tau * param.data + (1.0 - self.ivf_target_tau) * tparam.data)

	def update_ivf(self, s, s_prime):
		"""
		Perform a TD(0) update for the IVF.
		Target: `beta(s') + J_target(s')`, where `beta(s') = 1` if `is_term_true(s')` else `0`.
		Updates online IVF (`ivf_net`), then softly updates the target network and count-based statistics.
		Args:
			s (State): current state.
			s_prime (State): next state.
		"""
		# TD(0): target = beta(s') + J_target(s')
		beta_sp = 1.0 if self.is_term_true(s_prime) else 0.0
		s_np = self._state_features(s)
		sp_np = self._state_features(s_prime)

		states = torch.from_numpy(np.array(s_np)).float().unsqueeze(0).to(self.device)
		next_states = torch.from_numpy(np.array(sp_np)).float().unsqueeze(0).to(self.device)

		j_s = self.ivf_net(states)
		with torch.no_grad():
			j_sp_target = self.ivf_target_net(next_states)
		target = torch.clamp(j_sp_target + beta_sp, 0.0, 1.0)

		loss = torch.mean((j_s - target)**2)
		self.ivf_optimizer.zero_grad()
		loss.backward()
		self.ivf_optimizer.step()

		# update counts for count-based U
		if self.uncertainty_method == "count":
			pos = s_np[:2]
			key = (int(round(pos[0]*10)), int(round(pos[1]*10)), self.option_idx)
			self._counts[key] += 1

		self._soft_update_ivf()

	def trained_option_execution(self, mdp, outer_step_counter):
		"""
		Execute the option in the MDP until termination, terminal state, max steps, or timeout.
		Args:
			mdp (MDP): environment where actions are being taken
			outer_step_counter (int): how many steps have already elapsed in the outer control loop.
		Returns:
			score (float): cumulative reward obtained by executing the option
			state (State): state where option execution ended
			step_number (int): updated step number after option execution
			state_option_trajectory (list): list of (option_idx, state) tuples visited during option execution
		"""
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0
		state_option_trajectory = []

		while not self.is_term_true(state) and not state.is_terminal()\
				and step_number < self.max_steps and num_steps < self.timeout:
			state_option_trajectory.append((self.option_idx, deepcopy(state)))
			action = self.solver.act(state.features(), evaluation_mode=True)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number, state_option_trajectory
