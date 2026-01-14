# Python imports.
from __future__ import print_function
import random
import numpy as np
import pdb
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import itertools
from scipy.spatial import distance

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ddpg.DDPGAgentClass import DDPGAgent
from simple_rl.agents.func_approx.dsc.utils import Experience

class Option(object):
	'''
	Represents a single DSC skill (Option) with its own low-level policy (DDPGAgent from func_approx.ddpg.DDPGAgentClass),
	  initiation classifier, timeout, and bookkeeping for training and execution.

	Attributes:
		Args in __init__
	'''

	def __init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size, classifier_type="ocsvm",
				 subgoal_reward=0., max_steps=20000, seed=0, parent=None, num_subgoal_hits_required=3, buffer_length=20,
				 dense_reward=False, enable_timeout=True, timeout=100, initiation_period=2,
				 generate_plots=False, device=torch.device("cpu"), writer=None,
				 use_ivf=False, c_threshold=0.5, uncertainty_method="competence", c_u=0.1,
				 ivf_step_penalty=0.0):
		'''
		Args:
			overall_mdp (MDP) : The environment where this option will be used.
			name (str) : Identifier for the option; special names are global_option and overall_goal_policy.
			global_solver (DDPGAgent) : The global DDPG agent used to initialize the global option and share experience with other options. For non-global options, reference to the global DDPGAgent whose weights seed the new option; global owns its own solver.
			lr_actor (float) : Learning rate for the option's low-level DDPG policy.
			lr_critic (float) : Learning rate for the option's low-level DDPG critic.
			ddpg_batch_size (int) : Batch size for the option's low-level DDPG policy.
			num_subgoal_hits_required (int) : Number of successful terminations required before training the initiation set classifier and initializing the option's low-level policy.
			buffer_length (int) : Number of states to keep per initiation experience or experience buffer added to the option.
			classifier_type (str) : Type of classifier used for the initiation set. Either "ocsvm" (one-class SVM), "elliptic" (Elliptic Envelope), or "tcsvm" (two-class SVM).
			subgoal_reward (float) : Reward when the option hits its termination set; affects learning targets. Used to train the option's low-level policy (DDPGAgent).
			max_steps (int) : Maximum number of steps allowed in the overall MDP during option execution. Episode budget upper bound considered during option execution.
			seed (int) : Random seed for reproducibility.
			parent (Option) : Parent option whose initiation set defines this option's termination set (for chained options). None for global option and goal option.
			dense_reward (bool) : Whether to use dense subgoal rewards based on distance to goal/initiation set, or sparse (-1 per step) rewards.
			enable_timeout (bool) : Whether to enable timeout for option execution.
			timeout (int) : Timeout duration for option execution if enabled.
			initiation_period (int) : Period for initiation set evaluation.
			generate_plots (bool) : Whether to generate plots for debugging and analysis.
			device (torch.device) : Device on which to run the option's computations.
			writer (SummaryWriter) : Optional TensorBoard writer for logging.
		'''
		self.name = name
		self.subgoal_reward = subgoal_reward
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.dense_reward = dense_reward
		self.initiation_period = initiation_period
		self.enable_timeout = enable_timeout
		self.classifier_type = classifier_type
		self.generate_plots = generate_plots
		self.writer = writer
		self.use_ivf = use_ivf
		self.c_threshold = c_threshold
		self.uncertainty_method = uncertainty_method
		self.c_u = c_u
		self.ivf_step_penalty = ivf_step_penalty

		self.timeout = np.inf

		# Global option operates on a time-scale of 1 while child (learned) options are temporally extended
		if enable_timeout:
			self.timeout = 1 if name == "global_option" else timeout

		if self.name == "global_option":
			self.option_idx = 0
		elif self.name == "overall_goal_policy": # Goal option
			self.option_idx = 1
		else:
			self.option_idx = self.parent.option_idx + 1

		print("Creating {} with enable_timeout={}".format(name, enable_timeout))

		random.seed(seed)
		np.random.seed(seed)

		state_size = overall_mdp.state_space_size()
		action_size = overall_mdp.action_space_size()

		solver_name = "{}_ddpg_agent".format(self.name)
		self.global_solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, name=solver_name) if name == "global_option" else global_solver
		self.solver = DDPGAgent(state_size, action_size, seed, device, lr_actor, lr_critic, ddpg_batch_size, tensor_log=(writer is not None), writer=writer, name=solver_name)

		# IVF networks (small MLP, sigmoid output)
		self.device = device
		if self.use_ivf:
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

			# initialize target params
			for tparam, param in zip(self.ivf_target_net.parameters(), self.ivf_net.parameters()):
				tparam.data.copy_(param.data)

			self.ivf_optimizer = optim.Adam(self.ivf_net.parameters(), lr=1e-3)
			self.ivf_target_tau = 1e-3
			# buffers for uncertainty
			from collections import defaultdict
			self._counts = defaultdict(int)

		# Attributes related to initiation set classifiers
		self.num_goal_hits = 0
		self.positive_examples = []
		self.negative_examples = []
		self.experience_buffer = []
		self.initiation_classifier = None
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.buffer_length = buffer_length

		self.overall_mdp = overall_mdp
		self.final_transitions = []

		# Debug member variables
		self.num_executions = 0
		self.taken_or_not = []
		self.n_taken_or_not = 0

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

	def get_training_phase(self):
		""" Returns the current training phase of the option. """
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

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

		# Not using off_policy_update() because we have numpy arrays not state objects here
		for state, action, reward, next_state, done in self.global_solver.replay_buffer.memory:
			if self.is_init_true(state):
				if self.is_term_true(next_state):
					self.solver.step(state, action, self.subgoal_reward, next_state, True)
				else:
					subgoal_reward = self.get_subgoal_reward(next_state)
					self.solver.step(state, action, subgoal_reward, next_state, done)

	def batched_is_init_true(self, state_matrix):
		""" Check initiation set membership for a batch of states. """
		if self.name == "global_option":
			return np.ones((state_matrix.shape[0]))
		if self.use_ivf:
			vals = []
			for i in range(state_matrix.shape[0]):
				s = state_matrix[i, :]
				vals.append(float(self._ivf_forward(self.ivf_net, s)))
			return (np.array(vals) > self.c_threshold).astype(np.int32)
		position_matrix = state_matrix[:, :2]
		return self.initiation_classifier.predict(position_matrix) == 1

	def is_init_true(self, ground_state):
		""" 
		Check if the given ground state is in the option's initiation set.
		 True everywhere for global_option; IVF mode uses J(s,o) > c; else classifier.
		"""
		if self.name == "global_option":
			return True
		if self.use_ivf:
			j = float(self._ivf_forward(self.ivf_net, self._state_features(ground_state)))
			return j > self.c_threshold
		features = ground_state.features()[:2] if isinstance(ground_state, State) else ground_state[:2]
		return self.initiation_classifier.predict([features])[0] == 1

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

	def add_initiation_experience(self, states):
		""" 
		Add a list of states to the initiation experience buffer. 
		Args:
			states (list): List of State objects representing the initiation experience.
		Truncates trajectory to buffer_length and records positions into positive_examples.
		"""
		# IVF mode learns initiation online; skip classifier data collection
		if self.use_ivf:
			return
		assert type(states) == list, "Expected initiation experience sample to be a queue"
		segmented_states = deepcopy(states)
		if len(states) >= self.buffer_length:
			segmented_states = segmented_states[-self.buffer_length:]
		segmented_positions = [segmented_state.position for segmented_state in segmented_states]
		self.positive_examples.append(segmented_positions)

	def add_experience_buffer(self, experience_queue):
		""" 
		Add a list of experiences to the option's experience buffer. 
		Args:
			experience_queue (list): List of experiences (tuples) representing the experience buffer.
		Wraps (s,a,r,s') tuples as Experience objects, truncates trajectory to buffer_length and records into experience_buffer.
		"""
		assert type(experience_queue) == list, "Expected initiation experience sample to be a list"
		segmented_experiences = deepcopy(experience_queue)
		if len(segmented_experiences) >= self.buffer_length:
			segmented_experiences = segmented_experiences[-self.buffer_length:]
		experiences = [Experience(*exp) for exp in segmented_experiences]
		self.experience_buffer.append(experiences)

	@staticmethod
	def construct_feature_matrix(examples):
		"""
		Construct a feature matrix from a list of examples. Flattens list-of-lists of positions into an np.array for classifier fitting.
		Args:
			examples (list): List of lists of states.
		Returns:
			np.array: Feature matrix where each row is a state.
		"""
		states = list(itertools.chain.from_iterable(examples))
		return np.array(states)

	def get_distances_to_goal(self, position_matrix):
		"""
		Compute distances from a batch of positions to the option's goal (either overall MDP goal or parent's initiation set depending on option type).
		Args:
			position_matrix (np.array): Matrix of positions where each row is a position.
		Returns:
			np.array: Distances from each position to the goal/initiation set.
		"""
		if self.parent is None:
			goal_position = self.overall_mdp.goal_position
			return distance.cdist(goal_position[None, ...], position_matrix, "euclidean")

		# else distance to parent’s initiation boundary (via SVM decision function).
		distances = -self.parent.initiation_classifier.decision_function(position_matrix)
		distances[distances <= 0.] = 0. # Clamp negative distances to zero
		return distances

	@staticmethod
	def distance_to_weights(distances):
		"""
		Convert distances to normalized weights ([0,1]-scaled) using an exponential decay function. Purpose is to weight closer examples higher. (used for weighted training if needed)
		Args:
			distances (np.array): Array of distances.
		Returns:
			np.array: Weights corresponding to distances.
		"""
		weights = np.copy(distances)
		for row in range(weights.shape[0]):
			if weights[row] > 0.:
				weights[row] = np.exp(-1. * weights[row])
			else:
				weights[row] = 1.
		return weights

	def train_one_class_svm(self):
		"""
		Fits OneClassSVM on positive position examples.
		"""
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		# Smaller gamma -> influence of example reaches farther. Using scale leads to smaller gamma than auto.
		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_elliptic_envelope_classifier(self):
		"""
		Fits EllipticEnvelope on positive position examples.
		"""
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = EllipticEnvelope(contamination=0.2)
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self):
		"""
		Fits a two-class SVM on positive and negative position examples.
		""" 
		positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		# if len(self.negative_examples) >= 10:
		kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
		# else:
		# 	kwargs = {"kernel": "linear", "gamma": "scale"}

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		initiation_classifier = svm.SVC(**kwargs)
		initiation_classifier.fit(X, Y)

		training_predictions = initiation_classifier.predict(X)
		positive_training_examples = X[training_predictions == 1]

		self.initiation_classifier = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
		self.initiation_classifier.fit(positive_training_examples)

		self.classifier_type = "tcsvm"

	def train_initiation_classifier(self):
		"""
		Train the initiation set classifier based on the specified classifier type.
		"""
		if self.use_ivf:
			# IVF is learned online; no classifier training needed
			return
		if self.classifier_type == "ocsvm":
			self.train_one_class_svm()
		elif self.classifier_type == "elliptic":
			self.train_elliptic_envelope_classifier()
		else:
			raise NotImplementedError("{} not supported".format(self.classifier_type))

	def initialize_option_policy(self):
		"""
		Initialize the local DDPG solver with the weights of the global option's DDPG solver,
			syncs epsilon, and performs fitted Q-iteration using accumulated experience_buffer
		"""
		self.initialize_with_global_ddpg()

		self.solver.epsilon = self.global_solver.epsilon

		# Fitted Q-iteration on the experiences that led to triggering the current option's termination condition
		experience_buffer = list(itertools.chain.from_iterable(self.experience_buffer))
		for experience in experience_buffer:
			state, action, reward, next_state = experience.serialize()
			self.update_option_solver(state, action, reward, next_state)

	def train(self, experience_buffer, state_buffer):
		"""
		Called every time the agent hits the current option's termination set.
		Records initiation experiences and experience buffer, increments goal hit count,
			trains initiation classifier and initializes option policy if enough goal hits have occurred.
		Args:
			experience_buffer (list)
			state_buffer (list)
		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			self.initialize_option_policy()
			return True
		return False

	def get_subgoal_reward(self, state):

		# If at termination, caller should handle success reward; avoid double-counting here.
		if self.is_term_true(state):
			print("~~~~~ Warning: subgoal query at goal ~~~~~")
			return 0.0

		# IVF mode: use pure indicator at success (handled elsewhere) and 0 per-step otherwise.
		# This aligns option-policy updates with Algorithm 1 while avoiding double-counting.
		if self.use_ivf:
			# Apply optional per-step penalty in IVF mode (positive hyperparam → negative reward)
			return -float(self.ivf_step_penalty)

		# Classifier mode below
		# Return step penalty in sparse reward domain
		if not self.dense_reward:
			return -1.0

		# Rewards based on position only (classifier mode)
		position_vector = state.features()[:2] if isinstance(state, State) else state[:2]

		# For global and parent option, we use the negative distance to the goal state
		if self.parent is None:
			return -0.1 * self.overall_mdp.distance_to_goal(position_vector)

		# For every other option, we use the negative distance to the parent's initiation set classifier
		dist = self.parent.initiation_classifier.decision_function(position_vector.reshape(1, -1))[0]

		# Decision_function returns a negative distance for points not inside the classifier
		subgoal_reward = 0.0 if dist >= 0 else dist
		return subgoal_reward

	def off_policy_update(self, state, action, reward, next_state):
		""" 
		Make off-policy updates to the current option's low level DDPG solver from external transitions 
			only when inside initiation and not at termination.
		Args:
			state (State): current state
			action (int): action taken
			reward (float): reward received
			next_state (State): next state reached
		"""
		assert self.overall_mdp.is_primitive_action(action), "option should be markov: {}".format(action)
		assert not state.is_terminal(), "Terminal state did not terminate at some point"

		# Don't make updates while walking around the termination set of an option
		if self.is_term_true(state):
			print("[off_policy_update] Warning: called updater on {} term states: {}".format(self.name, state))
			return

		# Off-policy updates for states outside tne initiation set were discarded
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, self.subgoal_reward, next_state.features(), True)
		elif self.is_init_true(state):
			subgoal_reward = self.get_subgoal_reward(next_state)
			self.solver.step(state.features(), action, subgoal_reward, next_state.features(), next_state.is_terminal())

	def update_option_solver(self, s, a, r, s_prime):
		""" 
		Make on-policy updates to the current option's low-level DDPG solver during option execution;
			handles success (subgoal_reward), terminal, and intermediate shaped reward.
		Args:
			s (State): current state
			a (int): action taken
			r (float): reward received
			s_prime (State): next state reached
		"""
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s):
			print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
			return

		if self.is_term_true(s_prime):
			print("{} execution successful".format(self.name))
			reward_to_use = 1.0 if self.use_ivf else self.subgoal_reward
			self.solver.step(s.features(), a, reward_to_use, s_prime.features(), True)
		elif s_prime.is_terminal():
			print("[{}]: {} is_terminal() but not term_true()".format(self.name, s))
			reward_to_use = 1.0 if self.use_ivf else self.subgoal_reward
			self.solver.step(s.features(), a, reward_to_use, s_prime.features(), True)
		else:
			subgoal_reward = self.get_subgoal_reward(s_prime)
			self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), False)

		# IVF TD update
		if self.use_ivf:
			self.update_ivf(s, s_prime)

	def execute_option_in_mdp(self, mdp, step_number):
		"""
		Option main control loop.
		Checks if current state is in initiation set of option, then loops until termination, terminal state, max steps, or timeout

		Args:
			mdp (MDP): environment where actions are being taken
			step_number (int): how many steps have already elapsed in the outer control loop.

		Returns:
			option_transitions (list): list of (s, a, r, s') tuples
			discounted_reward (float): cumulative discounted reward obtained by executing the option
		"""
		start_state = deepcopy(mdp.cur_state)
		state = mdp.cur_state

		if self.is_init_true(state):
			option_transitions = []
			total_reward = 0.
			self.num_executions += 1
			num_steps = 0
			visited_states = []

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), evaluation_mode=False)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)

				if self.name != "global_option":
					self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
					self.global_solver.update_epsilon()

				self.solver.update_epsilon()
				option_transitions.append((state, action, reward, next_state))

				total_reward += reward
				state = next_state

				# step_number is to check if we exhaust the episodic step budget
				# num_steps is to appropriately discount the rewards during option execution (and check for timeouts)
				step_number += 1
				num_steps += 1

			# Don't forget to add the final state to the followed trajectory
			visited_states.append(state)

			if self.is_term_true(state):
				self.num_goal_hits += 1

			if self.get_training_phase() == "initiation" and self.name != "global_option":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps, step_number)

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
		"""
		Return the current IVF estimate `J(s, o)` for this option.
		For the `global_option`, returns 1.0. Uses `ivf_net` in IVF mode.
		"""
		if self.name == "global_option":
			return 1.0
		return float(self._ivf_forward(self.ivf_net, self._state_features(ground_state))) if self.use_ivf else 1.0

	def J_target_value(self, ground_state):
		"""
		Return the target-network IVF estimate `J_target(s, o)` for stability.
		For the `global_option`, returns 1.0.
		"""
		if self.name == "global_option":
			return 1.0
		return float(self._ivf_forward(self.ivf_target_net, self._state_features(ground_state))) if self.use_ivf else 1.0

	def U_value(self, ground_state):
		"""
		Estimate uncertainty `U(s, o)` about the IVF prediction.
		Methods:
		- competence: absolute difference between online and target IVF (`|J - J_target|`).
		- count: count-based bonus `c_u / sqrt(N(s, o))` using coarse position bins.
		Returns 0.0 when IVF mode is disabled.
		"""
		if not self.use_ivf:
			return 0.0
		if self.uncertainty_method == "competence":
			return abs(self.J_value(ground_state) - self.J_target_value(ground_state))
		# count-based, bin by position (first two dims)
		feats = self._state_features(ground_state)
		pos = feats[:2]
		key = (int(round(pos[0]*10)), int(round(pos[1]*10)), self.option_idx)
		n = max(1, self._counts[key])
		return float(self.c_u / np.sqrt(n))

	def J_plus(self, ground_state):
		"""
		Compute optimistic executability `J⁺(s, o) = clip(J + U, 0, 1)`.
		Returns 1.0 when IVF mode is disabled.
		"""
		if not self.use_ivf:
			return 1.0
		val = self.J_value(ground_state) + self.U_value(ground_state)
		return float(np.clip(val, 0.0, 1.0))

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

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps,
										 outer_step_number):
		"""
		Refine the initiation set classifier after each successful or timed-out execution during initiation phase.
		Args:
			visited_states (list): List of State objects visited during option execution.
			start_state (State): State where option execution started.
			final_state (State): State where option execution ended.
			num_steps (int): Number of steps taken during option execution.
			outer_step_number (int): Total number of steps taken in the overall MDP.
		"""
		if self.is_term_true(final_state):  # success
			positive_states = [start_state] + visited_states[-self.buffer_length:]
			positive_examples = [state.position for state in positive_states]
			self.positive_examples.append(positive_examples)

		elif num_steps == self.timeout:
			negative_examples = [start_state.position]
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal() or outer_step_number == self.max_steps, \
				"Hit else case, but {} was not terminal".format(final_state)

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			self.train_two_class_classifier()

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
