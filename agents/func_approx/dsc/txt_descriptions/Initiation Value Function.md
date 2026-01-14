### **3.1 Initiation Value Function (IVF)**

An option $o$’s initiation set $\mathcal{I}_{o}$ is defined as the set of states from which the option policy $\pi_{o}$ can succeed with high probability. By treating the success condition of the option as a cumulant $c_{o}:s\rightarrow\{0,1\}$ and setting the timescale $\gamma_{c_{o}}:=1$, the corresponding Generalized Value Function (GVF), $\mathcal{V}^{\pi_{o}}(s_{t})=\mathbb{E}_{\pi_{o}}[\sum_{t=0}^{t=H_{o}}c_{o}(s_{t+1})|s_{t+1}\sim\mathcal{T}(s_{t},\pi_{o}(s_{t}))]=\mathbb{P}(c_{o}=1|S=s_{t},O=o)$, represents the initiation probability at state $s_{t}$, where $H_{o}$ is the option horizon. Notably, the IVF ($\mathcal{V}^{\pi_{o}}$) is distinct from the value function used by the option policy for control ($V_{o}$); while $\pi_{o}$ maximises an arbitrary reward function $\mathcal{R}_{o}$, the resulting value function $V_{o}$ approximates the value of the optimal option policy and cannot be interpreted as an initiation probability.

**Using the IVF as an initiation set.** To use the IVF directly as the initiation set, a threshold must be selected above which a state is deemed to be in the option’s initiation set. Previous work attempted to reuse the option value function $V_{o}$ as the IVF but required heuristic schemes to pick thresholds; however, because the proposed $\mathcal{V}_{\phi}^{\pi_{o}}$ outputs an **interpretable number between 0 and 1** (enforced by a sigmoid layer), it is easy to threshold regardless of the reward function $\mathcal{R}_{o}$.

**Learning the IVF.** Computing the success probability of a policy $\pi_{o}^{t}$ is equivalent to **off-policy policy evaluation**. This approach must be sample-efficient to ensure the policy is evaluated sufficiently before it changes. Because the initiation cumulant $c_{o}$ is a sparse binary function, Monte Carlo estimation results in high variance; thus, **TD-learning (specifically TD(0))** is used to estimate $\mathcal{V}_{\phi}^{\pi_{o}}$. This allows the agent to propagate value from partial trajectories and bootstrap, making it more sample-efficient than a standard classifier.

***

### **3.3 Overcoming Pessimistic Bias**

Unsuccessful option trajectories cause the initiation set to shrink, meaning that once a state is outside the initiation set, the option is no longer executed from there. Consequently, even if the policy could eventually succeed from that state, it remains excluded, and the option ends up with a **smaller region of competence than required**. This **pessimistic bias** can prevent the learning of useful options and lower the effectiveness of hierarchical RL agents. To mitigate this, the initiation set is expanded to include states from which policy improvement is most likely.

The sources propose two simple approaches for identifying these states using bonus-based exploration:

1.  **Competence progress** attempts to capture regions where a policy is either improving or regressing. This is computed as **changes in the IVF over time**: $\mathcal{B}_{1}(s)=|\mathcal{V}_{o}^{t}(s)-\mathcal{V}_{o}^{t-K}(s)|$, where $\mathcal{V}_{o}^{t}$ is the current IVF and $\mathcal{V}_{o}^{t-K}$ is the IVF estimate $K$ timesteps ago obtained using the target network.
2.  **Count-based bonus approach** tracks the number of times $N(s,o)$ an option $o$ has been executed from state $s$. This is converted into an uncertainty measure $\mathcal{B}_{2}(s)=c/\sqrt{N(s,o)}$, where $c$ is a scalar hyperparameter.

The agent uses count-based bonuses when tabular counts are readily available; otherwise, it relies on competence progress.

***

### **4 Experiments**

The experiments aim to evaluate if these methods result in better, more efficiently learned initiation sets, whether this improves option learning as a whole, and if these changes allow state-of-the-art skill discovery methods to outperform baselines in **sparse-reward continuous control problems**.

**Implementation Details.** Option policies are learned using **Rainbow** for discrete action spaces and **TD3** for continuous ones. All options share the same **Universal Value Function Approximator (UVFA)** but are conditioned on their specific subgoals. The IVF is learned using **Fitted Q-Evaluation**, prioritised experience replay, and target networks. The IVF Q-function is parameterized using neural networks that have the same architecture as the Rainbow/TD3. Each option has a **"gestation period" of 5**, meaning its initiation set is optimistically initialised to be true everywhere until the option has seen 5 successful trajectories.
