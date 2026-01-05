
::: mermaid
flowchart TD
    subgraph Entry Point
    A["SkillChainingAgentClass.py <br/> (main block)"] -->|1. Init MDP & Agent| B(SkillChaining.__init__)
    A -->|2. Start Training| C(SkillChaining.skill_chaining)
    end

    subgraph SkillChaining Agent
    B -->|Init| D[Global Option <br/> Atomic Actions]
    B -->|Init| E[Goal Option <br/> First Skill]
    B -->|Init| F[Global DQN <br/> Policy over Options]
    
    C -->|Loop Episodes| G{take_action}
    G -->|1. Select Option| F
    G -->|2. Execute| H[Option.execute_option_in_mdp]
    G -->|3. Update Global| I[make_smdp_update]
    
    C -->|Check Discovery| J{"untrained_option <br/> reached goal?"}
    J -- Yes --> K[untrained_option.train]
    K -->|Train Classifier| L[SVM / EllipticEnvelope]
    K -->|Train Policy| M[DDPG Solver]
    
    J -- Yes --> N[create_child_option]
    N -->|Chain Backwards| O[New Option]
    O -->|Target| L
    
    J -- Yes --> P[_augment_agent_with_new_option]
    P -->|Expand Output| F
    end

    subgraph Option Execution
    H -->|Loop Steps| Q["solver.act <br/> (DDPG)"]
    Q -->|Primitive Action| R[MDP.execute_agent_action]
    R -->|Next State| S{is_term_true?}
    S -- No --> Q
    S -- Yes --> T[Return Reward]
    end
::: 