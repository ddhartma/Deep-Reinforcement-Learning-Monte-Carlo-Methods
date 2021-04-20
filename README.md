[image1]: assets/grid_world.png "image1"
[image2]: assets/mcm_equi.png "image2"
[image3]: assets/mc_policy_1.png "image3"
[image4]: assets/q_table.png "image4"
[image5]: assets/visits.png "image5"
[image6]: assets/algo9.png "image6"
[image7]: assets/first_visit_mc.png "image7"


# Deep Reinforcement Learning Theory - Monte Carlo Methods

## Content
- [Introduction](#intro)
- [Grid World Example](#grid_world)
- [Monte Carlo Methods](#mcm)
- [MC Prediction](#mcp)
- [OpenAI Gym: BlackJackEnv](#blackjackenv)
- [Greedy Policies](#greedy_policies)
- [Epsilon-Greedy Policies](#epsilon_greedy_policies)
- [MC Control](#mc_control)
- [Exploration vs. Exploitation](#explorat_vs_exploit)
- [Incremental Mean](#inc_mean)
- [Constant-alpha](#const_alpha)   
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## Grid World example <a name="grid_world"></a>
### States / Goal:
- Assume, there is an agent in a world with only **four possible states** (here, marked by stone, brick, wood, or grass)
- **At the beginning** of an episode, the agent always starts **in state one** and 
- its **goal** is to reach **state four**, which is a terminal state.
- A big **wall** separating state one from state four.

### Actions:
- At each time step, the agent can move up, down, left, or right.
- But let's say the world is really slippery,
- So, it's also possible that the agent tries to go up but ends up slamming into a wall instead.
- Let's assume: it moves in that direction with 70% probability,
but ends up moving in one of the other directions with 10 percent probability each.
- If an agent runs into a wall at the next time step, it just ends up in the same state where it started. 

We have four states, four actions.

### Reward:
- The agent gets a reward of negative one for most transitions.
- But if it lands in the terminal state, it gets a reward of 10.
- Then this ensures that the goal of the agent will be to get to that terminal state as quickly as possible.
- Here: Discount rate is one (no discount).

    ![image1]

## Monte Carlo Methods <a name="mcm"></a> 
- Monte Carlo methods - even though the underlying problem involves a great degree of randomness, we can infer useful information that we can trust just by collecting a lot of samples.
- The equiprobable random policy is the stochastic policy where - from each state - the agent randomly selects from the set of available actions, and each action is selected with equal probability.

    ![image2]

## MC Prediction <a name="mcp"></a> 
- **Prediction Problem**: Given a policy, how might the agent estimate the value function for that policy?
- Idea: Start with the equiprobable random policy, and then use episodes in combination with the Q-table to find a better policy.
- To populate an entry in the Q-table, we use the return that followed when the agent was in that state, and chose the action. If the agent has selected in one state the same action many times, we need only average the returns.

- Algorithms that solve the prediction problem determine the value function **v<sub>π</sub>** (or **q<sub>π</sub>**) corresponding to a policy **π**.
- When working with finite MDPs, we can estimate the action-value function **q<sub>π</sub>** corresponding to a policy **π** in a table known as a **Q-table**. This table has one row for each state and one column for each action. The entry in the **s-th** row and **a-th** column contains the agent's estimate for expected return that is likely to follow, if the agent starts in state **s**, selects action aaa, and then henceforth follows the policy **π**.


    ![image3]

    - If the agent follows a policy for many episodes, we can use the results to directly estimate the action-value function corresponding to the same policy.
    - The Q-Table is used to estimate the action-value function

    ![image4]

    - What if, in the same episode, we select the same action from a state multiple times?
    - For instance, say that at time step two, we select action down from state three, and, say we do the same thing at time step 99.
    - If we count from the first time, then we get a return of negative 87, and if we count from the last time, then we get a return of 10.

    ![image5]

### Pseudocode
- Each occurrence of the state-action pair **s**, **a** (**s ∈ S**, **a ∈ A**) in an episode is called a visit to **s**, **a**.
- There are two types of MC prediction methods (for estimating **q<sub>π</sub>**):
    - ***First-visit MC*** estimates **q<sub>π</sub>(s,a)** as the average of the returns following ***only*** first visits to **s**, **a** (that is, it ignores returns that are associated to later visits.
    - ***Every-visit MC*** estimates **q<sub>π</sub>(s,a)** as the average of the returns following ***all*** visits to **s**, **a**.

    ![image6]

### There are three relevant tables:
- ***Q-table***, with a row for each state and a column for each action. The entry corresponding to state **s** and action **a** is denoted **Q(s,a)**.
- ***N-table*** that keeps track of the number of first visits we have made to each state-action pair.
- ***returns_sum***-table that keeps track of the sum of the rewards obtained after first visits to each state-action pair.

- In the algorithm, the number of episodes the agent collects is equal to **num_episodes**. After each episode, **N** and **returns_sum** are updated to store the information contained in the episode. Then, after all of the episodes have been collected and the values in **N** and **returns_sum** have been finalized, we quickly obtain the final estimate for **Q**.

- Both the first-visit and every-visit method are **guaranteed to converge** to the true action-value function, as the number of visits to each state-action pair approaches infinity. (So, in other words, as long as the agent gets enough experience with each state-action pair, the value function estimate will be pretty close to the true value.) In the case of first-visit MC, convergence follows from the [Law of Large Numbers]https://en.wikipedia.org/wiki/Law_of_large_numbers), and the details are covered in section 5.1 of the [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf).

- Every-visit MC is [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), whereas first-visit MC is unbiased (see Theorems 6 and 7).
- Initially, every-visit MC has lower mean squared error (MSE), but as more episodes are collected, first-visit MC attains better MSE (see Corollary 9a and 10a, and Figure 4).

## OpenAI Gym: BlackJackEnv <a name="blackjackenv"></a>  
- Each state is a 3-tuple of:
    - the player's current sum ∈{0,1,…,31}
    - the dealer's face up card ∈{1,…,10}
    - whether or not the player has a usable ace (no = 0, yes =1).

- The agent has two potential actions:
    STICK = 0
    HIT = 1

- Policy:
    - STICK (with 80% prop) if the sum of cards exceeds 18
    - HIT (with 80% prop) if the sum is 18 or below


- Open Jupyter Notebook ```monte_carlo_methods.ipynb```
    ### Necessary packages
    ```
    import sys
    import gym
    import numpy as np
    from collections import defaultdict

    from plot_utils import plot_blackjack_values, plot_policy
    ```
    Use the code cell below to create an instance of the [Blackjack](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py) environment.
    ```
    env = gym.make('Blackjack-v0')
    ```
    ### Observation and Action Space
    ```
    # Information about the environment. 
    # 707 different states: 704 = 32*11*2
    print(env.observation_space)
    print(env.action_space)
    ```
    ### Get some experience: Play Blackjack three times 
    ```
    # Play some sample games.
    # Equiprobrable policy with 50% hit or stick.
    for i_episode in range(3):
        state = env.reset()
        while True:
            print(state)
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print('End game! Reward: ', reward)
                print('You won :)\n') if reward > 0 else print('You lost :(\n')
                break

    RESULT:
    (21, 9, True)
    End game! Reward:  1.0
    You won :)

    (12, 10, False)
    (13, 10, False)
    End game! Reward:  -1
    You lost :(

    (16, 3, False)
    End game! Reward:  -1.0
    You lost :(
    ```
    ### Genertaing an episode
    ```
    def generate_episode_from_limit_stochastic(bj_env):
        """ Generating an episode with policy:
            - STICK (with 80% prop) if the sum of cards exceeds 18
            - HIT (with 80% prop) if the sum is 18 or below
            
            INPUTS:
            ------------
                bj_env - (OpenAI instance) instance of OpenAI Gym's Blackjack environment.
            
            OUTPUTS:
            ------------
                episode - (list of tuples) list of (state, action, reward) tuples, 
                        episode[i] returns (S_i, A_i, R_i+1)
        """
        episode = []
        state = bj_env.reset()
        while True:
            probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=probs)
            next_state, reward, done, info = bj_env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode
    ```
    ### Play Blackjack with the policy three times
    ```
    for i in range(3):
        print(generate_episode_from_limit_stochastic(env))
    
    RESULT:
    [((20, 8, False), 1, -1)]
    [((17, 2, False), 0, -1.0)]
    [((21, 10, True), 0, 1.0)]
    ```
    ```
    episode = generate_episode_from_limit_stochastic(env)
    print(episode)
    states, actions, rewards = zip(*episode)
    print('\nstates: ', states)
    print('\nactions: ', actions)
    print('\nrewards: ', rewards)

    RESULT:
    [((12, 6, False), 1, 0), ((17, 6, False), 1, -1)]
    states:  ((12, 6, False), (17, 6, False))
    actions:  (1, 1)
    rewards:  (0, -1)
    ```
    ### First-Visit MC Prediction
    ```
    def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
        """ First-Visit MC Prediction (for action values)
        
            INPUTS:
            ------------
                env - (OpenAI instance) instance of an OpenAI Gym environment.
                num_episodes - (int) number of episodes that are generated through agent-environment interaction.
                generate_episode - (function) function that returns an episode of interaction.
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1).

            
            OUTPUTS:
            ------------
                Q - (dictionary of one-dimensional arrays) 
                    where Q[s][a] is the estimated action value corresponding to state s and action a
        """

        # initialize empty dictionaries of arrays
        returns_sum = defaultdict(lambda: np.zeros(env.action_space.n)) # sum of the rewards table
        N = defaultdict(lambda: np.zeros(env.action_space.n)) # N-table
        Q = defaultdict(lambda: np.zeros(env.action_space.n)) # Q-table
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            # generate an episode
            episode = generate_episode(env)
            # obtain the states, actions, and rewards
            states, actions, rewards = zip(*episode)
            # prepare for discounting
            discounts = np.array([gamma**i for i in range(len(rewards)+1)])
            # update the sum of the returns, number of visits, and action-value 
            # function estimates for each state-action pair in the episode
            for i, state in enumerate(states):
                returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
                N[state][actions[i]] += 1.0
                Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
            
        return Q
    ```
    ### Action-value function estimate 𝑄
    ```
    # obtain the action-value function
    Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

    # obtain the corresponding state-value function
    V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
            for k, v in Q.items())

    # plot the state-value function
    plot_blackjack_values(V_to_plot)
    ```
    ### Plot of the corresponding state-value function:

    ![image7]




## Greedy Policies <a name="greedy_policies"></a> 
- A policy is greedy with respect to an action-value function estimate **Q** if for every state **s ∈ S**, it is guaranteed to select an action **a ∈ A(s)** such that **a = argmax<sub>a∈A(s)</sub>Q(s,a)**. (It is common to refer to the selected action as the greedy action.)
- In the case of a finite MDP, the action-value function estimate is represented in a Q-table. Then, to get the greedy action(s), for each row in the table, we need only select the action (or actions) corresponding to the column(s) that maximize the row.





## Epsilon-Greedy Policies <a name="epsilon_greedy_policies"></a> 
- A policy is **ϵ-greedy** with respect to an action-value function estimate **Q** if for every state **s ∈ S**,
    - with probability **1 − ϵ**1, the agent selects the greedy action, and
    - with probability **ϵ**, the agent selects an action uniformly at random from the set of available (non-greedy AND greedy) actions.


## MC Control <a name="mc_control"></a>
- Algorithms designed to solve the **control problem** determine the optimal policy **π<sub>∗</sub>** from interaction with the environment.
- The **Monte Carlo control method** uses alternating rounds of policy evaluation and improvement to recover the optimal policy.


## Exploration vs. Exploitation <a name="explorat_vs_exploit"></a> 
- All reinforcement learning agents face the **Exploration-Exploitation Dilemma**, where they must find a way to balance the drive to behave optimally based on their current knowledge (exploitation) and the need to acquire knowledge to attain better judgment (exploration).
- In order for MC control to converge to the optimal policy, the Greedy in the Limit with Infinite Exploration (GLIE) conditions must be met:
    - every state-action pair **s**, **a**, (for all **s ∈ S** and **a ∈ A(s)** is visited infinitely many times, and
    - the policy converges to a policy that is greedy with respect to the action-value function estimate **Q**.


## Incremental Mean <a name="inc_mean"></a> 
- (In this concept, we amended the policy evaluation step to update the Q-table after every episode of interaction.)

## Constant-alpha <a name="const_alpha"></a> 
- (In this concept, we derived the algorithm for **constant-α** MC control, which uses a constant step-size parameter **α**.)
- The step-size parameter **α** must satisfy **0 < α ≤ 1**. Higher values of **α** will result in faster learning, but values of **α** that are too high can prevent MC control from converging to **π<sub>∗</sub>**.





## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)