def evaluate_policy(policy,
                    env_maker,
                    episodes: int,
                    max_steps: int,
                    rng: np.random.RandomState):
    """
    Returns (fitness, avg_reward, strict_success_rate).
    strict_success = used_key == True AND agent == goal_pos at end of episode.
    """
    total_reward = 0.0

    strict_successes = 0      # key+door+goal in correct order
    goal_hits = 0             # reached goal regardless of key usage
    key_episodes = 0          # picked up key at some point (or used it)
    door_open_eps = 0         # used_key == True

    for _ in range(episodes):
        env = env_maker()
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = select_action(obs)
            obs, reward, done, _info = env.step(action)
            ep_reward += reward
            steps += 1

        total_reward += ep_reward

        # End-of-episode stats
        agent_pos = getattr(env, "agent", None)
        goal_pos  = getattr(env, "goal_pos", None)
        has_key   = getattr(env, "has_key", False)
        used_key  = getattr(env, "used_key", False)

        at_goal = (agent_pos == goal_pos)
        if at_goal:
            goal_hits += 1
        if has_key or used_key:
            key_episodes += 1
        if used_key:
            door_open_eps += 1
        if at_goal and used_key:
            strict_successes += 1

    avg_reward   = total_reward / max(1, episodes)
    strict_sr    = strict_successes / max(1, episodes)
    goal_rate    = goal_hits        / max(1, episodes)
    key_rate     = key_episodes     / max(1, episodes)
    door_rate    = door_open_eps    / max(1, episodes)

    # ---- Fitness shaping ----
    # - avg_reward: general behavior
    # - +0.2 * key_rate: encourage picking up key
    # - +0.3 * door_rate: encourage opening door
    # - +0.5 * goal_rate: encourage hitting goal at all
    # - +1.0 * strict_sr: bonus for fully correct sequence
    fitness = (
        avg_reward
        + 0.2 * key_rate
        + 0.3 * door_rate
        + 0.5 * goal_rate
        + 1.0 * strict_sr
    )

    return fitness, avg_reward, strict_sr
