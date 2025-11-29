# experiments/sanity_tinygrid.py
from tasks.tinygrid import TinyGrid

def main():
    # Create easy 7x7 grid
    env = TinyGrid(size=7, max_steps=50, difficulty="easy")
    env.reset(seed=0)

    print("Initial layout:")
    env.render()
    print("agent:", env.agent, "key:", env.key_pos,
          "door:", env.door_pos, "goal:", env.goal_pos)

    # ---- CHEAT 1: Pick up key instantly ----
    env.agent = env.key_pos
    env.has_key = True
    env.grid[env.key_pos] = 0

    print("\nAfter picking up key:")
    env.render()

    # ---- CHEAT 2: Open door instantly ----
    env.agent = env.door_pos
    env.used_key = True
    env.grid[env.door_pos] = 0

    print("\nAfter opening door:")
    env.render()

    # ---- CHEAT 3: Move to goal ----
    env.agent = env.goal_pos

    print("\nAt goal:")
    env.render()

    # Trigger logic by doing 1 action that DOES NOT MOVE (action 4)
    obs, reward, done, info = env.step(4)

    print("\nAfter one more step to finalize episode:")
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
    print("agent:", env.agent, "goal:", env.goal_pos)
    print("has_key:", env.has_key, "used_key:", env.used_key)


if __name__ == "__main__":
    main()