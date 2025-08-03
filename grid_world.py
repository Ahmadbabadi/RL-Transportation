import numpy as np
from tqdm import tqdm


def get_reward(state):
    if state == (4, 4):
        return 100
    else:
        return -1

def get_cost(action):
    """
    cost = value_time * 1/velocity + value_price * price
    value_time is value of losing time for agent
    value_price is value of losing price for agent
    """
    value_time = 5
    value_price = 1
    if action[0] == "W":
        v, p = 1, 0
    elif action[0] == "B":
        v, p = 3, 0.5
    elif action[0] == "T":
        v, p = 7, 1.453
    cost = value_time * 1/v + value_price * p
    return -cost
    


def state_to_index(state):
    return state[0] * grid_size + state[1]

def is_valid(pos):
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

def epsilon_greedy(episodes):
    Q_xy_a = np.zeros((grid_size * grid_size, n_actions))
    # for episode in trange(episodes):
    for episode in range(episodes):
        state = (0, 0)
        while state != (4, 4):
            s_idx = state_to_index(state)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, n_actions - 1)
            else:
                action_idx = np.argmax(Q_xy_a[s_idx])
    
            action = actions[action_idx]
            delta = action_to_delta[action]
            next_state = (state[0] + delta[0], state[1] + delta[1])
    
            if not is_valid(next_state):
                next_state = state
    
            R_C = get_reward(next_state) + get_cost(action)
            ns_idx = state_to_index(next_state)

            Q_xy_a[s_idx][action_idx] += alpha * (R_C + gamma * np.max(Q_xy_a[ns_idx]) - Q_xy_a[s_idx][action_idx])

            state = next_state
    return Q_xy_a



if __name__ == "__main__":
    grid_size = 5
    # direction: up, down, left, right --> U, D, L, R
    # transportaion: walk, bike, taxi -->  W, B, T
    action_to_delta = {
                    'WU': (-1, 0),
                    'WD': (1, 0),
                    'WL': (0, -1),
                    'WR': (0, 1),
                    'BU': (-1, 0),
                    'BD': (1, 0),
                    'BL': (0, -1),
                    'BR': (0, 1),
                    'TU': (-1, 0),
                    'TD': (1, 0),
                    'TL': (0, -1),
                    'TR': (0, 1),
                    }
    actions = list(action_to_delta.keys())
    n_actions = len(actions)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    episodes = 100000
    Q_xy_a = epsilon_greedy(episodes)

    print("Q-vals")
    print(np.round(Q_xy_a, 2), end="\n\n")

    print("\nLearned policy:")
    policy_grid = []
    for row in range(grid_size):
        row_policy = []
        for col in range(grid_size):
            idx = state_to_index((row, col))
            best_action = actions[np.argmax(Q_xy_a[idx])] + str(np.abs(np.round(np.max(Q_xy_a[idx]), 2)))
            row_policy.append(best_action)
        policy_grid.append(row_policy)

    for row in policy_grid:
        print(" ".join(row))