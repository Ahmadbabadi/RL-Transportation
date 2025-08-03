import numpy as np
from tqdm import trange


def get_reward(state):
    if state == (4, 4):  
        return 10
    else:
        return -1 

def state_to_index(state):
    return state[0] * grid_size + state[1]

def is_valid(pos):
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size



def epsilon_greedy(epsilon, gamma, alpha, episodes):
    Q = np.zeros((grid_size * grid_size, n_actions))
    # for episode in trange(episodes):
    for episode in range(episodes):
        state = (0, 0)
        while state != (4, 4): 
            s_idx = state_to_index(state)

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, n_actions - 1)
            else:
                action_idx = np.argmax(Q[s_idx])

            action = actions[action_idx]
            delta = action_to_delta[action]
            next_state = (state[0] + delta[0], state[1] + delta[1])

            if not is_valid(next_state):
                next_state = state 

            r = get_reward(next_state)
            ns_idx = state_to_index(next_state)

            Q[s_idx][action_idx] += alpha * (r + gamma * np.max(Q[ns_idx]) - Q[s_idx][action_idx])

            state = next_state  
    return Q


if __name__ == "__main__":
    grid_size = 5

    action_to_delta = {
                        'U': (-1, 0),
                        'D': (1, 0),
                        'L': (0, -1),
                        'R': (0, 1),
                        }
    actions = list(action_to_delta.keys())
    n_actions = len(actions)

    
    alpha = 0.1    
    gamma = 0.9     
    epsilon = 0.2   
    episodes = 50000
    
    Q = epsilon_greedy(epsilon, gamma, alpha, episodes)

    print("Q-vals")
    print(np.round(Q, 2), end="\n\n")

    print("best policy of each state")
    policy_grid = []
    for row in range(grid_size):
        row_policy = []
        for col in range(grid_size):
            idx = state_to_index((row, col))
            best_action = actions[np.argmax(Q[idx])]
            row_policy.append(best_action)
        policy_grid.append(row_policy)

    for row in policy_grid:
        print(" ".join(row))