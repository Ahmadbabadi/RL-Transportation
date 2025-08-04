import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from string import ascii_uppercase


class DistrictEnv:
    def __init__(self, adj_matrix, num_nodes):
        self.adj_matrix = adj_matrix
        self.goal = np.zeros(num_nodes, dtype=np.float32)
        self.goal[-1] = 1
        self.num_nodes = num_nodes

    def reset(self):
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.state[np.random.randint(self.num_nodes-1)] = 1
        return self.state
    
    def get_cost(self, action):
        v_p = np.array([[1, 0], [3, 1], [9, 3]]) 
        v, p =  v_p[np.argmax(action[-3:])]
        value_time = 1
        value_price = 2
        distance = self.adj_matrix[np.argmax(self.state)][np.argmax(action[:-3])]
        cost = value_time*(distance/v)**1.8 + distance**0.5 * p * value_price
        return -cost

    def step(self, action):
        if self.adj_matrix[np.argmax(self.state)][np.argmax(action[:-3])] != 0:
            self.state = action[:-3]
        # reward
        reward = 100 if np.argmax(self.state) == np.argmax(self.goal) else -1
        # cost
        cost = self.get_cost(action)
        done = (np.argmax(self.state) == np.argmax(self.goal))
        return self.state, reward + cost, done


def get_feature_vector(state, action, num_nodes):
    vec = torch.zeros(2 * num_nodes + 3, dtype=torch.float32)
    vec[:num_nodes] = torch.from_numpy(state)
    vec[num_nodes:] = torch.from_numpy(action)
    return vec

def get_actions_of_state(adj_matrix, state):
    actions = []
    for next_s in np.argwhere(adj_matrix[np.argmax(state)] > 0):
        for vehicle in range(3):
            act = np.zeros(num_nodes + 3, dtype=np.float32)
            act[next_s] = 1
            act[num_nodes+vehicle] = 1
            actions.append(act)
    return actions


class LinearQNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

adj_matrix = np.load("adjacency_matrix.npy", allow_pickle=False)
num_nodes = adj_matrix.shape[0]
env = DistrictEnv(adj_matrix, num_nodes)
episodes = 1000
input_dim = 2 * num_nodes +3
model = LinearQNet(input_dim)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
gamma = 0.9
epsilon = 0.1

for episode in tqdm(range(episodes)):
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 60:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.zeros(num_nodes+3, dtype=np.float32)
            action[np.random.randint(num_nodes)] = 1
            action[num_nodes + np.random.randint(3)] = 1
        else:
            q_vals = []
            possible_actions = get_actions_of_state(env.adj_matrix, state)
            for act in possible_actions:
                phi = get_feature_vector(state, act, num_nodes)
                with torch.no_grad():
                    q_vals.append(model(phi).item())
            best_idx = torch.argmax(torch.tensor(q_vals))
            action = possible_actions[best_idx]

        # Step in environment
        next_state, reward, done = env.step(action)

        # Target computation
        if np.sum(env.adj_matrix[next_state==1])>0:
            next_qs = []
            possible_actions = get_actions_of_state(env.adj_matrix, next_state)
            for act_next in possible_actions:
                phi_next = get_feature_vector(next_state, act_next, num_nodes)
                with torch.no_grad():
                    next_qs.append(model(phi_next).item())
            max_q_next = max(next_qs)
        else:
            max_q_next = 0.0

        td_target = reward + gamma * max_q_next

        # Update model
        phi = get_feature_vector(state, action, num_nodes)
        q_pred = model(phi)
        loss = loss_fn(q_pred, torch.tensor(td_target, dtype=torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        steps += 1

# print(env.adj_matrix)
print("Final weights:")
for name, param in model.named_parameters():
    print(name, param.data)
    
def select_best_action(state):
    best_action = None
    max_q = float('-inf')
    possible_actions = get_actions_of_state(env.adj_matrix, state)
    for action in possible_actions:
        x = get_feature_vector(state, action, num_nodes)
        q_val = model(torch.tensor(x, dtype=torch.float32)).item()
        if q_val > max_q:
            max_q = q_val
            best_action = action
    return best_action, round(max_q, 2)

states = np.zeros((15, 15), dtype=np.float32)
for i in range(15):
    states[i, i] = 1

for s in states:
    best_action, max_q = select_best_action(s)
    print(ascii_uppercase[ np.argmax(best_action[:-3]) ],
          best_action[-3:],
          f"Q(s, a) = { max_q }")
    
s = np.zeros((15), dtype=np.float32)
s[np.random.randint(14)] = 1 

g = np.zeros((15), dtype=np.float32)
g[-1] = 1 
while not np.all(s == g):
    best_action, max_q = select_best_action(s)     
    print(ascii_uppercase[ np.argmax(s) ], "--->",
          ascii_uppercase[ np.argmax(best_action[:-3]) ])
    s = best_action[:-3]

    
    