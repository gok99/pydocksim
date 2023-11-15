import math
from enum import Enum
from dataclasses import dataclass
from visualiser import visualise_game_state
from visualiser import Action
import numpy as np

import matplotlib.pyplot as plt

# actions import from visualiser: 
# F = forward L meters
# B = backward L meters
# R = move right L meters
# L = move left L meters
# Encircle(p, d) = encircle point p in d direction

class Direction(Enum):
    CW = 0
    CCW = 1

# The task of the USV is to (Assume animal positions are known and given): 
# 1) encircle the platypus (P) CW
# 2) encircle the turtle (T) CCW
# 3) remain 10m away from the crocodile (C)

@dataclass
class GameState:
    start_pos: tuple   # (x, y) start position of USV
    usv_dims: tuple    # (width, height) of USV
    P: tuple           # (x, y) position of platypus
    T: tuple           # (x, y) position of turtle    
    C: tuple           # (x, y) position of crocodile
    grid_length: int   # length of grid
    move_time: float   # time to move 1 grid length
    circle_time: float # time to encircle 1m
    time_limit: float  # time limit for episode
    time: float        # time elapsed so far

@dataclass
class State:
    pos: tuple       # (x, y) position of USV
    P: bool          # True if encircled
    T: bool          # True if encircled
    C: bool          # True if > 10m away so far
    time: float      # time elapsed so far

# rewards:
# -1 for each time step
# +30 for encircling platypus (if not collided with animal)
# +30 for encircling turtle   (if not collided with animal)
# +30 for staying > 10m away from crocodile (at the end of the episode)

def pseudo_vrx_score(x, y, ae, game_state: GameState):
    reward = 0
    reward -= game_state.time
    P, T, C = get_animal_decoding(ae)
    
    if P:
        reward += 30
    if T:
        reward += 30
    
    time_exceeded = (x + y) >= game_state.time_limit
    game_over = (P and T) or time_exceeded
    
    if game_over and C:
        reward += 30
    
    return reward

# coord convention:
#  y
#  |
#  ---> x
def will_collide(x, y, game_state: GameState, action: Action, pc: tuple = None):
    # 
    # radius = inner_radius = outer_radius = bottom_y = top_y = left_x = right_x = 0

    def encircle_movement_check():
        radius = math.sqrt((x - pc[0])**2 + (y - pc[1])**2)
        inner_radius = radius - game_state.usv_dims[1]/2
        outer_radius = radius + game_state.usv_dims[1]/ 2

        return lambda animal_pos : \
            inner_radius <= math.sqrt((animal_pos[0] - pc[0])**2 + (animal_pos[1] - pc[1])**2) <= outer_radius

    def pos_movement_check():  
        bottom_y = y - game_state.usv_dims[1]/2
        top_y = y + game_state.usv_dims[1]/2
        left_x = x - game_state.usv_dims[0]/2
        right_x = x + game_state.usv_dims[0]/2

        if action == Action.F:
            top_y += game_state.grid_length
        elif action == Action.B:
            bottom_y -= game_state.grid_length
        elif action == Action.R:
            right_x += game_state.grid_length
        elif action == Action.L:
            left_x -= game_state.grid_length

        return lambda animal_pos : \
            bottom_y <= animal_pos[1] <= top_y and \
            left_x <= animal_pos[0] <= right_x
    
    if action == Action.Encircle:
        movement_check = encircle_movement_check()
    else:
        movement_check = pos_movement_check()

    for animal_pos in [game_state.P, game_state.T, game_state.C]:
        if movement_check(animal_pos):
            return True

def position_encircled(usv_pos: tuple, circle_pos: tuple, test_pos: tuple):
    radius = (usv_pos[0] - circle_pos[0])**2 + (usv_pos[1] - circle_pos[1])**2
    return (test_pos[0] - circle_pos[0])**2 + (test_pos[1] - circle_pos[1])**2 <= radius

def get_animal_encoding(P: bool, T: bool, C: bool):
    pv = 1 if P else 0
    tv = 1 if T else 0
    cv = 1 if C else 0
    return pv * 4 + tv * 2 + cv

def get_animal_decoding(encoding: int):
    P = encoding // 4 == 1
    encoding %= 4
    T = encoding // 2 == 1
    encoding %= 2
    C = encoding == 1
    return (P, T, C)

def next_state(x, y, ae, a, gs: GameState):
    new_x, new_y = x, y
    action, pc, dir = a
    new_P, new_T, new_C = get_animal_decoding(ae)

    if action == Action.Encircle:
        if position_encircled((x, y), pc, gs.P) and dir == Direction.CW:
            new_P = True
        
        if position_encircled((x, y), pc, gs.T) and dir == Direction.CCW:
            new_T = True

        pc_to_croc = math.sqrt((pc[0] - gs.C[0])**2 + (pc[1] - gs.C[1])**2)
        pc_to_usv = math.sqrt((pc[0] - x)**2 + (pc[1] - y)**2)
        if pc_to_croc - pc_to_usv < 10:
            new_C = False
    else:
        if action == Action.F:
            new_y += gs.grid_length
        elif action == Action.B:
            new_y -= gs.grid_length
        elif action == Action.R:
            new_x += gs.grid_length
        elif action == Action.L:
            new_x -= gs.grid_length

        usv_to_croc = (new_x - gs.C[0])**2 + (new_y - gs.C[1])**2
        if usv_to_croc < 100:
            new_C = False
    
    return (new_x, new_y, get_animal_encoding(new_P, new_T, new_C))

# does not perform validation. only use actions from available_actions
def reward(x, y, ae, a, gs: GameState):
    delta_reward = 0
    
    action, pc, _ = a
    P, T, C = get_animal_decoding(ae)

    new_x, new_y, new_ae = next_state(x, y, ae, a, gs)
    new_P, new_T, new_C = get_animal_decoding(new_ae)

    if action == Action.F or action == Action.B or action == Action.R or action == Action.L:
        delta_reward -= gs.move_time * 0.8
    else: 
        # action is encircle
        twopir = 2 * math.pi * math.sqrt((x - pc[0])**2 + (y - pc[1])**2)
        delta_reward -= twopir * gs.circle_time
    
    if new_P and not P:
        delta_reward += 30
    
    if new_T and not T:
        delta_reward += 30

    if new_P and new_T and (not P or not T):
        delta_reward += 0

    if not new_C and C:
        delta_reward -= 30
    
    return (delta_reward, (new_x, new_y, new_ae))

def available_actions(x, y, game_state: GameState, lb, rb, bb, tb):
    actions = []
    curr_pos = (x, y)
    def in_bounds(x, y):
        return lb <= x <= rb and bb <= y <= tb

    # check if moving is possible
    if not will_collide(x, y, game_state, Action.F):
        x = curr_pos[0]
        y = curr_pos[1] + game_state.grid_length
        if in_bounds(x, y):
            actions.append((Action.F, None, None))
    if not will_collide(x, y, game_state, Action.B):
        x = curr_pos[0]
        y = curr_pos[1] - game_state.grid_length
        if in_bounds(x, y):
            actions.append((Action.B, None, None))
    if not will_collide(x, y, game_state, Action.R):
        x = curr_pos[0] + game_state.grid_length
        y = curr_pos[1]
        if in_bounds(x, y):
            actions.append((Action.R, None, None))
    if not will_collide(x, y, game_state, Action.L):
        x = curr_pos[0] - game_state.grid_length
        y = curr_pos[1]
        if in_bounds(x, y):
            actions.append((Action.L, None, None))
    if not will_collide(x, y, game_state, Action.Encircle, game_state.P):
        actions.append((Action.Encircle, game_state.P, Direction.CW))
        actions.append((Action.Encircle, game_state.P, Direction.CCW))
    if not will_collide(x, y, game_state, Action.Encircle, game_state.T):
        actions.append((Action.Encircle, game_state.T, Direction.CW))
        actions.append((Action.Encircle, game_state.T, Direction.CCW))
    
    return actions

def get_vi_utilities(game_state: GameState):
    # actions should be appended to a list according to planner
    left_bound = int(min(0, game_state.P[0], game_state.T[0], game_state.C[0])) - 2
    right_bound = int(max(0, game_state.P[0], game_state.T[0], game_state.C[0])) + 2

    bottom_bound = int(min(0, game_state.P[1], game_state.T[1], game_state.C[1])) - 2
    top_bound = int(max(0, game_state.P[1], game_state.T[1], game_state.C[1])) + 2

    def idx(x):
        return (x - left_bound) // game_state.grid_length
    
    def idy(y):
        return (y - bottom_bound) // game_state.grid_length

    print(left_bound, right_bound, bottom_bound, top_bound)

    # precompute actions
    pre_actions = { (x, y) : available_actions(x, y, game_state, left_bound, right_bound, bottom_bound, top_bound) for 
                    x in range(left_bound, right_bound + 1, game_state.grid_length) for
                    y in range(bottom_bound, top_bound + 1, game_state.grid_length) }

    WIDTH = len(range(left_bound, right_bound + 1, game_state.grid_length))
    HEIGHT = len(range(bottom_bound, top_bound + 1, game_state.grid_length))

    # perform value iteration
    value_policy = np.zeros((WIDTH, HEIGHT, 8))

    for x in range(left_bound, right_bound + 1, game_state.grid_length):
        for y in range(bottom_bound, top_bound + 1, game_state.grid_length):
            for s in range(8):
                value_policy[idx(x)][idy(y)][s] = 0 # pseudo_vrx_score(x, y, s, game_state)

    max_iter = 10000
    gamma = 1

    # value iteration
    for i in range(max_iter):
        delta = 0
        for x in range(left_bound, right_bound + 1, game_state.grid_length):
            for y in range(bottom_bound, top_bound + 1, game_state.grid_length):
                for s in range(8):
                    new_v = -10 # value_policy[idx(x)][idy(y)][s]
                    actions = pre_actions[x, y]

                    P, T, _ = get_animal_decoding(s)
                    if P and T:
                        continue

                    for a in actions:
                        delta_reward, new_state = reward(x, y, s, a, game_state)
                        new_v = max(new_v, delta_reward + gamma * value_policy[idx(new_state[0])][idy(new_state[1])][new_state[2]])
    
                    delta = max(delta, abs(new_v - value_policy[idx(x)][idy(y)][s]))
                    value_policy[idx(x)][idy(y)][s] = new_v
        print (i, delta)
        if delta < 0.01:
            break
    
    return value_policy


def main():
    game_states = [GameState(
        start_pos = (0, 0),
        usv_dims = (1, 1),
        P = (0, 10),
        T = (10, 0),
        C = (15, 15),
        grid_length = 1,
        time_limit = 100,
        move_time = 1,
        circle_time = 1,
        time = 0)
    # ), GameState(
    #     start_pos = (0, 0),
    #     usv_dims = (1, 1),
    #     P = (39, 39),
    #     T = (10, 10),
    #     C = (30, 30),
    #     grid_length = 1,
    #     time_limit = 100,
    #     move_time = 1,
    #     circle_time = 1,
    #     time = 0
    # ), GameState(
    #     start_pos = (0, 0),
    #     usv_dims = (1, 1),
    #     P = (20, 20),
    #     T = (10, 10),
    #     C = (-20, 0),
    #     grid_length = 1,
    #     time_limit = 100,
    #     move_time = 1,
    #     circle_time = 1,
    #     time = 0
    # )]
    ]

    #####################
    ## U estimation with NN
    
    import torch
    from torch import nn
    from torch.optim import Adam

    # Step 1: Prepare data
    X = []
    Y = []

    for game_state in game_states:
        left_bound = int(min(0, game_state.P[0], game_state.T[0], game_state.C[0])) - 2
        right_bound = int(max(0, game_state.P[0], game_state.T[0], game_state.C[0])) + 2
        bottom_bound = int(min(0, game_state.P[1], game_state.T[1], game_state.C[1])) - 2
        top_bound = int(max(0, game_state.P[1], game_state.T[1], game_state.C[1])) + 2

        def idx(x):
            return (x - left_bound) // game_state.grid_length
    
        def idy(y):
            return (y - bottom_bound) // game_state.grid_length

        P = game_state.P
        T = game_state.T
        C = game_state.C

        value_policy = get_vi_utilities(game_state)

        for x in range(left_bound, right_bound + 1, game_state.grid_length):
            for y in range(bottom_bound, top_bound + 1, game_state.grid_length):
                for ae in range(8):
                    X.append([x, y, P[0], P[1], T[0], T[1], C[0], C[1], ae])
                    Y.append(value_policy[idx(x)][idy(y)][ae])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    # Step 2: Define model
    model = nn.Sequential(
        nn.Linear(9, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # Step 3: Train
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(10000):  # Adjust as needed
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, Y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    #####################
    game_state = game_states[-1]
    
    left_bound = int(min(0, game_state.P[0], game_state.T[0], game_state.C[0])) - 2
    right_bound = int(max(0, game_state.P[0], game_state.T[0], game_state.C[0])) + 2

    bottom_bound = int(min(0, game_state.P[1], game_state.T[1], game_state.C[1])) - 2
    top_bound = int(max(0, game_state.P[1], game_state.T[1], game_state.C[1])) + 2

    P = game_state.P
    T = game_state.T
    C = game_state.C

    test_X = [[x, y, P[0], P[1], T[0], T[1], C[0], C[1], ae] for x in range(left_bound, right_bound + 1, game_state.grid_length) for y in range(bottom_bound, top_bound + 1, game_state.grid_length) for ae in range(8)]
    test_X = torch.tensor(test_X, dtype=torch.float32)

    # Make sure the model is in evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        test_predictions = model(test_X)

    # Convert predictions to numpy array
    test_predictions = test_predictions.numpy()

    print(test_predictions.shape)

    WIDTH = len(range(left_bound, right_bound + 1, game_state.grid_length))
    HEIGHT = len(range(bottom_bound, top_bound + 1, game_state.grid_length))

    value_policy = np.zeros((WIDTH, HEIGHT, 8))

    i = 0
    for x in range(value_policy.shape[0]):
        for y in range(value_policy.shape[1]):
            for s in range(8):
                value_policy[x][y][s] = test_predictions[i]
                i += 1

    #####################
    # Utility heatmap

    utility_values = np.zeros((WIDTH, HEIGHT))

    for x in range(value_policy.shape[0]):
        for y in range(value_policy.shape[1]):
            # Assume s=0 for simplicity. Change this if you have different states.
            utility_values[x][y] = value_policy[x][y][1]

    utility_values = np.transpose(utility_values)

    plt.imshow(utility_values, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Utility values')
    plt.show()

    #####################

    actions = []
    curr_state = (0, 0, get_animal_encoding(False, False, True))
    count = 0

    pre_actions = { (x, y) : available_actions(x, y, game_state, left_bound, right_bound, bottom_bound, top_bound) for 
                x in range(left_bound, right_bound + 1, game_state.grid_length) for
                y in range(bottom_bound, top_bound + 1, game_state.grid_length) }

    while curr_state[2] < 6 and count < 100:
        x, y, ae = curr_state
        pactions = pre_actions[x, y]

        best_action = None
        best_reward = -100000

        print (x, y, ae)

        P, T, _ = get_animal_decoding(ae)
        best_action_found = False
        CUTOFF = 3

        new_dist_to_plat = math.sqrt((x - game_state.P[0])**2 + (y - game_state.P[1])**2)
        new_dist_to_turt = math.sqrt((x - game_state.T[0])**2 + (y - game_state.T[1])**2)

        if new_dist_to_plat <= CUTOFF and not P:
            best_action = (Action.Encircle, game_state.P, Direction.CW)
            best_action_found = True
                
        if new_dist_to_turt <= CUTOFF and not T:
            best_action = (Action.Encircle, game_state.T, Direction.CCW)
            best_action_found = True

        if not best_action_found:
            for a in pactions:
                rew, ns = reward(x, y, ae, a, game_state)
                new_x, new_y, new_ae = ns
                utility = rew + value_policy[idx(new_x)][idy(new_y)][new_ae]
            
                if utility > best_reward:
                    best_action = a
                    best_reward = utility

        if best_action is None:
            break

        print(best_action)
        print("<<<<<<<<<<<<<<<<<<<<<<<<")

        if best_action[0] == Action.Encircle:
            actions.append(best_action[0])
            actions.append(best_action[1])
        else:
            actions.append(best_action[0])

        curr_state = next_state(x, y, ae, best_action, game_state)
        count += 1

    # print ()

    print (actions)
    visualise_game_state(game_state, actions)

if __name__ == "__main__":
    main()
