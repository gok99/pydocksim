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
    game_state = GameState(
        start_pos = (0, 0),
        usv_dims = (1, 1),
        # case 1
        # P = (0, 10),
        # T = (10, 0),
        # C = (15, 15),

        # case 2
        # P = (0, 30),
        # T = (15, 15),
        # C = (0, 15),

        # case 3
        P = (39, 39),
        T = (10, 10),
        C = (30, 30),

        # case 4
        # P = (-7, 15),
        # T = (15, 15),
        # C = (0, 15), 

        # case 5
        # P = (-7, 15),
        # T = (7, 15),
        # C = (0, 15),

        grid_length = 1,
        time_limit = 100,
        move_time = 1,
        circle_time = 1,
        time = 0
    )

    #####################
    # Utility heatmap

    # utility_values = np.zeros((WIDTH, HEIGHT))

    # for x in range(value_policy.shape[0]):
    #     for y in range(value_policy.shape[1]):
    #         # Assume s=0 for simplicity. Change this if you have different states.
    #         utility_values[x][y] = value_policy[x][y][1]

    # utility_values = np.transpose(utility_values)

    # plt.imshow(utility_values, cmap='hot', interpolation='nearest', origin='lower')
    # plt.colorbar(label='Utility values')
    # plt.show()
    #####################
    # potentials heatmap

    game_state = game_state
    
    left_bound = int(min(0, game_state.P[0], game_state.T[0], game_state.C[0])) - 2
    right_bound = int(max(0, game_state.P[0], game_state.T[0], game_state.C[0])) + 2

    bottom_bound = int(min(0, game_state.P[1], game_state.T[1], game_state.C[1])) - 2
    top_bound = int(max(0, game_state.P[1], game_state.T[1], game_state.C[1])) + 2

    P = game_state.P
    T = game_state.T
    C = game_state.C

    WIDTH = len(range(left_bound, right_bound + 1, game_state.grid_length))
    HEIGHT = len(range(bottom_bound, top_bound + 1, game_state.grid_length))

    pre_actions = { (x, y) : available_actions(x, y, game_state, left_bound, right_bound, bottom_bound, top_bound) for
                    x in range(left_bound, right_bound + 1, game_state.grid_length) for
                    y in range(bottom_bound, top_bound + 1, game_state.grid_length) }

    # function approximation
    # U(s) = t0 + t1 * f1(s) + t2 * f2(s) + t3 * f3(s) + t4 * f4(s) + ...
    # use value_policy as training data

    # constant 1
    def f0(x, y, P, T, C):
        return 1

    # manhattan from P
    def f1(x, y, P, T, C):
        return (1 - P) / ((x - game_state.P[0])**2 + (y - game_state.P[1])**2)**0.5
        # (3 - ((x - game_state.P[0])**2 + (y - game_state.P[1])**2))
        
    # manhattan from T
    def f2(x, y, P, T, C):
        return (1 - T) / ((x - game_state.T[0])**2 + (y - game_state.T[1])**2)**0.5
        # (3 - ((x - game_state.T[0])**2 + (y - game_state.T[1])**2)) 
    
    # manhattan from C
    def f3(x, y, P, T, C):
        croc_dist = math.sqrt((x - game_state.C[0])**2 + (y - game_state.C[1])**2)
        return (1.2 if croc_dist < 10 else 0.9)
        # (C) / ((x - game_state.C[0])**2 + (y - game_state.C[1])**2)**0.15
        # (1 - ((x - game_state.C[0])**2 + (y - game_state.C[1])**2)) 

    # add constant reward for P encircled
    def f4(x, y, P, T, C):
        return P * 3
    
    # add constant reward for T encircled
    def f5(x, y, P, T, C):
        return T * 3

    def f6(x, y, P, T, C):
        return (1 - C) * 3

    alpha = 0.01
    max_iter = 10000
    NUM_FEATURES = 4
    params = [3, 1, 1, -3]
    fns = [f0, f1, f2, f3]

    def U(x, y, P, T, C):
        eval = [params[i] * fns[i](x, y, P, T, C) for i in range(NUM_FEATURES)]
        return sum(eval)

    #####################
    # potentials heatmap

    potential_values = np.zeros((WIDTH, HEIGHT))

    for x in range(left_bound, right_bound + 1, game_state.grid_length):
        for y in range(bottom_bound, top_bound + 1, game_state.grid_length):
            if (x, y) in [game_state.P, game_state.T, game_state.C]:
                continue
            P = 0
            T = 1
            C = 1
            potential_values[x][y] = U(x, y, P, T, C)

    # clamp value to postive
    # potential_values = np.clip(potential_values, 0, None)

    potential_values = np.transpose(potential_values)

    plt.imshow(potential_values, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Potential values')
    plt.show()

    #####################
    print()
    # policy extraction
    actions = []
    curr_state = (0, 0, get_animal_encoding(False, False, True))
    count = 0

    while curr_state[2] < 6 and count < 100:
        x, y, ae = curr_state
        P, T, C = get_animal_decoding(ae)
        P = 1 if P else 0
        T = 1 if T else 0
        C = 1 if C else 0
        pactions = pre_actions[x, y]

        best_action = None
        best_reward = U(x, y, P, T, C)

        print (x, y, ae)
        print (best_reward)

        for a in pactions:
            _, new_state = reward(x, y, ae, a, game_state)
            new_x, new_y, new_ae = new_state
            new_P, new_T, new_C = get_animal_decoding(new_ae)
            new_P = 1 if new_P else 0
            new_T = 1 if new_T else 0
            new_C = 1 if new_C else 0

            if (new_x, new_y) in [game_state.P, game_state.T, game_state.C]:
                continue

            new_U = U(new_x, new_y, new_P, new_T, new_C)
            print (a, new_U)
            if new_U > best_reward:
                print ("better!")
                best_reward = new_U
                best_action = a

        dist_to_plat = math.sqrt((x - game_state.P[0])**2 + (y - game_state.P[1])**2)
        dist_to_turt = math.sqrt((x - game_state.T[0])**2 + (y - game_state.T[1])**2)
        if best_action is None and dist_to_plat < 10:
            best_action = (Action.Encircle, game_state.P, Direction.CW)
        elif best_action is None and dist_to_turt < 10:
            best_action = (Action.Encircle, game_state.T, Direction.CCW)
        elif best_action is None:
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

    print ()

    print (actions)
    visualise_game_state(game_state, actions)

if __name__ == "__main__":
    main()
