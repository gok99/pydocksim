import math
from enum import Enum
from dataclasses import dataclass
from visualiser import visualise_game_state
from visualiser import Action
import random

# actions: 
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

@dataclass
class State:
    pos: tuple       # (x, y) position of USV
    P: bool          # True if encircled
    T: bool          # True if encircled
    C: bool          # True if > 10m away so far
    time: float      # time elapsed so far

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

# rewards:
# -1 for each time step
# +30 for encircling platypus (if not collided with animal)
# +30 for encircling turtle   (if not collided with animal)
# +30 for staying > 10m away from crocodile (at the end of the episode)
def pseudo_vrx_score(x, y, ae, time, game_state: GameState):
    reward = 0
    reward -= time
    print (time)

    P, T, C = get_animal_decoding(ae)
    
    if P:
        reward += 30
        print ("bonus P")
    if T:
        reward += 30
        print ("bonus T")
    
    time_exceeded = time >= game_state.time_limit
    game_over = (P and T) or time_exceeded
    
    if game_over and C:
        reward += 30
        print ("bonus C")
    
    return reward

def score_solution(actions, game_state: GameState):
    total_time = 0
    P, T, C = False, False, True
    x, y = game_state.start_pos
    for a in actions:
        if a == Action.F:
            y += game_state.grid_length
            total_time += game_state.move_time
        elif a == Action.B:
            y -= game_state.grid_length
            total_time += game_state.move_time
        elif a == Action.R:
            x += game_state.grid_length
            total_time += game_state.move_time
        elif a == Action.L:
            x -= game_state.grid_length
            total_time += game_state.move_time
        elif a == Action.Encircle:
            continue
        else:
            # encircle
            cx, cy = a
            twopir = 2 * math.pi * math.sqrt((x - cx)**2 + (y - cy)**2)
            total_time += twopir * game_state.circle_time
            if (cx, cy) == game_state.P:
                P = True
            elif (cx, cy) == game_state.T:
                T = True

        dist_to_croc = math.sqrt((x - game_state.C[0])**2 + (y - game_state.C[1])**2)
        if dist_to_croc < 10:
            C = False
    
    return pseudo_vrx_score(x, y, get_animal_encoding(P, T, C), total_time, game_state)

# coord convention:
#  y
#  |
#  ---> x

def will_collide(usv_state: State, game_state: GameState, action: Action, pc: tuple = None, dir: Direction = None):
    action = determine_action(usv_state.pos, game_state.P, game_state.T, game_state.C, usv_state.P, usv_state.T)
    pos = usv_state.pos
    if action == Action.F:
        pos = (pos[0], pos[1] + game_state.grid_length)
    elif action == Action.B:
        pos = (pos[0], pos[1] - game_state.grid_length)
    elif action == Action.R:
        pos = (pos[0] + game_state.grid_length, pos[1])
    elif action == Action.L:
        pos = (pos[0] - game_state.grid_length, pos[1])
    croc_pos = game_state.C
    print(pos, croc_pos)
    if abs(pos[0]-croc_pos[0]) + abs(pos[1]-croc_pos[1]) <= 10:
        return True

def position_encircled(usv_pos: tuple, circle_pos: tuple, test_pos: tuple):
    radius = math.sqrt((usv_pos[0] - circle_pos[0])**2 + (usv_pos[1] - circle_pos[1])**2)
    return math.sqrt((test_pos[0] - circle_pos[0])**2 + (test_pos[1] - circle_pos[1])**2) <= radius

def run_action(usv_state: State, game_state: GameState, action: Action, pc: tuple = None, dir: Direction = None):
    new_usv_state = State(
        pos = usv_state.pos,
        P = usv_state.P,
        T = usv_state.T,
        C = usv_state.C,
        time = usv_state.time
    )
    
    if will_collide(usv_state, game_state, action):
        return None, usv_state

    if action == Action.Encircle:
        assert pc is not None and dir is not None, "Must specify pc and dir for Encircle action"

        if dir == Direction.CW:
            plat_circled = position_encircled(usv_state.pos, pc, game_state.P)
            new_usv_state.P = new_usv_state.P or plat_circled

        elif dir == Direction.CCW:
            turtle_circled = position_encircled(usv_state.pos, pc, game_state.T)
            new_usv_state.T = new_usv_state.T or turtle_circled

        twopiR = 2 * math.pi * math.sqrt((usv_state.pos[0] - pc[0])**2 + (usv_state.pos[1] - pc[1])**2)
        new_usv_state.time += twopiR * game_state.circle_time
    
    # update position
    else:
        new_usv_state.time += 1
        if action == Action.F:
            new_usv_state.pos = (usv_state.pos[0], usv_state.pos[1] + game_state.grid_length)
        elif action == Action.B:
            new_usv_state.pos = (usv_state.pos[0], usv_state.pos[1] - game_state.grid_length)
        elif action == Action.R:
            new_usv_state.pos = (usv_state.pos[0] + game_state.grid_length, usv_state.pos[1])
        elif action == Action.L:
            new_usv_state.pos = (usv_state.pos[0] - game_state.grid_length, usv_state.pos[1])

    return action, new_usv_state

def determine_action(usv_pos, platypus_pos, turtle_pos, crocodile_pos, platypus_encircled, turtle_encircled):

    if random.random() < 0.2:
            return random.choice([Action.F, Action.L, Action.B, Action.R]), None, None
    
    def calculate_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    usv_to_platypus_distance = calculate_distance(usv_pos, platypus_pos)
    usv_to_turtle_distance = calculate_distance(usv_pos, turtle_pos)
    usv_to_crocodile_distance = calculate_distance(usv_pos, crocodile_pos)

    if usv_to_crocodile_distance <= 10:
        # Move away from the crocodile
        if crocodile_pos[1] > usv_pos[1]:
            return Action.B, None, None
        elif crocodile_pos[1] < usv_pos[1]:
            return Action.F, None, None
        elif crocodile_pos[0] > usv_pos[0]:
            return Action.L, None, None
        else:
            return Action.R, None, None

    if not platypus_encircled:
        if usv_to_platypus_distance < 3:
            return Action.Encircle, platypus_pos, Direction.CW
    if not turtle_encircled:
        if usv_to_turtle_distance < 3:
            return Action.Encircle, turtle_pos, Direction.CCW
        
    if not platypus_encircled and not turtle_encircled:   
        if usv_to_platypus_distance < usv_to_turtle_distance:
            if platypus_pos[1] > usv_pos[1]:
                return Action.F, None, None
            elif platypus_pos[1] < usv_pos[1]:
                return Action.B, None, None
            elif platypus_pos[0] > usv_pos[0]:
                return Action.R, None, None
            else:
                return Action.L, None, None
        else:
            if turtle_pos[1] > usv_pos[1]:
                return Action.F, None, None
            elif turtle_pos[1] < usv_pos[1]:
                return Action.B, None, None
            elif turtle_pos[0] > usv_pos[0]:
                return Action.R, None, None
            else:
                return Action.L, None, None
    elif not platypus_encircled:
        if platypus_pos[1] > usv_pos[1]:
            return Action.F, None, None
        elif platypus_pos[1] < usv_pos[1]:
            return Action.B, None, None
        elif platypus_pos[0] > usv_pos[0]:
            return Action.R, None, None
        else:
            return Action.L, None, None
    elif not turtle_encircled:
        if turtle_pos[1] > usv_pos[1]:
            return Action.F, None, None
        elif turtle_pos[1] < usv_pos[1]:
            return Action.B, None, None
        elif turtle_pos[0] > usv_pos[0]:
            return Action.R, None, None
        else:
            return Action.L, None, None
    else:
        # If both animals are encircled, move forward
        return None, None, None

def main():
    game_state = GameState(
        start_pos = (0, 0),
        usv_dims = (1, 1),
        P = (45, 45),
        T = (10, 10),
        C = (30, 30),
        grid_length = 1,
        time_limit = 200,
        move_time = 1,
        circle_time = 1
    )
    state = State(
        pos=game_state.start_pos,
        P=False,
        T=False,
        C=True,
        time=0
    )
    action_history = []
    curr_state = state
    
    while not (state.P and state.T):
        print(curr_state)
        
        if curr_state.time > game_state.time_limit:
            break
        action, pc, dir = determine_action(curr_state.pos, game_state.P, game_state.T, game_state.C, state.P, state.T)
        if action == None:
            break
        take_action, curr_state = run_action(curr_state, game_state, action, pc, dir)
        if take_action == None:
            curr_state.time +=1
            continue
        action_history.append(action)
        print(action)
        if action == Action.Encircle:
            if dir == Direction.CW:
                state.P = True
                action_history.append(game_state.P)

            elif dir == Direction.CCW:
                state.T = True
                action_history.append(game_state.T)

    print(f"Total time taken: {curr_state.time}")
    print(score_solution(action_history, game_state))
    visualise_game_state(game_state, action_history)
    

if __name__ == "__main__":
    main()
