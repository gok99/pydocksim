import math
from enum import Enum
from dataclasses import dataclass
from visualiser import visualise_game_state
from visualiser import Action

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

def reward(usv_state: State, game_state: GameState):
    reward = 0
    reward -= usv_state.time
    
    if usv_state.P:
        reward += 30
    if usv_state.T:
        reward += 30
    
    time_exceeded = usv_state.time >= game_state.time_limit
    game_over = (usv_state.P and usv_state.T) or time_exceeded
    
    if game_over and usv_state.C:
            reward += 30
    
    return reward

# coord convention:
#  y
#  |
#  ---> x

def will_collide(usv_state: State, game_state: GameState, action: Action, pc: tuple = None):
    # 
    # radius = inner_radius = outer_radius = bottom_y = top_y = left_x = right_x = 0

    def encircle_movement_check():
        radius = math.sqrt((usv_state.pos[0] - pc[0])**2 + (usv_state.pos[1] - pc[1])**2)
        inner_radius = radius - game_state.usv_dims[1]/2
        outer_radius = radius + game_state.usv_dims[1]/2

        return lambda animal_pos : \
            inner_radius <= math.sqrt((animal_pos[0] - pc[0])**2 + (animal_pos[1] - pc[1])**2) <= outer_radius

    def pos_movement_check():  
        bottom_y = usv_state.pos[1] - game_state.usv_dims[1]/2
        top_y = usv_state.pos[1] + game_state.usv_dims[1]/2
        left_x = usv_state.pos[0] - game_state.usv_dims[0]/2
        right_x = usv_state.pos[0] + game_state.usv_dims[0]/2

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
    
    if will_collide(usv_state, game_state, action, pc):
        # TODO: figure out what to do with collisions
        return new_usv_state

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
        new_usv_state.time += game_state.move_time
        if action == Action.F:
            new_usv_state.pos = (usv_state.pos[0], usv_state.pos[1] + game_state.grid_length)
        elif action == Action.B:
            new_usv_state.pos = (usv_state.pos[0], usv_state.pos[1] - game_state.grid_length)
        elif action == Action.R:
            new_usv_state.pos = (usv_state.pos[0] + game_state.grid_length, usv_state.pos[1])
        elif action == Action.L:
            new_usv_state.pos = (usv_state.pos[0] - game_state.grid_length, usv_state.pos[1])

    # TODO: use whole path, not just end point
    new_usv_state.C = new_usv_state.C and \
        math.sqrt((new_usv_state.pos[0] - game_state.C[0])**2 + (new_usv_state.pos[1] - game_state.C[1])**2) > 10

    return new_usv_state

def main():
    game_state = GameState(
        start_pos = (0, 0),
        usv_dims = (1, 1),
        P = (10, 0),
        T = (0, 10),
        C = (20, 20),
        grid_length = 1,
        time_limit = 100,
        move_time = 1,
        circle_time = 1
    )
    state = State(
        pos = game_state.start_pos,
        P = False,
        T = False,
        C = True,
        time = 0
    )
    print(game_state)
    print(state)
    print(reward(state, game_state))
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)
    state = run_action(state, game_state, Action.R)

    print(state)
    print(reward(state, game_state))
    state = run_action(state, game_state, Action.Encircle, pc = game_state.P, dir = Direction.CW)
    print(state)
    print(reward(state, game_state))
    state = run_action(state, game_state, Action.L)
    state = run_action(state, game_state, Action.L)
    state = run_action(state, game_state, Action.L)
    state = run_action(state, game_state, Action.L)
    
    print(state)
    print(reward(state, game_state))

    # actions should be appended to a list according to planner
    dummy_actions = [Action.R, Action.R, Action.R, Action.R, Action.R, Action.R, Action.R, Action.Encircle, game_state.P, Action.F, Action.F, Action.F, Action.F, Action.F, Action.F, Action.Encircle, game_state.T]

    visualise_game_state(game_state, dummy_actions)

if __name__ == "__main__":
    main()
