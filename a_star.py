import math
from enum import Enum
from dataclasses import dataclass
from visualiser import visualise_game_state
from visualiser import Action
from queue import PriorityQueue

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

    def __lt__(self, other):
        # Define how States are compared based on their f-score
        return self.time < other.time

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

    if not game_over:
        reward -= 100
    else:
        reward -= time
    
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

def pos_change(action: Action, game_state: GameState):
    if action == Action.F:
        return (0, game_state.grid_length)
    elif action == Action.B:
        return (0, -game_state.grid_length)
    elif action == Action.R:
        return (game_state.grid_length, 0)
    elif action == Action.L:
        return (-game_state.grid_length, 0)
    else:
        return (0, 0)

def run_action(usv_state: State, game_state: GameState, action: Action, pc: tuple = None, dir: Direction = None):
    new_usv_state = State(
        pos = usv_state.pos,
        P = usv_state.P,
        T = usv_state.T,
        C = usv_state.C,
        time = usv_state.time
    )

    if action == Action.Encircle:
        assert pc is not None and dir is not None, "Must specify pc and dir for Encircle action"

        if dir == Direction.CW:
            new_usv_state.P = True

        elif dir == Direction.CCW:
            new_usv_state.T = True

        twopiR = 2 * math.pi * math.sqrt((usv_state.pos[0] - pc[0])**2 + (usv_state.pos[1] - pc[1])**2)
        new_usv_state.time += twopiR * game_state.circle_time
    
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

    return new_usv_state

def a_star_search(game_state: GameState, start_state: State, target: tuple):

    def heuristic(pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def croc_obstacles (game_state: GameState):
    # 10m euclidean radius circle around crocodile
        obstacles = []
        for i in range(game_state.C[0]-10, game_state.C[0]+10):
            for j in range(game_state.C[1]-10, game_state.C[1]+10):
                if (i-game_state.C[0])**2 + (j-game_state.C[1])**2 <= 100:
                    obstacles.append((i,j))
        return obstacles 
    
    croc_obstacles = croc_obstacles(game_state)
    
    def neighbours(state: State):
        movements = [(Action.F, (state.pos[0], state.pos[1] + game_state.grid_length)),
                     (Action.B, (state.pos[0], state.pos[1] - game_state.grid_length)), 
                     (Action.R, (state.pos[0] + game_state.grid_length, state.pos[1])), 
                     (Action.L, (state.pos[0] - game_state.grid_length, state.pos[1]))]
        valid_movements = []
        for action, new_pos in movements:
            if new_pos in croc_obstacles:
                continue
            else:
                valid_movements.append((action, new_pos))
        return valid_movements
    
    def reconstruct_path(came_from, current_state, parent_act):
        total_path = [parent_act]
        while current_state.pos in came_from:
            current_state, parent_act = came_from[current_state.pos]
            total_path.append(parent_act)
        return total_path[::-1]
    
    open_set = PriorityQueue() ## (f_score, state, action_from_parent)
    open_set.put((0, start_state, None))
    came_from = {}

    g_score = {start_state.pos: 0}
    f_score = {start_state.pos: heuristic(start_state.pos, target)}

    while not open_set.empty():
        curr_f_score, current_state, parent_act = open_set.get()

        if current_state.pos == target:
            path = reconstruct_path(came_from, current_state, parent_act)
            return path

        for action, neighbor_pos in neighbours(current_state):
            neighbor_state = State(
                pos=neighbor_pos,
                P=current_state.P,
                T=current_state.T,
                C=current_state.C,
                time=current_state.time + 1
            )

            tentative_g_score = g_score[current_state.pos] + 1
            if neighbor_state.pos in g_score and tentative_g_score >= g_score[neighbor_state.pos]:
                continue

            came_from[neighbor_state.pos] = current_state, action
            g_score[neighbor_state.pos] = tentative_g_score
            f_score[neighbor_state.pos] = tentative_g_score + heuristic(neighbor_pos, target)

            open_set.put((f_score[neighbor_state.pos], neighbor_state, action))

    return None  # No path found

    

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
        # P = (39, 39),
        # T = (10, 10),
        # C = (30, 30),

        # case 4
        # P = (-7, 15),
        # T = (15, 15),
        # C = (0, 15), 

        # case 5
        P = (-7, 15),
        T = (7, 15),
        C = (0, 15),
        grid_length = 1,
        time_limit = 100,
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
    ## determine a grid point that is less than 3 meters away from platypus or turtle
    ## A* search a path to that point and then encircle platypus or turtle, whichever is closer
    ## Then repeat A* again for the other animal
    ## Treat the radius 10m of the crocodile as obstacles


    animals_to_encircle = [(game_state.P,"platypus"), (game_state.T,"turtle")]
    action_history = []
    time = 0

    while not (state.P and state.T):
        if len(animals_to_encircle) == 0:
            break
        possible_targets = []
        for animal in animals_to_encircle:
            possible_targets.append(((animal[0][0]+3, animal[0][1]), animal[1]))
            possible_targets.append(((animal[0][0]-3, animal[0][1]), animal[1]))
            possible_targets.append(((animal[0][0], animal[0][1]+3), animal[1]))
            possible_targets.append(((animal[0][0], animal[0][1]-3), animal[1]))
            print(possible_targets)

        best = possible_targets[0]
        
        for target in possible_targets:
            if abs(target[0][0]- game_state.start_pos[0]) + abs(target[0][1]- game_state.start_pos[1]) < abs(best[0][0]- game_state.start_pos[0]) + abs(best[0][1]- game_state.start_pos[1]):
                best = target

        path = a_star_search(game_state, state, best[0])

        if not path:
            print("No valid path found.")
            return None
        else:
            if best[1] == "platypus":
                animals_to_encircle.remove((game_state.P,"platypus"))
                state.P = False
            else:
                animals_to_encircle.remove((game_state.T,"turtle"))
                state.T = False
        
        for action in path:
            state = run_action(state, game_state, action)
            action_history.append(action)
            time += 1
            
        if best[1] == "platypus":
            action_history.append(Action.Encircle)
            action_history.append(game_state.P)
            time += 2 * math.pi * 3

        else:
            action_history.append(Action.Encircle)
            action_history.append(game_state.T)
            time += 2 * math.pi * 3

    print(f"Total time taken: {time}")
    print(score_solution(action_history, game_state))
    visualise_game_state(game_state, action_history)

if __name__ == "__main__":
    main()
