import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from enum import Enum

class Action(Enum):
    F = 0
    B = 1
    R = 2
    L = 3
    Encircle = 4

def calculate_grid_size(game_state):
    # Calculate the maximum and minimum coordinates of the animals
    max_x = max(game_state.P[0], game_state.T[0], game_state.C[0])
    max_y = max(game_state.P[1], game_state.T[1], game_state.C[1])

    grid_size = max(max_x, max_y) + 20  # add some padding

    return grid_size

def get_radius(usv_x, usv_y, target_x, target_y):
    radius = ((usv_x - target_x)** 2 + (usv_y - target_y)** 2)** 0.5    
    return radius

def visualise_game_state(game_state, actions):
    # Get the padded grid size
    grid_size = calculate_grid_size(game_state)

    # Create a continuous grid map using numpy
    grid = np.zeros((grid_size, grid_size), dtype=float)

    # Place entities on the grid with the USV at (0, 0) with offset
    usv_x = game_state.start_pos[0] + 10
    usv_y = game_state.start_pos[1] + 10
    grid[usv_y][usv_x] = 1  # USV
    grid[game_state.P[1] + 10][game_state.P[0] + 10] = 2  # Platypus
    grid[game_state.T[1] + 10][game_state.T[0] + 10] = 3  # Turtle
    grid[game_state.C[1] + 10][game_state.C[0] + 10] = 4  # Crocodile

    # Define labels for the legend
    entity_labels = {
        1: 'USV initial position',
        2: 'Platypus',
        3: 'Turtle',
        4: 'Crocodile',
        5: 'Path taken',
        6: 'USV final position',
    }

    # Create a custom colormap
    cmap = mcolors.ListedColormap(['white', 'tab:cyan', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink', 'k', ])

    # Create legend patches
    legend_patches = [mpatches.Patch(color=cmap(i), label=entity_labels[i]) for i in entity_labels]

    # Adjust x and y axis ticks and labels relative to the USV position
    plt.xticks(range(usv_x - 10, usv_x + grid_size, 5), range(-10, grid_size, 5))
    plt.yticks(range(usv_y - 10, usv_y + grid_size, 5), range(-10, grid_size, 5))

    # create danger zone within 10m of crocodile
    danger = plt.Circle((game_state.C[0] + 10, game_state.C[1] + 10), 10, 
    fc='none', color=cmap(4))
    plt.gca().add_patch(danger)

    for i in range(len(actions)):
        if actions[i] == Action.F:
            # Move forward
            for _ in range(game_state.grid_length):
                usv_y += 1  # Move one step forward in the x-axis
                grid[usv_y][usv_x] = 5  # Draw a line in the forward direction
        elif actions[i] == Action.B:
            # Move backward
            for _ in range(game_state.grid_length):
                usv_y -= 1  # Move one step backward in the x-axis
                grid[usv_y][usv_x] = 5  # Draw a line in the backward direction
        elif actions[i] == Action.R:
            # Turn right (clockwise)
            usv_x += 1  # Move one step to the right in the y-axis
            grid[usv_y][usv_x] = 5  # Draw a line in the right direction
        elif actions[i] == Action.L:
            # Turn left (counterclockwise)
            usv_x -= 1  # Move one step to the left in the y-axis
            grid[usv_y][usv_x] = 5  # Draw a line in the left direction
        elif actions[i] == Action.Encircle:
            # Encircle the target based on the provided parameters
            entity = actions[i + 1]
            centre_x = entity[0] + 10
            centre_y = entity[1] + 10
            radius = get_radius(usv_x, usv_y, centre_x, centre_y)
            # Draw the circumference of the circle
            circle = plt.Circle((centre_x, centre_y), radius, fc='none', linewidth = 5.5, color=cmap(5))
            plt.gca().add_patch(circle)
        else:
            continue

    grid[usv_y][usv_x] = 6  # Final position of USV
    
    # Visualize the grid with the custom colormap
    plt.imshow(grid, cmap=cmap, origin='lower')
    plt.grid(which='both', color='black', linewidth=0.5)
    plt.legend(handles=legend_patches, loc=(1.04, 0.5))
    plt.title('Game State')
    plt.show()