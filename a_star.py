import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define the Node class


class Node:
    def __init__(self, x, y, z, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost from current node to end
        self.f = g + h  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)


def a_star(grid, start, end):
    # Ensure the grid is 20x20x20
    if grid.shape != (20, 20, 20):
        raise ValueError("The grid must be 20x20x20 in size.")

    open_list = []
    closed_list = set()
    start_node = Node(start[0], start[1], start[2])
    end_node = Node(end[0], end[1], end[2])

    # Initialize all g* values to infinity
    g_values = np.full(grid.shape, np.inf)
    g_values[start_node.x, start_node.y, start_node.z] = 0

    heapq.heappush(open_list, start_node)
    expanded_states = []

    while open_list:
        current_node = heapq.heappop(open_list)
        expanded_states.append(
            (current_node.x, current_node.y, current_node.z))

        if (current_node.x, current_node.y, current_node.z) == (end_node.x, end_node.y, end_node.z):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y, current_node.z))
                current_node = current_node.parent
            return path[::-1], g_values, expanded_states

        closed_list.add((current_node.x, current_node.y, current_node.z))

        # Adjacent cubes
        for new_position in [(0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]:
            node_position = (
                current_node.x + new_position[0], current_node.y + new_position[1], current_node.z + new_position[2])

            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0]) - 1) or node_position[1] < 0 or node_position[2] > (len(grid[0][0]) - 1) or node_position[2] < 0:
                continue

            if grid[node_position[0]][node_position[1]][node_position[2]] != 0:
                continue

            if node_position in closed_list:
                continue

            g = current_node.g + 1
            h = heuristic(Node(*node_position), end_node)
            new_node = Node(
                node_position[0], node_position[1], node_position[2], g, h, current_node)

            if not any(node for node in open_list if node.x == new_node.x and node.y == new_node.y and node.z == new_node.z and node.g <= new_node.g):
                heapq.heappush(open_list, new_node)
                g_values[new_node.x, new_node.y, new_node.z] = new_node.g

    return None, g_values, expanded_states

# Generate a 20x20x20 grid with random obstacles


def generate_grid(size=20, num_obstacles=5, obstacle_size=5):
    grid = np.zeros((size, size, size), dtype=int)
    obstacles = []

    for _ in range(num_obstacles):
        x = random.randint(0, size - obstacle_size)
        y = random.randint(0, size - obstacle_size)
        z = random.randint(0, size - obstacle_size)
        grid[x:x+obstacle_size, y:y+obstacle_size, z:z+obstacle_size] = 1
        obstacles.append((x, y, z))

    return grid, obstacles

# Select random start and end points


def select_random_points(grid):
    while True:
        start = (random.randint(0, len(grid) - 1), random.randint(0,
                 len(grid) - 1), random.randint(0, len(grid) - 1))
        end = (random.randint(0, len(grid) - 1), random.randint(0,
               len(grid) - 1), random.randint(0, len(grid) - 1))
        if grid[start[0]][start[1]][start[2]] == 0 and grid[end[0]][end[1]][end[2]] == 0 and start != end:
            return start, end

# Visualize the grid, obstacles, and g* values


def visualize(grid, g_values, start, end):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Normalize g* values for colormap
    finite_g_values = g_values[np.isfinite(g_values)]
    if len(finite_g_values) > 0:
        norm = plt.Normalize(finite_g_values.min(), finite_g_values.max())
        colors = cm.viridis(norm(g_values))
    else:
        colors = np.full(g_values.shape + (4,), [0, 0, 0, 1])

    # Plot obstacles
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x, y, z] == 1:
                    ax.scatter(x, y, z, color='black')

    # Plot g* values
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if np.isfinite(g_values[x, y, z]):
                    ax.scatter(x, y, z, color=colors[x, y, z], s=10)

    # Plot start and end points
    ax.scatter(start[0], start[1], start[2],
               color='green', s=100, label='Start')
    ax.scatter(end[0], end[1], end[2], color='red', s=100, label='End')

    plt.legend()
    plt.show()
