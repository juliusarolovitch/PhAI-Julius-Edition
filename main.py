import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from a_star import a_star
import numpy as np


class PhAI:
    def __init__(self, size=(20, 20, 20), obstacle_size=(5, 5, 5), num_obstacles=5):
        self.size = size
        self.obstacle_size = obstacle_size
        self.num_obstacles = num_obstacles

    def generate_map(self):
        occupancy_map = torch.zeros(self.size)

        max_x, max_y, max_z = self.size
        o_x, o_y, o_z = self.obstacle_size

        for i in range(self.num_obstacles):
            x = random.randint(0, max_x - o_x)
            y = random.randint(0, max_y - o_y)
            z = random.randint(0, max_z - o_z)
            occupancy_map[x:x + o_x, y:y + o_y, z:z + o_z] = 1

        return occupancy_map

    def generate_and_save_maps(self, num_maps):
        for i in tqdm(range(num_maps), desc="Generating maps"):
            map_tensor = self.generate_map()
            torch.save(map_tensor, f'maps/map_{i}.pt')

    def visualize_map(self, file_name):
        map_tensor = torch.load(file_name)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(map_tensor.numpy(), edgecolor='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def find_unoccupied_coordinate(self, occupancy_map):
        while True:
            x = random.randint(0, self.size[0] - 1)
            y = random.randint(0, self.size[1] - 1)
            z = random.randint(0, self.size[2] - 1)
            if occupancy_map[x, y, z] == 0:
                return (x, y, z)

    def generate_start_and_goal(self, occupancy_map):
        start = self.find_unoccupied_coordinate(occupancy_map)
        goal = self.find_unoccupied_coordinate(occupancy_map)
        while goal == start:  # Ensure start and goal are not the same
            goal = self.find_unoccupied_coordinate(occupancy_map)
        return start, goal

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def generate_paths(self, num_paths, map_index):
        map_file = f'maps/map_{map_index}.pt'
        occupancy_map = torch.load(map_file).numpy()  # Convert tensor to numpy

        for i in tqdm(range(num_paths), desc=f"Generating paths for map {map_index}"):
            start, goal = self.generate_start_and_goal(occupancy_map)
            _, forward_g, _ = a_star(occupancy_map, start, goal)
            _, backward_g, _ = a_star(occupancy_map, goal, start)
            f_star_map = self.calculate_f_star_values(forward_g, backward_g)

            path_data = {
                'start': start,
                'goal': goal,
                # Convert back to tensor
                'f_star_map': torch.tensor(f_star_map),
                # Save the forward_g map for g values
                'forward_g': torch.tensor(forward_g)
            }
            torch.save(path_data, f'paths/path_{i}_map_{map_index}.pt')

    def calculate_f_star_values(self, forward_g, backward_g):
        f_star_map = np.full(self.size, float('inf'))

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    if forward_g[x, y, z] < float('inf') and backward_g[x, y, z] < float('inf'):
                        f_star_map[x, y, z] = forward_g[x, y, z] + \
                            backward_g[x, y, z]

        return f_star_map

    def visualize_f_star_map(self, f_star_map, occupancy_map):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        f_star_np = f_star_map.numpy()
        occupancy_np = occupancy_map.numpy()

        mask = f_star_np < float('inf')

        x, y, z = mask.nonzero()

        f_star_values = f_star_np[mask]
        norm = plt.Normalize(vmin=f_star_values.min(),
                             vmax=f_star_values.max())
        colors = plt.cm.viridis(norm(f_star_values))

        ax.scatter(x, y, z, c=colors, marker='o', label='f* values')

        obstacle_mask = occupancy_np == 1
        ox, oy, oz = obstacle_mask.nonzero()
        ax.scatter(ox, oy, oz, c='red', marker='s', label='Obstacles')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()

    def prepare_dataset(self, num_maps, num_paths):
        for map_index in range(num_maps):
            occupancy_map = torch.load(f'maps/map_{map_index}.pt')
            dataset = []

            for path_index in tqdm(range(num_paths), desc=f"Building dataset for map {map_index}"):
                path = torch.load(
                    f'paths/path_{path_index}_map_{map_index}.pt')
                start = path['start']
                goal = path['goal']
                f_star_map = path['f_star_map'].numpy()
                forward_g_map = path['forward_g'].numpy()

                mask = f_star_map < float('inf')
                x, y, z = mask.nonzero()

                for i in range(len(x)):
                    coord = (x[i], y[i], z[i])
                    f_star_value = f_star_map[coord]
                    g_value = forward_g_map[coord]
                    h_value = self.heuristic(coord, goal)

                    start_tensor = torch.zeros(self.size)
                    goal_tensor = torch.zeros(self.size)
                    coord_tensor = torch.zeros(self.size)

                    start_tensor[start] = 1
                    goal_tensor[goal] = 1
                    coord_tensor[coord] = 1

                    input_tensor = torch.cat((occupancy_map.unsqueeze(0),
                                              start_tensor.unsqueeze(0),
                                              goal_tensor.unsqueeze(0),
                                              coord_tensor.unsqueeze(0)), dim=0)

                    dataset.append(
                        (input_tensor, g_value, h_value, f_star_value))

            torch.save(dataset, f'datasets/dataset_map_{map_index}.pt')
            print(f"Dataset size for map {map_index}: {len(dataset)}")


num_maps = 150
generator = PhAI()
num_paths = 200
generator.generate_and_save_maps(num_maps)

for j in tqdm(range(0, num_maps), desc="Generating paths"):
    generator.generate_paths(num_paths, j)

generator.prepare_dataset(num_maps, num_paths)
