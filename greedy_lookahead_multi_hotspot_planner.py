import numpy as np
from typing import List, Tuple
import time
import random

"""
In Grid I read out the txt file as an np array. I do this for faster calculations.
Recharge function increases the cells' collected values over time.
Collect function deals with a cell when it gets visited by the drone
"""


class Grid:
    def __init__(self, path: str, increment: float = 0.07, clip_max: float = 5.0):
        self.original_grid = np.loadtxt(path, dtype=int)
        self.float_grid = self.original_grid.astype(float).copy()
        self.mutable_grid = self.original_grid.copy()
        self.visited = np.zeros_like(self.original_grid, dtype=bool)
        self.cooldown = np.full_like(self.original_grid, fill_value=-1, dtype=int)
        self.N = self.original_grid.shape[0]

        # parameters
        self.increment_factor = increment
        self.clip_max = clip_max

    def recharge(self):
        increment = self.increment_factor * self.original_grid
        increment = np.clip(increment, 0.0, self.clip_max)

        active = self.cooldown >= 0
        self.cooldown[active] += 1

        mask = self.cooldown > 2
        self.float_grid[mask] = np.minimum(
            self.float_grid[mask] + increment[mask],
            self.original_grid[mask].astype(float),
        )

        self.mutable_grid = np.floor(self.float_grid).astype(int)

    def collect(self, pos: Tuple[int, int]) -> int:
        value = int(np.floor(self.float_grid[pos]))
        self.float_grid[pos] = 0.0
        self.mutable_grid[pos] = 0
        self.visited[pos] = True
        self.cooldown[pos] = 0
        return value


"""
In Drone I initialise a drone. It's starting position is either specified or is randomly chosen on an edge of the grid.
In each Drone object, the score is tracked independently.
"""


class Drone:
    def __init__(self, start: Tuple[int, int]):
        self.position = start
        self.path: List[Tuple[int, int]] = [start]
        self.score = 0

    def move(self, new_pos: Tuple[int, int]):
        self.position = new_pos
        self.path.append(new_pos)

    def add_score(self, value: int):
        self.score += value


"""
In Planner, I plan the path for a drone to take.
I start off by looking for hotspots and ranking them based on cell value, average neighbour value, chebyshev distance, and whether it has been visited.
I then use a greedy lookahead function that greedily navigates towards a hotspot so that the chebyhev distance is always reduced. 
This greedy lookahead function is applied to as many hotspots as possible and the path with the highest average reward per time step is kept.
"""


class Planner:
    def __init__(
        self,
        grid: Grid,
        drone: Drone,
        alpha: float = 0.5,
        beta: float = 0.5,
        lookahead_depth: int = 2,
    ):
        self.grid = grid
        self.drone = drone
        self.alpha = alpha
        self.beta = beta
        self.lookahead_depth = lookahead_depth
        self.active_path: List[Tuple[int, int]] = []

    def chebyshev(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def find_all_hotspots(
        self, steps_remaining: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Return all candidate hotspots with their scores."""
        N = self.grid.N
        x, y = self.drone.position
        M = min(5 + (N // 20), steps_remaining * 2)
        half_M = M // 2

        grid_max = np.max(self.grid.original_grid)

        x_min, x_max = max(0, x - half_M), min(N, x + half_M + 1)
        y_min, y_max = max(0, y - half_M), min(N, y + half_M + 1)

        scored = []

        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                value = self.grid.mutable_grid[i, j]
                if value <= 0:
                    continue

                distance = self.chebyshev((x, y), (i, j))
                if distance > steps_remaining:
                    continue

                neighbors = [
                    (i + dx, j + dy)
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    if not (dx == 0 and dy == 0) and 0 <= i + dx < N and 0 <= j + dy < N
                ]
                neighbor_values = [
                    self.grid.mutable_grid[nx, ny] for nx, ny in neighbors
                ]
                hotspot_score = value + (
                    np.mean(neighbor_values) if neighbor_values else 0
                )

                exploration_bonus = (
                    self.beta * grid_max if not self.grid.visited[i, j] else 0
                )

                total_score = (
                    hotspot_score / (1 + self.alpha * distance)
                ) + exploration_bonus
                scored.append(((i, j), total_score))

        return scored

    def greedy_with_lookahead(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Greedy baseline path planner with small lookahead depth for tie-breaking."""
        path = [start]
        current = start

        while current != goal:
            x, y = current
            best_val, best_move = -1, None
            current_dist = self.chebyshev(current, goal)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid.N and 0 <= ny < self.grid.N:
                        new_pos = (nx, ny)
                        if self.chebyshev(new_pos, goal) >= current_dist:
                            continue

                        # evaluate short greedy rollout for tie-breaking
                        score = self.simulate_rollout(
                            new_pos, goal, self.lookahead_depth
                        )
                        if score > best_val:
                            best_val, best_move = score, new_pos

            if not best_move:
                break
            current = best_move
            path.append(current)

        return path

    def simulate_rollout(
        self, start: Tuple[int, int], goal: Tuple[int, int], depth: int
    ) -> int:
        """Estimate cumulative reward by rolling out greedily for a few steps."""
        pos = start
        total = int(self.grid.mutable_grid[pos])
        for _ in range(depth - 1):
            x, y = pos
            neighbors = [
                (x + dx, y + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if not (dx == 0 and dy == 0)
                and 0 <= x + dx < self.grid.N
                and 0 <= y + dy < self.grid.N
            ]
            if not neighbors:
                break
            pos = max(neighbors, key=lambda p: self.grid.mutable_grid[p])
            total += int(self.grid.mutable_grid[pos])
            if pos == goal:
                break
        return total

    def evaluate_path(self, path: List[Tuple[int, int]]) -> float:
        total, steps = 0, 0
        sim_grid = self.grid.float_grid.copy()
        for pos in path:
            total += int(np.floor(sim_grid[pos]))
            steps += 1
        return total / steps if steps > 0 else 0

    def plan_step(self, time_limit_ms: int, steps_remaining: int):
        if not self.active_path:
            start_time = time.perf_counter()

            candidates = self.find_all_hotspots(steps_remaining)
            candidates.sort(key=lambda x: x[1], reverse=True)

            best_path, best_score = None, float("-inf")

            for hotspot, _ in candidates:
                if (time.perf_counter() - start_time) * 1000 >= time_limit_ms:
                    break

                path = self.greedy_with_lookahead(self.drone.position, hotspot)
                eval_score = self.evaluate_path(path)
                if eval_score > best_score:
                    best_score = eval_score
                    best_path = path

            if best_path:
                self.active_path = best_path

        if self.active_path:
            next_pos = self.active_path.pop(0)
            if next_pos == self.drone.position and self.active_path:
                next_pos = self.active_path.pop(0)
            self.drone.move(next_pos)
            reward = self.grid.collect(next_pos)
            self.drone.add_score(reward)
        else:
            # fallback move = best neighbor
            x, y = self.drone.position
            neighbors = [
                (x + dx, y + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if not (dx == 0 and dy == 0)
                and 0 <= x + dx < self.grid.N
                and 0 <= y + dy < self.grid.N
            ]
            if neighbors:
                values = [self.grid.mutable_grid[pos] for pos in neighbors]
                max_val = max(values)
                best_candidates = [
                    pos for pos, val in zip(neighbors, values) if val == max_val
                ]
                best_pos = random.choice(best_candidates)
                self.drone.move(best_pos)
                reward = self.grid.collect(best_pos)
                self.drone.add_score(reward)


class Simulation:
    def __init__(
        self,
        grid_path: str,
        num_drones: int = 1,
        T: int = 500,
        t: int = 0,
        start_positions: List[Tuple[int, int]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        increment: float = 0.07,
        clip_max: float = 5.0,
        lookahead_depth: int = 2,
    ):
        self.grid = Grid(grid_path, increment=increment, clip_max=clip_max)
        self.N = self.grid.N
        self.t = int((self.N * self.N) // 2) if t == 0 else t
        self.T = T
        self.steps_executed = 0

        if start_positions is None:
            self.drones: List[Drone] = [
                Drone(choose_edge_position(self.grid, i + 1)) for i in range(num_drones)
            ]
        else:
            if len(start_positions) != num_drones:
                raise ValueError("Number of start_positions must match num_drones")
            self.drones: List[Drone] = [Drone(pos) for pos in start_positions]

        self.planners: List[Planner] = [
            Planner(
                self.grid,
                drone,
                alpha=alpha,
                beta=beta,
                lookahead_depth=lookahead_depth,
            )
            for drone in self.drones
        ]

        for drone in self.drones:
            drone.add_score(self.grid.collect(drone.position))

    def run(self):
        for step in range(self.t):
            steps_remaining = self.t - step
            for planner in self.planners:
                planner.plan_step(time_limit_ms=self.T, steps_remaining=steps_remaining)
            self.grid.recharge()
            self.steps_executed = step + 1

    def results(self):
        return {
            "final_scores": [drone.score for drone in self.drones],
            "paths": [drone.path for drone in self.drones],
            "steps_executed": self.steps_executed,
        }


def choose_edge_position(grid: Grid, drone_id: int) -> Tuple[int, int]:
    rows, cols = grid.mutable_grid.shape
    choice = (
        input(
            f"Choose drone {drone_id} start edge (top/bottom/left/right or Enter=random): "
        )
        .strip()
        .lower()
    )
    if choice not in ["top", "bottom", "left", "right"]:
        choice = np.random.choice(["top", "bottom", "left", "right"])
        print(f"No valid choice, picking randomly for drone {drone_id}: {choice}")

    if choice == "top":
        return (0, np.random.randint(cols))
    elif choice == "bottom":
        return (rows - 1, np.random.randint(cols))
    elif choice == "left":
        return (np.random.randint(rows), 0)
    else:
        return (np.random.randint(rows), cols - 1)


if __name__ == "__main__":
    path_to_grid = "grids/100.txt"

    sim = Simulation(
        grid_path=path_to_grid,
        num_drones=1,
        T=10,  # ms budget per planning step
        t=200,  # total timesteps (moves) (set to 0 for t = N*N//2)
        start_positions=[
            (0, 0),
        ],  # set to None if we want to randomly set position on an edge. Set to [(x1, y1), (x2, y2)] if you want to specify its starting position
        alpha=0.5,  # weight factor for nearby hotspots
        beta=0.5,  # weight factor for non-visited hotspots
        increment=0.07,  # percentage of original value increment
        clip_max=5.0,  # maximum value increment
        lookahead_depth=3,  # lookahead depth for greedy planner
    )
    sim.run()
    results = sim.results()

    print(f"Grid size N = {sim.N}")
    print(f"Time steps t = {sim.t}")
    print(f"Time limit T = {sim.T} ms")
    print(f"Final scores: {results['final_scores']}")
    print(f"Path lengths: {[len(p) for p in results['paths']]}")
    print(f"Path = {results["paths"]}")
