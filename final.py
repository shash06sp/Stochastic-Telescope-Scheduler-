import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class StochasticObservatory:
    def __init__(self, time_horizon=24, num_targets=50):
        self.horizon = time_horizon
        self.num_targets = num_targets
        # Transition Matrix: Clear -> Clear (85%), Cloudy -> Cloudy (60%)
        self.transition_matrix = np.array([
            [0.85, 0.15],
            [0.40, 0.60]
        ])

    def generate_targets(self):
        np.random.seed(42)  # For reproducible results
        targets = []
        for i in range(self.num_targets):
            weight = np.random.randint(10, 100)
            duration = np.random.randint(1, 4)
            start_window = np.random.randint(0, self.horizon - duration)
            end_window = np.random.randint(start_window + duration, self.horizon + 1)

            targets.append({
                'target_id': i,
                'weight': weight,
                'duration': duration,
                'start_window': start_window,
                'end_window': end_window
            })
        return pd.DataFrame(targets)

    def simulate_weather_path(self, initial_state=1):
        weather_path = [initial_state]
        current_state = initial_state
        for _ in range(1, self.horizon):
            probabilities = self.transition_matrix[current_state]
            next_state = np.random.choice([1, 0], p=probabilities)
            weather_path.append(next_state)
            current_state = next_state
        return np.array(weather_path)

class GreedyScheduler:
    def __init__(self, targets_df, weather_path):
        self.targets = targets_df.copy()
        self.weather = weather_path
        self.horizon = len(weather_path)

    def run(self):
        schedule = []
        total_weight = 0
        t = 0
        pending_targets = self.targets.sort_values(by='weight', ascending=False)

        while t < self.horizon:
            if self.weather[t] == 0:
                t += 1
                continue

            valid_mask = (
                    (pending_targets['start_window'] <= t) &
                    (pending_targets['end_window'] >= t + pending_targets['duration']) &
                    (t + pending_targets['duration'] <= self.horizon)
            )
            valid_targets = pending_targets[valid_mask]

            if valid_targets.empty:
                t += 1
                continue

            best_target = valid_targets.iloc[0]
            dur = int(best_target['duration'])

            if np.all(self.weather[t: t + dur] == 1):
                total_weight += best_target['weight']
                status = 'Success'
            else:
                status = 'Failed'

            schedule.append({
                'id': best_target['target_id'], 'start': t, 'duration': dur, 'status': status
            })

            pending_targets = pending_targets[pending_targets['target_id'] != best_target['target_id']]
            t += dur

        return total_weight, schedule

class StochasticSAScheduler:
    def __init__(self, observatory, targets_df, actual_weather):
        self.obs = observatory
        self.targets = targets_df.copy()
        self.actual_weather = actual_weather
        self.horizon = observatory.horizon

    def dispatch(self, priority_list, weather_path, track_schedule=False):
        t = 0
        total_weight = 0
        schedule = []
        pending = self.targets.set_index('target_id').reindex(priority_list).dropna()

        while t < self.horizon and not pending.empty:
            if weather_path[t] == 0:
                t += 1
                continue

            valid_mask = (
                    (pending['start_window'] <= t) &
                    (pending['end_window'] >= t + pending['duration']) &
                    (t + pending['duration'] <= self.horizon)
            )
            valid_targets = pending[valid_mask]

            if valid_targets.empty:
                t += 1
                continue

            best_target = valid_targets.iloc[0]
            dur = int(best_target['duration'])
            target_id = best_target.name

            if np.all(weather_path[t: t + dur] == 1):
                total_weight += best_target['weight']
                status = 'Success'
            else:
                status = 'Failed'

            if track_schedule:
                schedule.append({
                    'id': target_id, 'start': t, 'duration': dur, 'status': status
                })

            pending = pending.drop(target_id)
            t += dur

        if track_schedule:
            return total_weight, schedule
        return total_weight

    def evaluate_expected_value(self, priority_list, num_simulations=10):
        # We don't need to track the schedule during the thousands of simulations
        scores = [self.dispatch(priority_list, self.obs.simulate_weather_path(), track_schedule=False)
                  for _ in range(num_simulations)]
        return np.mean(scores)

    def optimize(self, initial_temp=1000, cooling_rate=0.95, iterations=100):
        print("Optimizing Stochastic Schedule (Evaluating probabilistic weather futures)...")
        current_state = list(self.targets['target_id'].values)
        np.random.shuffle(current_state)
        current_ev = self.evaluate_expected_value(current_state)

        best_state = copy.deepcopy(current_state)
        best_ev = current_ev
        temp = initial_temp

        for i in range(iterations):
            neighbor_state = copy.deepcopy(current_state)
            idx1, idx2 = np.random.choice(len(neighbor_state), 2, replace=False)
            neighbor_state[idx1], neighbor_state[idx2] = neighbor_state[idx2], neighbor_state[idx1]

            neighbor_ev = self.evaluate_expected_value(neighbor_state)

            if neighbor_ev > current_ev:
                current_state = neighbor_state
                current_ev = neighbor_ev
                if neighbor_ev > best_ev:
                    best_state = copy.deepcopy(neighbor_state)
                    best_ev = neighbor_ev
            else:
                delta = neighbor_ev - current_ev
                if np.random.rand() < math.exp(delta / temp):
                    current_state = neighbor_state
                    current_ev = neighbor_ev

            temp *= cooling_rate
            if i % 25 == 0:
                print(f"  Iteration {i} | Temp: {temp:.2f} | Best Expected Value: {best_ev:.2f}")

        return best_state

    def run_actual(self, optimized_priority_list):
        # Turn tracking ON for the final run against ground-truth weather
        return self.dispatch(optimized_priority_list, self.actual_weather, track_schedule=True)

def plot_observatory_gantt(weather, greedy_schedule, ai_schedule, horizon=24):
    """Generates the flagship visual for the portfolio."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 1. Plot Stochastic Weather Background (Gray hatch = Clouds, White = Clear)
    for t in range(horizon):
        if weather[t] == 0:
            ax.axvspan(t, t + 1, facecolor='#d3d3d3', alpha=0.6, hatch='///', edgecolor='none')

    # 2. Function to draw the observation blocks
    def draw_blocks(schedule, y_pos):
        for item in schedule:
            start = item['start']
            dur = item['duration']
            # Green for successful science, Red for clouded-out failures
            color = '#2ecc71' if item['status'] == 'Success' else '#e74c3c'

            ax.barh(y_pos, dur, left=start, color=color, edgecolor='black', height=0.4, zorder=3)
            # Label the target ID inside the block
            ax.text(start + dur / 2, y_pos, f"T-{int(item['id'])}",
                    ha='center', va='center', color='white', fontweight='bold', zorder=4)

    draw_blocks(greedy_schedule, 1)
    draw_blocks(ai_schedule, 0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Stochastic AI\n(Simulated Annealing)', 'Baseline Heuristic\n(Greedy)'], fontsize=12)
    ax.set_xlim(0, horizon)
    ax.set_xticks(range(0, horizon + 1, 1))
    ax.set_xlabel("Operational Time Horizon (Hours)", fontsize=12, fontweight='bold')
    ax.set_title("Telescope Observation Dispatch: AI Optimization vs Stochastic Cloud Cover", fontsize=14,
                 fontweight='bold')
    green_patch = mpatches.Patch(color='#2ecc71', label='Science Captured (Success)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Telescope Time Wasted (Clouded Out)')
    gray_patch = mpatches.Patch(facecolor='#d3d3d3', hatch='///', alpha=0.6, label='Atmospheric Interference')
    ax.legend(handles=[green_patch, red_patch, gray_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Initialize the Environment
    print("--- 1. Generating Observatory Environment ---")
    obs = StochasticObservatory(time_horizon=24, num_targets=15)
    df_targets = obs.generate_targets()

    weather = obs.simulate_weather_path(initial_state=1)
    print(f"Actual Weather Array: \n{weather}")
    print(f"Total Clear Hours: {np.sum(weather)} / 24\n")

    # 2. Run the Greedy Baseline
    print("--- 2. Running Greedy Baseline Scheduler ---")
    greedy = GreedyScheduler(df_targets, weather)
    greedy_score, greedy_schedule = greedy.run()
    print(f"Greedy Score Achieved: {greedy_score}\n")

    # 3. Run the AI / Simulated Annealing Optimizer
    print("--- 3. Running Stochastic Simulated Annealing ---")
    sa_scheduler = StochasticSAScheduler(obs, df_targets, weather)
    best_queue = sa_scheduler.optimize(initial_temp=500, cooling_rate=0.90, iterations=150)

    sa_score, sa_schedule = sa_scheduler.run_actual(best_queue)

    print(f"\n=========================================")
    print(f"RESULTS: SCIENTIFIC VALUE ACHIEVED")
    print(f"=========================================")
    print(f"Greedy Baseline:        {greedy_score}")
    print(f"Stochastic AI Engine:   {sa_score}")
    print(f"=========================================")

    if greedy_score > 0:
        improvement = ((sa_score - greedy_score) / greedy_score) * 100
        print(f"Performance Delta:      {improvement:+.1f}%")

    print("\nRendering Gantt Chart visualization...")
    plot_observatory_gantt(weather, greedy_schedule, sa_schedule)