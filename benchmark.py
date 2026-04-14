import time
import numpy as np
import matplotlib.pyplot as plt
from barnes_hut_final import Body, BoundingBox, QuadTree


def generate_random_stars(num_stars):
    bodies = []
    for i in range(num_stars):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        bodies.append(Body(f"S{i}", x, y, mass=10))
    return bodies

def run_benchmark():
    n_values = [50, 100, 250, 500, 1000, 2000]
    bh_times = []
    bf_times = []

    print("Starting Performance Benchmark...")
    print(f"{'Stars (N)':<10} | {'Barnes-Hut (s)':<15} | {'Brute Force (s)'}")
    print("-" * 45)

    for n in n_values:
        stars = generate_random_stars(n)
        universe = BoundingBox(0, 0, 200)

        # 1. Build the Tree
        tree = QuadTree(universe)
        for star in stars:
            tree.insert(star)

        # 2. Benchmark Barnes-Hut (Theta = 0.5)
        start_bh = time.perf_counter()
        for star in stars:
            star.reset_force()
            tree.calculate_force(star, theta=0.5)
        bh_time = time.perf_counter() - start_bh
        bh_times.append(bh_time)

        # 3. Benchmark Brute Force (Theta = 0.0)
        start_bf = time.perf_counter()
        for star in stars:
            star.reset_force()
            tree.calculate_force(star, theta=0.0)
        bf_time = time.perf_counter() - start_bf
        bf_times.append(bf_time)

        print(f"{n:<10} | {bh_time:<15.4f} | {bf_time:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, bf_times, marker='o', color='#e74c3c', label='Direct Summation O(N²)')
    plt.plot(n_values, bh_times, marker='s', color='#2ecc71', label='Barnes-Hut O(N log N)')

    plt.title('Algorithmic Complexity: Barnes-Hut vs Direct Summation', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Bodies (N)', fontsize=12)
    plt.ylabel('Computation Time per Frame (seconds)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('complexity_graph.png')  # Saves the image for your README
    plt.show()

if __name__ == "__main__":
    run_benchmark()