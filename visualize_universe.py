import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from barnes_hut_final import Body, BoundingBox, QuadTree  # Assuming your previous file is barnes_hut.py


def generate_galaxy(num_stars, radius, center_x, center_y, mass_range=(10, 50)):
    """Generates a cluster of stars using a normal (Gaussian) distribution."""
    bodies = []
    for i in range(num_stars):
        # Gaussian distribution clusters stars heavily in the center, like a real galaxy
        x = np.random.normal(center_x, radius / 3)
        y = np.random.normal(center_y, radius / 3)
        mass = np.random.uniform(mass_range[0], mass_range[1])

        # Give them a slight tangential velocity so they swirl instead of just collapsing
        angle = np.arctan2(y - center_y, x - center_x)
        speed = np.random.uniform(0.5, 2.0)
        vx = -speed * np.sin(angle)
        vy = speed * np.cos(angle)

        bodies.append(Body(f"S{i}", x, y, mass, vx, vy))

    # Add a Supermassive Black Hole at the center to hold the galaxy together!
    bodies.append(Body("SMBH", center_x, center_y, mass=5000, vx=0, vy=0))
    return bodies



NUM_STARS = 200
UNIVERSE_SIZE = 200
THETA = 0.5
DT = 0.1

# 1. Spawn the Galaxy
print(f"Spawning galaxy with {NUM_STARS} stars...")
bodies = generate_galaxy(NUM_STARS, radius=80, center_x=0, center_y=0)

# 2. Setup the Matplotlib Canvas
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('Barnes-Hut N-Body Simulation')
ax.set_facecolor('black')  # Deep space background
ax.set_xlim(-UNIVERSE_SIZE, UNIVERSE_SIZE)
ax.set_ylim(-UNIVERSE_SIZE, UNIVERSE_SIZE)
ax.set_xticks([])
ax.set_yticks([])


scatter = ax.scatter([], [], s=[], c='white', alpha=0.8, edgecolors='none')


def animate(frame):
    """The game loop for Matplotlib."""
    # A. Build the QuadTree for the current frame
    universe = BoundingBox(0, 0, UNIVERSE_SIZE * 2)
    tree = QuadTree(universe)
    for body in bodies:
        tree.insert(body)


    for body in bodies:
        body.reset_force()
        tree.calculate_force(body, theta=THETA, G=1.0)

    # C. Update Physics
    x_data, y_data, sizes = [], [], []
    for body in bodies:
        body.update_physics(DT)
        x_data.append(body.x)
        y_data.append(body.y)
        # Make the Supermassive Black Hole visually distinct
        sizes.append(100 if body.id == "SMBH" else body.mass / 5)

        # D. Update the visualization
    scatter.set_offsets(np.c_[x_data, y_data])
    scatter.set_sizes(sizes)

    # Progress tracker
    if frame % 10 == 0:
        print(f"Rendered Frame {frame}...")

    return scatter,

print("Initializing gravity engine. Close the window to stop.")
# interval=20 runs the animation at roughly 50 FPS
ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

plt.show()

# NOTE: If you want to save this as a GIF for your GitHub README, uncomment below:
print("Saving to GIF... this may take a minute.")
ani.save('galaxy_simulation.gif', writer='pillow', fps=30)