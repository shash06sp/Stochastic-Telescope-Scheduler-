import math
import numpy as np


class Body:
    """Represents a star or galaxy in 2D space."""

    def __init__(self, body_id, x, y, mass, vx=0.0, vy=0.0):
        self.id = body_id
        self.x = x
        self.y = y
        self.mass = mass

        # Velocity components
        self.vx = vx
        self.vy = vy

        # Force components
        self.fx = 0.0
        self.fy = 0.0

    def reset_force(self):
        self.fx = 0.0
        self.fy = 0.0

    def update_physics(self, dt):
        """Symplectic Euler Integration to update velocity and position."""
        # 1. Calculate acceleration (a = F/m)
        ax = self.fx / self.mass
        ay = self.fy / self.mass

        # 2. Update velocity
        self.vx += ax * dt
        self.vy += ay * dt

        # 3. Update position using the NEW velocity
        self.x += self.vx * dt
        self.y += self.vy * dt


class BoundingBox:
    def __init__(self, x_center, y_center, half_dimension):
        self.x = x_center
        self.y = y_center
        self.half_dim = half_dimension

    def contains(self, body):
        return (self.x - self.half_dim <= body.x < self.x + self.half_dim and
                self.y - self.half_dim <= body.y < self.y + self.half_dim)


class QuadTree:
    def __init__(self, boundary):
        self.boundary = boundary
        self.body = None
        self.is_leaf = True

        self.total_mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0

        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None

    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        hd = self.boundary.half_dim / 2.0

        self.nw = QuadTree(BoundingBox(x - hd, y + hd, hd))
        self.ne = QuadTree(BoundingBox(x + hd, y + hd, hd))
        self.sw = QuadTree(BoundingBox(x - hd, y - hd, hd))
        self.se = QuadTree(BoundingBox(x + hd, y - hd, hd))
        self.is_leaf = False

    def insert(self, body):
        if not self.boundary.contains(body): return False

        new_total_mass = self.total_mass + body.mass
        self.com_x = (self.com_x * self.total_mass + body.x * body.mass) / new_total_mass
        self.com_y = (self.com_y * self.total_mass + body.y * body.mass) / new_total_mass
        self.total_mass = new_total_mass

        if self.is_leaf and self.body is None:
            self.body = body
            return True

        if self.is_leaf:
            self.subdivide()
            old_body = self.body
            self.body = None
            self._insert_into_children(old_body)

        return self._insert_into_children(body)

    def _insert_into_children(self, body):
        if self.nw.insert(body): return True
        if self.ne.insert(body): return True
        if self.sw.insert(body): return True
        if self.se.insert(body): return True
        return False

    def calculate_force(self, target_body, theta=0.5, G=1.0, softening=0.1):
        """
        Recursively traverses the tree to calculate the net force on target_body.
        softening: Prevents division by zero if stars get too close.
        """
        # 1. Base Case: If this quadrant is empty, it exerts no force.
        if self.total_mass == 0:
            return

        # Distance from target star to this quadrant's center of mass
        dx = self.com_x - target_body.x
        dy = self.com_y - target_body.y
        dist_sq = dx ** 2 + dy ** 2 + softening ** 2
        dist = math.sqrt(dist_sq)

        # 2. Base Case: If this is a leaf node, calculate exact force
        if self.is_leaf:
            if self.body is not None and self.body != target_body:
                force = (G * target_body.mass * self.body.mass) / dist_sq
                target_body.fx += force * (dx / dist)
                target_body.fy += force * (dy / dist)
            return

        # 3. The Barnes-Hut MAC Logic: s / d < theta
        s = self.boundary.half_dim * 2.0  # Width of the quadrant
        if (s / dist) < theta:
            # IT'S FAR ENOUGH! Treat this entire quadrant as one giant star.
            force = (G * target_body.mass * self.total_mass) / dist_sq
            target_body.fx += force * (dx / dist)
            target_body.fy += force * (dy / dist)
        else:
            # IT'S TOO CLOSE! Open the quadrant and calculate children recursively.
            self.nw.calculate_force(target_body, theta, G, softening)
            self.ne.calculate_force(target_body, theta, G, softening)
            self.sw.calculate_force(target_body, theta, G, softening)
            self.se.calculate_force(target_body, theta, G, softening)

if __name__ == "__main__":
    print("--- Day 3: Symplectic Numerical Integration ---")

    # 1. Initialize our bodies (giving them some initial velocity so they orbit instead of just crashing)
    bodies = [
        Body("S1", x=10, y=10, mass=100, vx=-0.5, vy=0.5),
        Body("S2", x=15, y=12, mass=50, vx=0.2, vy=-0.1),
        Body("S3", x=-20, y=-20, mass=200, vx=0.8, vy=0.8)
    ]

    # Simulation Parameters
    dt = 0.1  # Time step
    frames = 5  # Number of simulation steps to run
    theta = 0.5  # MAC threshold

    for frame in range(frames):
        # STEP A: Create the boundaries and build a FRESH QuadTree
        universe = BoundingBox(0, 0, 100)
        tree = QuadTree(universe)

        for body in bodies:
            tree.insert(body)

        # STEP B: Calculate Forces
        for body in bodies:
            body.reset_force()
            tree.calculate_force(body, theta=theta)

        # STEP C: Move the Bodies
        for body in bodies:
            body.update_physics(dt)

        # STEP D: Print the new positions for tracking
        print(f"Frame {frame + 1}:")
        for body in bodies:
            print(f"  {body.id} is now at X: {body.x:.2f}, Y: {body.y:.2f} (Speed: {math.hypot(body.vx, body.vy):.2f})")
        print("-" * 30)