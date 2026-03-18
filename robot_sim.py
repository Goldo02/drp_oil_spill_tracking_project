import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Circle parameters
xc, yc = 0, 0
R = 2
R2 = R**2

# Control gains
k1 = 0.02
k2 = 0.5

# Simulation parameters
N_ROBOTS = 5
dt = 0.1
total_time = 15
num_steps = int(total_time / dt)

# Initial positions (random)
# Random points in [-2R, 2R]
pos = (np.random.rand(N_ROBOTS, 2) - 0.5) * 4 * R

# History for plotting trajectory: shape (steps, N_ROBOTS, 2)
history = [pos.copy()]

def get_velocity(p):
    px, py = p[:, 0], p[:, 1]
    # e = (x-xc)^2 + (y-yc)^2 - R^2
    e = (px - xc)**2 + (py - yc)**2 - R2
    
    # n = gradient of (x-xc)^2 + (y-yc)^2 - R^2 = [2(x-xc), 2(y-yc)]
    n = np.column_stack([2 * (px - xc), 2 * (py - yc)])
    
    # t = (-y, x) relative to center
    t = np.column_stack([-(py - yc), (px - xc)])
    
    # v = -k1*e*n + k2*t
    # e[:, np.newaxis] to multiply each row of n by the corresponding error
    v = -k1 * e[:, np.newaxis] * n + k2 * t
    return v

# Pre-calculate simulation
for _ in range(num_steps):
    v = get_velocity(pos)
    pos = pos + v * dt
    history.append(pos.copy())

history = np.array(history)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-2.5*R, 2.5*R)
ax.set_ylim(-2.5*R, 2.5*R)
ax.set_title(f"Simulation with {N_ROBOTS} Robots")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Plot the target circle
circle_angles = np.linspace(0, 2*np.pi, 200)
ax.plot(xc + R * np.cos(circle_angles), yc + R * np.sin(circle_angles), 'r--', label='Target Circle', alpha=0.5)

# Robot and Trajectory plots (using lists for multiple robots)
lines = [ax.plot([], [], '-', lw=1, alpha=0.6)[0] for _ in range(N_ROBOTS)]
robot_dots, = ax.plot([], [], 'go', markersize=8, label='Robots')

def init():
    for line in lines:
        line.set_data([], [])
    robot_dots.set_data([], [])
    return lines + [robot_dots]

def update(frame):
    current_history = history[:frame+1]
    for i in range(N_ROBOTS):
        lines[i].set_data(current_history[:, i, 0], current_history[:, i, 1])
    
    # Update robot dots (all at once)
    robot_dots.set_data(history[frame, :, 0], history[frame, :, 1])
    return lines + [robot_dots]

ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=dt*1000)
ax.legend(loc='upper right')

# Save the animation
try:
    print("Saving animation to robot_simulation.gif...")
    ani.save('robot_simulation.gif', writer='pillow', fps=int(1/dt))
    print("Animation saved.")
except Exception as e:
    print(f"Could not save animation: {e}")

# Static result plot
plt.figure(figsize=(8, 8))
plt.plot(xc + R * np.cos(circle_angles), yc + R * np.sin(circle_angles), 'r--', label='Target Circle')
for i in range(N_ROBOTS):
    plt.plot(history[:, i, 0], history[:, i, 1], '-', alpha=0.6)
    plt.plot(history[0, i, 0], history[0, i, 1], 'ro', markersize=4)
    plt.plot(history[-1, i, 0], history[-1, i, 1], 'go', markersize=6)

# Custom legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='r', linestyle='--', label='Target Circle'),
    Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='r', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='End', markerfacecolor='g', markersize=10),
]
plt.legend(handles=legend_elements)

plt.axis('equal')
plt.grid(True)
plt.title(f"Static Result with {N_ROBOTS} Robots")
plt.savefig('simulation_result.png')
print("Static plot saved to simulation_result.png")

plt.show()
