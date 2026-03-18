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
dt = 0.1
total_time = 15
num_steps = int(total_time / dt)

# Initial position (random)
# Random point in [-2R, 2R]
x = (np.random.rand() - 0.5) * 4 * R
y = (np.random.rand() - 0.5) * 4 * R
pos = np.array([x, y])

# History for plotting trajectory
history = [pos.copy()]

def get_velocity(p):
    px, py = p[0], p[1]
    # e = (x-xc)^2 + (y-yc)^2 - R^2
    e = (px - xc)**2 + (py - yc)**2 - R2
    
    # n = gradient of (x-xc)^2 + (y-yc)^2 - R^2 = [2(x-xc), 2(y-yc)]
    n = np.array([2 * (px - xc), 2 * (py - yc)])
    
    # t = (-y, x) relative to center
    t = np.array([-(py - yc), (px - xc)])
    
    # v = -k1*e*n + k2*t
    v = -k1 * e * n + k2 * t
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
ax.set_title("Robot Circular Path Tracking Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Plot the target circle
circle_angles = np.linspace(0, 2*np.pi, 200)
ax.plot(xc + R * np.cos(circle_angles), yc + R * np.sin(circle_angles), 'r--', label='Target Circle', alpha=0.5)

# Robot and Trayectory plots
line, = ax.plot([], [], 'b-', lw=1, alpha=0.7, label='Trajectory')
robot_dot, = ax.plot([], [], 'go', markersize=10, label='Robot')

def init():
    line.set_data([], [])
    robot_dot.set_data([], [])
    return line, robot_dot

def update(frame):
    current_history = history[:frame+1]
    line.set_data(current_history[:, 0], current_history[:, 1])
    robot_dot.set_data([history[frame, 0]], [history[frame, 1]])
    return line, robot_dot

ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=dt*1000)
plt.legend()

# Save the animation as a video if possible, or just show it.
# Since we are in an agentic environment, showing might not be direct.
# I will save it as a gif/mp4 to provide to the user.
try:
    print("Saving animation to robot_simulation.gif...")
    ani.save('robot_simulation.gif', writer='pillow', fps=int(1/dt))
    print("Animation saved.")
except Exception as e:
    print(f"Could not save animation: {e}")

# Also plot the final result statically
plt.figure(figsize=(8, 8))
plt.plot(xc + R * np.cos(circle_angles), yc + R * np.sin(circle_angles), 'r--', label='Target Circle')
plt.plot(history[:, 0], history[:, 1], 'b-', label='Trajectory')
plt.plot(history[0, 0], history[0, 1], 'ro', label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'go', label='End')
plt.axis('equal')
plt.grid(True)
plt.title("Static Result of 15s Simulation")
plt.legend()
plt.savefig('simulation_result.png')
print("Static plot saved to simulation_result.png")

plt.show() # This might not work in the environment, but good to have.
