import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ----- PARAMETRI SIMULAZIONE -----
N_ROBOTS = 5
dt = 0.1
total_time = 300
num_steps = int(total_time / dt)

# Controlli
k1 = 0.02      # forza verso il bordo
k2 = 0.3       # moto tangenziale
k_cons = 0.2   # consensus angolare
k_rep = 0.1    # repulsione
d_min = 0.5    # distanza minima tra robot

# Circonferenza
R = 2

# ----- POSIZIONI INIZIALI -----
pos = (np.random.rand(N_ROBOTS, 2) - 0.5) * 4 * R

# ----- STORICO -----
history = [pos.copy()]

# ----- CENTRO DINAMICO -----
def moving_center(t):
    # Moto lineare + sinusoidale
    xc = 0.5 * np.sin(0.2 * t)
    yc = 0.5 * np.cos(0.1 * t)
    return xc, yc

# ----- FUNZIONE VELOCITA -----
def get_velocity(p, t):
    xc, yc = moving_center(t)
    diff = p - np.array([xc, yc])
    r = np.linalg.norm(diff, axis=1)

    # ----- Controllo bordo (normale) -----
    v_border = -k1 * (r - R)[:, np.newaxis] * (diff / (r[:, np.newaxis]+1e-6))

    # ----- Tangente unitario -----
    t_unit = np.column_stack([-diff[:,1], diff[:,0]])
    t_unit /= np.linalg.norm(t_unit, axis=1, keepdims=True)+1e-6

    # ----- Consensus angolare per distribuzione uniforme -----
    angles = np.arctan2(diff[:,1], diff[:,0])
    order = np.argsort(angles)
    v_tang = np.zeros_like(p)

    for i in range(N_ROBOTS):
        idx = order[i]
        next_idx = order[(i+1)%N_ROBOTS]
        prev_idx = order[(i-1)%N_ROBOTS]

        delta_next = angles[next_idx] - angles[idx]
        delta_next = (delta_next + np.pi) % (2*np.pi) - np.pi
        delta_prev = angles[idx] - angles[prev_idx]
        delta_prev = (delta_prev + np.pi) % (2*np.pi) - np.pi

        # errore rispetto alla distanza angolare desiderata
        error = (delta_next - 2*np.pi/N_ROBOTS) - (delta_prev - 2*np.pi/N_ROBOTS)
        v_tang[idx] = k_cons * error * t_unit[idx]

    # ----- Repulsione tangenziale per sicurezza -----
    v_rep = np.zeros_like(p)
    for i in range(N_ROBOTS):
        for j in range(N_ROBOTS):
            if i != j:
                diff_ij = p[i] - p[j]
                dist = np.linalg.norm(diff_ij)
                if dist < d_min:
                    v_rep[i] += k_rep * (diff_ij / (dist + 1e-6))

    # ----- Velocità totale -----
    v_total = v_border + v_tang + v_rep
    return v_total

# ----- SIMULAZIONE -----
for step in range(num_steps):
    t_curr = step * dt
    v = get_velocity(pos, t_curr)
    pos = pos + v * dt
    history.append(pos.copy())

history = np.array(history)

# ----- ANIMAZIONE -----
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title("Robots Following Moving Circle Uniformly")
ax.set_xlabel("X")
ax.set_ylabel("Y")

lines = [ax.plot([], [], '-', lw=1, alpha=0.6)[0] for _ in range(N_ROBOTS)]
robot_dots, = ax.plot([], [], 'go', markersize=8)
circle_plot, = ax.plot([], [], 'r--', alpha=0.5, label='Moving Circle')

circle_angles = np.linspace(0, 2*np.pi, 200)

def init():
    for line in lines: line.set_data([],[])
    robot_dots.set_data([],[])
    circle_plot.set_data([],[])
    return lines + [robot_dots, circle_plot]

def update(frame):
    current_history = history[:frame+1]
    for i in range(N_ROBOTS):
        lines[i].set_data(current_history[:, i, 0], current_history[:, i, 1])
    robot_dots.set_data(history[frame,:,0], history[frame,:,1])

    t_curr = frame * dt
    xc, yc = moving_center(t_curr)
    circle_plot.set_data(xc + R*np.cos(circle_angles), yc + R*np.sin(circle_angles))
    return lines + [robot_dots, circle_plot]

ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=dt*1000)
ax.legend(loc='upper right')
plt.show()

# ----- RISULTATO STATICO FINALE -----
plt.figure(figsize=(8,8))
# Cerchio finale
xc, yc = moving_center(total_time)
plt.plot(xc + R*np.cos(circle_angles), yc + R*np.sin(circle_angles), 'r--', alpha=0.5, label='Final Circle')

# Traiettorie robot
for i in range(N_ROBOTS):
    plt.plot(history[:,i,0], history[:,i,1], '-', alpha=0.6)
    plt.plot(history[0,i,0], history[0,i,1], 'ro', markersize=4)  # inizio
    plt.plot(history[-1,i,0], history[-1,i,1], 'go', markersize=6) # fine

# Legenda personalizzata
legend_elements = [
    Line2D([0],[0], color='r', linestyle='--', label='Final Circle'),
    Line2D([0],[0], marker='o', color='w', label='Start', markerfacecolor='r', markersize=8),
    Line2D([0],[0], marker='o', color='w', label='End', markerfacecolor='g', markersize=10),
]
plt.legend(handles=legend_elements)

plt.axis('equal')
plt.grid(True)
plt.title("Static Result of Robot Paths")
plt.savefig('simulation_result_final.png')
plt.show()