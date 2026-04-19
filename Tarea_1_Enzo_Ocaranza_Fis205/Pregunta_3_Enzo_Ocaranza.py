import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


#MISIL Y OBJETIVO

g = 9.81              
m = 100.0             
rho = 1.225           
Cd = 0.3              
A = 0.05              
Omega = 7.2921e-5     
latitud = np.radians(45) 

Omega_vec = np.array([0, Omega * np.cos(latitud), Omega * np.sin(latitud)])
k_roce = 0.5 * rho * Cd * A / m

def misil_edo(t, Y):
    r = Y[0:3]
    v = Y[3:6]
    v_mag = np.linalg.norm(v)
    a_grav = np.array([0, 0, -g])
    a_roce = -k_roce * v_mag * v
    a_coriolis = -2 * np.cross(Omega_vec, v)
    a_total = a_grav + a_roce + a_coriolis
    return [v[0], v[1], v[2], a_total[0], a_total[1], a_total[2]]

def hit_ground(t, Y):
    return Y[2] 
hit_ground.terminal = True
hit_ground.direction = -1 

print("Simulando la trayectoria del misil objetivo")
r1_0 = np.array([0.0, 0.0, 0.0])
v1_mag = 500.0
theta1 = np.radians(45) 
psi1 = np.radians(30)   

v1_0 = np.array([
    v1_mag * np.cos(theta1) * np.cos(psi1),
    v1_mag * np.cos(theta1) * np.sin(psi1),
    v1_mag * np.sin(theta1)
])

Y1_0 = np.concatenate((r1_0, v1_0))
sol1 = solve_ivp(misil_edo, (0, 200), Y1_0, events=hit_ground, dense_output=True, rtol=1e-8, atol=1e-8)
print(f"El misil objetivo impactaría el suelo en t = {sol1.t[-1]:.2f} s si no es interceptado.")


#ALGORITMO DE INTERCEPCIÓN
tau = 10.0
r2_0 = np.array([5000.0, 2000.0, 0.0])

def distancia_minima(parametros):
    v2_mag, theta2, psi2 = parametros
    v2_0 = np.array([
        v2_mag * np.cos(theta2) * np.cos(psi2),
        v2_mag * np.cos(theta2) * np.sin(psi2),
        v2_mag * np.sin(theta2)
    ])
    
    Y2_0 = np.concatenate((r2_0, v2_0))
    #Simulamos el misil 2
    sol2 = solve_ivp(misil_edo, (tau, sol1.t[-1]), Y2_0, dense_output=True, rtol=1e-6, atol=1e-6)
    
    #Buscamos la distancia
    t_eval = np.linspace(tau, min(sol1.t[-1], sol2.t[-1]), 300)
    distancias = np.linalg.norm(sol1.sol(t_eval)[0:3] - sol2.sol(t_eval)[0:3], axis=0)
    return np.min(distancias)

t_guess = 35.0
r_aim = sol1.sol(t_guess)[0:3]
dr = r_aim - r2_0
v_guess = np.linalg.norm(dr) / (t_guess - tau) + 50
theta_guess = np.arctan2(dr[2], np.hypot(dr[0], dr[1])) + np.radians(20)
psi_guess = np.arctan2(dr[1], dr[0])
guess_inicial = [v_guess, theta_guess, psi_guess]

resultado_opt = minimize(
    distancia_minima, guess_inicial, method='Nelder-Mead', options={'disp': True, 'maxiter': 500}
)

v2_opt, theta2_opt, psi2_opt = resultado_opt.x
dist_min_lograda = resultado_opt.fun

print(f"\nResultados del Interceptor")
print(f"Distancia mínima lograda: {dist_min_lograda:.2f} m")
print(f"Velocidad inicial v2_0: {v2_opt:.2f} m/s")
print(f"Ángulo de elevación theta2: {np.degrees(theta2_opt):.2f}°")
print(f"Azimut psi2: {np.degrees(psi2_opt):.2f}°")

if dist_min_lograda <= 10.0:
    print("INTERCEPCIÓN EXITOSA")
else:
    print("No se logró bajar de los 10 m de tolerancia.")


#TIEMPO DE ENCUENTRO Y GRÁFICO

v2_0_opt = np.array([
    v2_opt * np.cos(theta2_opt) * np.cos(psi2_opt),
    v2_opt * np.cos(theta2_opt) * np.sin(psi2_opt),
    v2_opt * np.sin(theta2_opt)
])

sol2_opt = solve_ivp(misil_edo, (tau, sol1.t[-1]), np.concatenate((r2_0, v2_0_opt)), dense_output=True, rtol=1e-8, atol=1e-8)

t_fino = np.linspace(tau, min(sol1.t[-1], sol2_opt.t[-1]), 2000)
dist_fina = np.linalg.norm(sol1.sol(t_fino)[0:3] - sol2_opt.sol(t_fino)[0:3], axis=0)
t_colision = t_fino[np.argmin(dist_fina)]
punto_colision = sol1.sol(t_colision)[0:3]

print(f"\nTiempo de Encuentro")
print(f"La colisión ocurre en t = {t_colision:.2f} segundos.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

t_plot1 = np.linspace(0, t_colision, 200)
t_plot2 = np.linspace(tau, t_colision, 200)

ax.plot(*sol1.sol(t_plot1)[0:3], label='Misil Objetivo', color='red', linewidth=2)
ax.plot(*sol2_opt.sol(t_plot2)[0:3], label='Misil Interceptor', color='blue', linewidth=2, linestyle='--')

ax.scatter(*r1_0, color='darkred', marker='^', s=80, label='Lanzamiento 1 (t=0)')
ax.scatter(*r2_0, color='darkblue', marker='^', s=80, label='Lanzamiento 2 (t=10s)')
ax.scatter(*punto_colision, color='orange', marker='*', s=300, edgecolor='black', label=f'Colisión (t={t_colision:.1f}s)')

ax.set_xlabel('Este X (m)')
ax.set_ylabel('Norte Y (m)')
ax.set_zlabel('Altitud Z (m)')
ax.set_title('Trayectorias de Intercepción de Misiles Balísticos')
ax.legend()
plt.show()

from matplotlib.animation import FuncAnimation


#ANIMACIÓN 
print("\nGenerando animación 3D... (Cierra esta ventana al terminar)")

fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

x_max = max(np.max(sol1.y[0]), np.max(sol2_opt.y[0])) + 500
y_max = max(np.max(sol1.y[1]), np.max(sol2_opt.y[1])) + 500
z_max = max(np.max(sol1.y[2]), np.max(sol2_opt.y[2])) + 500

ax_anim.set_xlim([0, x_max])
ax_anim.set_ylim([0, y_max])
ax_anim.set_zlim([0, z_max])
ax_anim.set_xlabel('Este X (m)')
ax_anim.set_ylabel('Norte Y (m)')
ax_anim.set_zlabel('Altitud Z (m)')
ax_anim.set_title('Misión de Intercepción en Tiempo Real')

line1, = ax_anim.plot([], [], [], color='red', lw=2, label='Misil Objetivo')
line2, = ax_anim.plot([], [], [], color='blue', lw=2, linestyle='--', label='Interceptor')
pt1, = ax_anim.plot([], [], [], marker='o', color='darkred', markersize=6)
pt2, = ax_anim.plot([], [], [], marker='o', color='darkblue', markersize=6)
texto_tiempo = ax_anim.text2D(0.05, 0.95, "", transform=ax_anim.transAxes, fontsize=12)

ax_anim.legend()

frames_t = np.linspace(0, t_colision, 200)

def init():
    line1.set_data_3d([], [], [])
    line2.set_data_3d([], [], [])
    pt1.set_data_3d([], [], [])
    pt2.set_data_3d([], [], [])
    texto_tiempo.set_text("")
    return line1, line2, pt1, pt2, texto_tiempo

def update(frame):
    t_actual = frames_t[frame]
    
    #Actualizar Misil 1
    t_hist1 = np.linspace(0, t_actual, 50)
    pos_m1 = sol1.sol(t_hist1)[0:3]
    line1.set_data_3d(pos_m1[0], pos_m1[1], pos_m1[2])
    pt1.set_data_3d([pos_m1[0, -1]], [pos_m1[1, -1]], [pos_m1[2, -1]])
    
    #Actualizar Misil 2 
    if t_actual >= tau:
        t_hist2 = np.linspace(tau, t_actual, 50)
        pos_m2 = sol2_opt.sol(t_hist2)[0:3]
        line2.set_data_3d(pos_m2[0], pos_m2[1], pos_m2[2])
        pt2.set_data_3d([pos_m2[0, -1]], [pos_m2[1, -1]], [pos_m2[2, -1]])
    else:
        
        line2.set_data_3d([], [], [])
        pt2.set_data_3d([r2_0[0]], [r2_0[1]], [r2_0[0]*0]) # Z=0
        
    texto_tiempo.set_text(f"Tiempo: {t_actual:.1f} s")
    return line1, line2, pt1, pt2, texto_tiempo

anim = FuncAnimation(fig_anim, update, frames=len(frames_t), init_func=init, blit=False, interval=40)
plt.show()