import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# =====================================================================
# 1. PARÁMETROS CONFIGURABLES DEL SISTEMA (Modífalos a tu gusto)
# =====================================================================
params = {
    'm1': 3.0,       # Masa del carro [kg]
    'm2': 0.50,       # Masa del péndulo [kg]
    'l': 1.0,        # Longitud del péndulo [m]
    'k': 25.0,       # Constante del resorte [N/m]
    'l0': 1.0,       # Longitud natural del resorte [m]
    'g': 9.81,       # Aceleración de la gravedad (No cambiar para mas realismo) [m/s^2]
    'mu': 0.00,       # Coeficiente de rozamiento (0 para sin rozamiento)
}

#PARA CAMBIAR LA POSICIÓN INICIAL DE LA m1 Y MÁS, BAJE , ESTA SE ENCUENTRA MÁS ABAJO EN EL CÓDIGO, EN LA SECCIÓN DE CONDICIONES INICIALES

# SELECCIÓN DE CASOS LÍMITE (Criterio 6)
# 0: Personalizado (usa las C.I. declaradas abajo)
# 1: Masa m2 muy pequeña (Masa m1 como oscilador armónico casi puro)
# 2: Carro muy pesado m1 >> m2 (Péndulo simple casi independiente)
# 3: Pequeñas oscilaciones (Modos acoplados lineales)
# 4: Gran ángulo / Rotación completa (Régimen no lineal)
caso_limite = 0

# =====================================================================
# 2. CONDICIONES INICIALES [x0, vx0, theta0, w0]
# =====================================================================
if caso_limite == 1: # m2 -> 0
    params['m2'] = 0.001
    z0 = [1.5, 0.0, np.radians(10), 0.0]
elif caso_limite == 2: # m1 >> m2
    params['m1'] = 1000.0
    z0 = [params['l0'], 0.0, np.radians(30), 0.0]
elif caso_limite == 3: # Pequeñas oscilaciones
    z0 = [params['l0'] + 0.1, 0.0, np.radians(5), 0.0]
elif caso_limite == 4: # Gran ángulo / Rotación
    z0 = [params['l0'], 0.0, np.radians(170), 2.0]
else: # Configuración personalizada
    x0 = 2.5                  # Posición inicial del carro [m]
    vx0 = 0.0                 # Velocidad inicial del carro [m/s]
    theta0 = np.radians(45)   # Ángulo inicial [rad]
    w0 = 0.0                  # Velocidad angular inicial [rad/s]
    z0 = [x0, vx0, theta0, w0]

# =====================================================================
# 3. ECUACIONES DEL MOVIMIENTO Y SISTEMA DINÁMICO
# =====================================================================
def ecuaciones_movimiento(t, z, p):
    x, vx, theta, w = z

    # Matriz de masa A * [ax, alpha]^T = B
    A = np.array([
        [p['m1'] + p['m2'], p['m2'] * p['l'] * np.cos(theta)],
        [np.cos(theta), p['l']]
    ])
    
    F1 = p['m2'] * p['l'] * (w**2) * np.sin(theta) - p['k'] * (x - p['l0'])
    F2 = -p['g'] * np.sin(theta)
    
    B = np.array([F1, F2])
    
    # Resolver aceleraciones
    ax, alpha = np.linalg.solve(A, B)
    
    # Fuerza Normal (Apartado d)
    N = (p['m1'] + p['m2']) * p['g'] + p['m2'] * p['l'] * (alpha * np.sin(theta) + (w**2) * np.cos(theta))
    
    # Rozamiento con regularización suave para evitar la no diferenciabilidad
    # de sign(vx): esto mantiene la simulación estable y evita que RK45 se
    # vuelva muy lento al pasar por velocidad cercana a cero.
    if p['mu'] > 0:
        vel_suave = np.tanh(vx / 1e-3)
        F_roz = -p['mu'] * abs(N) * vel_suave
        B[0] += F_roz
        ax, alpha = np.linalg.solve(A, B)

    return [vx, ax, w, alpha]

def calcular_energia_y_normal(t_arr, z_arr, p):
    E_arr = []
    N_arr = []
    for i in range(len(t_arr)):
        x, vx, theta, w = z_arr[:, i]

        # Aceleraciones instantáneas
        dz = ecuaciones_movimiento(t_arr[i], [x, vx, theta, w], p)
        ax, alpha = dz[1], dz[3]
        
        # Fuerza normal
        N = (p['m1'] + p['m2']) * p['g'] + p['m2'] * p['l'] * (alpha * np.sin(theta) + (w**2) * np.cos(theta))
        
        # Energía cinética y potencial
        T = 0.5 * (p['m1'] + p['m2']) * (vx**2) + 0.5 * p['m2'] * (p['l']**2) * (w**2) + p['m2'] * p['l'] * vx * w * np.cos(theta)
        V = 0.5 * p['k'] * ((x - p['l0'])**2) - p['m2'] * p['g'] * p['l'] * np.cos(theta)
        
        E_arr.append(T + V)
        N_arr.append(N)
        
    return np.array(E_arr), np.array(N_arr)

# =====================================================================
# 4. RESOLUCIÓN NUMÉRICA
# =====================================================================
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 600) # 600 frames para animación fluida

sol = solve_ivp(
    fun=lambda t, z: ecuaciones_movimiento(t, z, params),
    t_span=t_span,
    y0=z0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-5,
    atol=1e-7
)

t = sol.t
x, vx, theta, w = sol.y
E_tot, Normal = calcular_energia_y_normal(t, sol.y, params)

# Coordenadas cartesianas de m2
px1 = x
py1 = np.full_like(x, 0.15)
px2 = px1 + params['l'] * np.sin(theta)
py2 = -params['l'] * np.cos(theta)

# =====================================================================
# 5. ANIMACIÓN Y GRÁFICOS CON MATPLOTLIB
# =====================================================================
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('white')

# Subplot 1: Animación
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Animación del Movimiento en Vivo')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')

ax1.set_xlim(min(-0.5, float(np.min(px1)) - 0.8, float(np.min(px2)) - 0.5),
             max(x) + params['l'] + 0.8)
ax1.set_ylim(-params['l'] - 0.5, 0.8)

# Suelo y pared
ax1.plot([ax1.get_xlim()[0], max(x) + 1.5], [0, 0], 'k-', lw=2)

# Elementos móviles
line_resorte, = ax1.plot([], [], 'r--', lw=1.5, label='Resorte')
carro = plt.Rectangle((0, 0), 0.4, 0.3, fc='#3399CC', ec='k', zorder=3)
ax1.add_patch(carro)
line_pila, = ax1.plot([], [], 'k-', lw=2)
masa2, = ax1.plot([], [], 'ro', ms=10, zorder=4)
traza, = ax1.plot([], [], 'b:', lw=1, alpha=0.6)

# Subplot 2: Evolución Temporal
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.plot(t, x, 'b-', label='x(t) [m]')
ax2.plot(t, np.degrees(theta), 'r-', label=r'$\theta(t)$ [deg]')
ax2.set_title('Coordenadas Generalizadas')
ax2.set_xlabel('Tiempo (s)')
ax2.grid(True)
ax2.legend()

# Subplot 3: Energía y Normal
ax3 = plt.subplot2grid((2, 2), (1, 1))
color = 'green'
ax3.set_xlabel('Tiempo (s)')
ax3.set_ylabel('Energía Total (J)', color=color)
ax3.plot(t, E_tot, color=color, lw=1.5)
ax3.tick_params(axis='y', labelcolor=color)
ax3.grid(True)

ax3_right = ax3.twinx()
color = 'magenta'
ax3_right.set_ylabel('Fuerza Normal N (N)', color=color)
ax3_right.plot(t, Normal, color=color, linestyle='--', lw=1.2)
ax3_right.tick_params(axis='y', labelcolor=color)
ax3.set_title('Energía Mecánica y Fuerza Normal')

# Función de actualización para FuncAnimation
def animate(i):
    # Actualizar resorte
    line_resorte.set_data([0, px1[i]], [0.15, 0.15])
    # Actualizar posición carro
    carro.set_xy((px1[i] - 0.2, 0))
    # Actualizar péndulo
    line_pila.set_data([px1[i], px2[i]], [0.15, py2[i]])
    masa2.set_data([px2[i]], [py2[i]])
    # Actualizar traza
    traza.set_data(px2[:i+1], py2[:i+1])
    return line_resorte, carro, line_pila, masa2, traza

ani = animation.FuncAnimation(
    fig, animate, frames=len(t), interval=25, blit=True
)

plt.tight_layout()
plt.show()