import numpy as np
import matplotlib.pyplot as plt

# Definimos la función f(t, y) de la EDO dy/dt = f(t, y)
# Ejemplo: dy/dt = y
def f(t, y):
    return y

# Solución analítica para comparación (y(t) = e^t)
def solucion_exacta(t):
    return np.exp(t)

#### 1. Implementación del Método de Euler ####
def euler_method(f, y0, t_span, h):
    """
    Resuelve una EDO dy/dt = f(t, y) usando el método de Euler.

    Args:
        f (function): La función f(t, y) que define la EDO.
        y0 (float): La condición inicial y(t_span[0]).
        t_span (tuple): Una tupla (t_inicio, t_fin) del intervalo de tiempo.
        h (float): El tamaño de paso.

    Returns:
        tuple: Arrays de tiempo (t_vals) y valores aproximados de y (y_vals).
    """
    t_inicio, t_fin = t_span
    t_vals = np.arange(t_inicio, t_fin + h, h) # Generar puntos de tiempo
    y_vals = np.zeros_like(t_vals) # Inicializar array para resultados
    
    y_vals[0] = y0 # Asignar la condición inicial
    
    print("\n--- Método de Euler ---")
    print(f"t_k       y_k (Aprox)       f(t_k, y_k)       y_k+1 (Aprox)")
    print(f"{t_vals[0]:.4f}    {y_vals[0]:.6f}         ---                ---")

    for i in range(len(t_vals) - 1):
        y_k = y_vals[i]
        t_k = t_vals[i]
        
        # Fórmula de Euler
        dy_dt_at_k = f(t_k, y_k)
        y_vals[i+1] = y_k + h * dy_dt_at_k
        
        print(f"{t_vals[i+1]:.4f}    {y_vals[i+1]:.6f}         {dy_dt_at_k:.6f}         {y_vals[i+1]:.6f}")
        
    return t_vals, y_vals

#### 2. Implementación del Método de Runge-Kutta de Orden 4 (RK4) ####
def rk4_method(f, y0, t_span, h):
    """
    Resuelve una EDO dy/dt = f(t, y) usando el método de Runge-Kutta de Orden 4.

    Args:
        f (function): La función f(t, y) que define la EDO.
        y0 (float): La condición inicial y(t_span[0]).
        t_span (tuple): Una tupla (t_inicio, t_fin) del intervalo de tiempo.
        h (float): El tamaño de paso.

    Returns:
        tuple: Arrays de tiempo (t_vals) y valores aproximados de y (y_vals).
    """
    t_inicio, t_fin = t_span
    t_vals = np.arange(t_inicio, t_fin + h, h)
    y_vals = np.zeros_like(t_vals)
    
    y_vals[0] = y0
    
    print("\n--- Método de Runge-Kutta de Orden 4 ---")
    print(f"t_k       y_k (Aprox)       k1        k2        k3        k4        y_k+1 (Aprox)")
    print(f"{t_vals[0]:.4f}    {y_vals[0]:.6f}    ---       ---       ---       ---       ---")

    for i in range(len(t_vals) - 1):
        y_k = y_vals[i]
        t_k = t_vals[i]
        
        # Cálculos de k1, k2, k3, k4
        k1 = h * f(t_k, y_k)
        k2 = h * f(t_k + h/2, y_k + k1/2)
        k3 = h * f(t_k + h/2, y_k + k2/2)
        k4 = h * f(t_k + h, y_k + k3)
        
        # Fórmula RK4
        y_vals[i+1] = y_k + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        print(f"{t_vals[i+1]:.4f}    {y_vals[i+1]:.6f}    {k1:.6f}    {k2:.6f}    {k3:.6f}    {k4:.6f}    {y_vals[i+1]:.6f}")

    return t_vals, y_vals

# --- Configuración del Problema ---
y0_val = 1.0        # Condición inicial y(0) = 1
t_intervalo = (0.0, 1.0) # Queremos resolver de t=0 a t=1
paso_h = 0.2        # Tamaño de paso (lo haremos pequeño para Euler, luego veremos con más grande)

# --- Ejecución y Comparación ---

# 1. Resolver con Método de Euler
t_euler, y_euler = euler_method(f, y0_val, t_intervalo, paso_h)

# 2. Resolver con Método de Runge-Kutta de Orden 4
t_rk4, y_rk4 = rk4_method(f, y0_val, t_intervalo, paso_h)

# 3. Calcular la solución exacta para comparación
t_exacta = np.linspace(t_intervalo[0], t_intervalo[1], 100) # Más puntos para una curva suave
y_exacta = solucion_exacta(t_exacta)

# 4. Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(t_exacta, y_exacta, 'k-', label='Solución Exacta ($e^t$)', linewidth=2)
plt.plot(t_euler, y_euler, 'bo--', label=f'Método de Euler (h={paso_h})', markersize=6)
plt.plot(t_rk4, y_rk4, 'rs--', label=f'Método RK4 (h={paso_h})', markersize=6)

plt.title('Comparación de Métodos de Solución de EDOs para dy/dt = y, y(0)=1')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# --- Mostrar errores al final del intervalo ---
print("\n--- Resumen de Resultados ---")
print(f"Solución Exacta en t=1.0: {solucion_exacta(1.0):.10f}")
print(f"Euler Aproximación en t=1.0: {y_euler[-1]:.10f}, Error Absoluto: {abs(solucion_exacta(1.0) - y_euler[-1]):.10f}")
print(f"RK4 Aproximación en t=1.0: {y_rk4[-1]:.10f}, Error Absoluto: {abs(solucion_exacta(1.0) - y_rk4[-1]):.10f}")

# --- Prueba con un paso más grande para RK4 para mostrar su robustez ---
paso_h_grande_rk4 = 0.5
t_rk4_grande, y_rk4_grande = rk4_method(f, y0_val, t_intervalo, paso_h_grande_rk4)
print(f"\nRK4 Aproximación en t=1.0 (h={paso_h_grande_rk4}): {y_rk4_grande[-1]:.10f}, Error Absoluto: {abs(solucion_exacta(1.0) - y_rk4_grande[-1]):.10f}")
