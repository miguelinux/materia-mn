#!/usr/bin/env python3
# c-basic-offset: 4; tab-width: 8; indent-tabs-mode: nil
# vi: set shiftwidth=4 tabstop=8 expandtab:
# :indentSize=4:tabSize=8:noTabs=true:
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Ejemplo de un modulo
"""
from math import *
import numpy as np
import matplotlib.pyplot as plt

def test1(x): # y'' = x
    """Función de prueba"""
    return x


def test2(x): # y'' = -x
    """Función de prueba"""
    return -x

def test3(x):
    """Función de prueba"""
    return -9.8

def test4(t,x): 
    """Función de prueba"""
    return -9.8*t

def Verlet(a,b, alpha, beta, f, N):
    """
    Impleentación método de Verlet
    Entradas:
    a -- inicio intervalo
    b -- fin intervalo
    alpha -- aproximación inicial y0 = alpha
    beta -- aproximación inicial  y'0= beta
    f -- función
    N -- pasos

    Salida:
    y_i -- aproximación final
    """
    h = (b-a)/N
    y0 = alpha
    y_i = y0 + beta*h + 0.5 * f(y0) * h**2  # y_-1 (arranque)

    t_vals = np.arange(a, b + h, h) # Generar puntos de tiempo
    y_vals = np.zeros_like(t_vals) # Inicializar array para resultados

    print(f"a0  = {a:0.2f},  y0  = {y0:0.12f}")
    y_i_1 = y0 # yi-1

    for i in range(1, N+1):
        y_ip1 = 2*y_i - y_i_1 + f(y_i) * h**2
        s = a + i*h
        print(f"a{i:<2} = {s:0.2f},  y_i{i:<2} = {y_i:.12f}")
        y_vals[i-1] = y_i_1

        y_i_1 = y_i
        y_i = y_ip1

    y_vals[-1] = y_i_1

    return t_vals, y_vals

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


def main():
    """
    Principal
    """
    # y''= x, a=0, b=1, alpha=1, beta = 1, N = 20
    print("Método de Verlet:")
    Verlet(0,1,1,1,test1,20)

    # y''= -x, a=0, b=1, alpha=1, beta = 0, N = 20
    print("Método de Verlet:")
    Verlet(0,1,1,0,test2,20)

    # y''= -x, a=0, b=1, alpha=1, beta = 0, N = 20
    print("Método de Verlet:")
    v, y = Verlet(0,4.5,100, 0, test3, 10)

    ve, ye  = euler_method(test4, 100, (0,4.5), 4.5/10)
    print(ve)
    print(ye)
    
    #plt.figure(figsize=(10, 6))
    plt.figure()
    #plt.plot(t_exacta, y_exacta, 'k-', label='Solución Exacta ($e^t$)', linewidth=2)
    #plt.plot(t_euler, y_euler, 'bo--', label=f'Método de Euler (h={paso_h})', markersize=6)
    #plt.plot(t_rk4, y_rk4, 'rs--', label=f'Método RK4 (h={paso_h})', markersize=6)
    plt.plot(v,y)
    plt.plot(ve,ye)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

