#Including all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#We define the functions for plotting, in order to keep the next sections as clean as possible.
def plot_trajectory(sol, t):
    plt.figure()
    plt.title("Trajectories as function of time")
    plt.xlabel('t')
    plt.ylabel('position')
    plt.plot(t,sol.sol(t)[2].T,label='x(t)')
    plt.plot(t,sol.sol(t)[3].T,label='y(t)')
    plt.legend()

def plot_velocities(sol, t):
    plt.figure()
    plt.title("Velocities as function of time")
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.plot(t,sol.sol(t)[0].T,label='vx(t)')
    plt.plot(t,sol.sol(t)[1].T,label='vy(t)')
    plt.legend()

def plot_energy(sol, t):
    plt.figure()
    plt.title("Energy as function of time")
    plt.xlabel('t')
    plt.ylabel('energy')
    energ = energy_fct(sol.sol(t)[2].T, sol.sol(t)[3].T, sol.sol(t)[0].T, sol.sol(t)[1].T)
    plt.plot(t, energ,label='vx')

def plot_xy_plane(sol, t):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Parametric plot of trajectory")
    plt.plot(sol.sol(t)[2].T,sol.sol(t)[3].T)

def plot_vxvy_plane(sol):
    plt.figure()
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.title("Parametric plot of velocity")
    plt.plot(sol.sol(t)[0],sol.sol(t)[1].T)

#This is the general solver for the equations of motion of two variables, x and y (not necessarily the
#Cartesian coordinates, can be generalized coordinates).
#Inputs: t0 and tf - initial and final time
#        x0, y0 - position of the particle at t0, that is, x(t0), y(t0)
#        vx0, vy0 - velocity of the particle at t0, that is, \dot{x}(t0), \dot{y}(t0)
#        n - number of points in which time is divided into
#Output: it returns t, x(t), y(t), vx(t), vy(t)
def solve_eom(t0, tf, x0, y0, vx0, vy0, force_fct, n=None):
    init_cond = [vx0, vy0, x0, y0]

    def eom(t, p):
        vx, vy, x, y = p
        return force_fct(t, x, y, vx, vy)

    sol = solve_ivp(eom, [t0, tf], init_cond, method='DOP853', dense_output=True)
    if sol.success == False:
        print("Sorry, there has been an error solving the ODEs!")

    if n is None:
        n = int(np.ceil((tf-t0)*20))

    t = np.linspace(t0, tf, n)
    return [sol, t]
