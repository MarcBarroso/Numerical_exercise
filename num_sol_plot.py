#Including all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

#We define the functions for plotting, in order to keep the next sections as clean as possible.
def plot_trajectory(x, y, t):
    plt.figure()
    plt.title("Trajectories as function of time")
    plt.xlabel('t')
    plt.ylabel('position')
    plt.plot(t,x,label='x(t)')
    plt.plot(t,y,label='y(t)')
    plt.legend()

def plot_velocities(dx, dy, t):
    plt.figure()
    plt.title("Velocities as function of time")
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.plot(t,dx,label='vx(t)')
    plt.plot(t,dy,label='vy(t)')
    plt.legend()

def plot_energy(sol, t):
    plt.figure()
    plt.title("Energy as function of time")
    plt.xlabel('t')
    plt.ylabel('energy')
    energ = energy_fct(sol.sol(t)[2].T, sol.sol(t)[3].T, sol.sol(t)[0].T, sol.sol(t)[1].T)
    plt.plot(t, energ,label='vx')

def plot_xy_plane(x, y):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Parametric plot of trajectory")
    plt.plot(x, y)

def plot_xy_plane(x1, y1, x2=None, y2=None):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Parametric plot of trajectory")
    plt.plot(x1, y1)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2)

def plot_vxvy_plane(dx, dy):
    plt.figure()
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.title("Parametric plot of velocity")
    plt.plot(dx,dy)

def plot_pendulum_parametric(sol, t):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Parametric plot of trajectory")
    xt = sol.sol(t)[2].T
    theta = sol.sol(t)[3].T
    plt.plot(xt+np.sin(theta), -np.cos(theta))

def animate(x, y, t, r=None, f_skip=None):
    fig, ax = plt.subplots()
    xmin = np.min(x)
    xmin = xmin+0.1*xmin
    xmax = np.max(x)
    xmax = xmax+0.1*xmax
    ymin = np.min(y)
    ymin = ymin+0.1*ymin
    ymax = np.max(y)
    ymax = ymax+0.1*ymax

    if r is None:
        r = 0.05

    if f_skip is None:
        f_skip = int(len(t)*0.01);

    def animate_frame(i):
        #plt.cla()
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        p = Circle((x[i], y[i]), r, fc='b', ec='b', zorder=10)
        ax.add_patch(p)
        #ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')

    for i in range(0, len(t), f_skip):
        animate_frame(i)
        plt.pause(0.01)

    plt.show()

def anim_double_pendulum(x1, y1, x2, y2, t):
    fig, ax = plt.subplots()
    xmin = np.min(x2)-0.5
    xmax = np.max(x2)+0.5
    ymin = np.min(y2)-0.5
    ymax = np.max(y2)+0.5

    def animate_frame(i):
        plt.cla()
        ax.set_xlim( xmin, xmax)
        ax.set_ylim( ymin, ymax)
        ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
        p = Circle((0, 0), 0.05, fc='b', ec='b', zorder=10)
        c1 = Circle((x1[i], y1[i]), 0.05, fc='r', ec='b', zorder=10)
        c2 = Circle((x2[i], y2[i]), 0.05, fc='r', ec='b', zorder=10)

        ax.add_patch(p)
        ax.add_patch(c1)
        ax.add_patch(c2)

    for i in range(0, len(t), 15):
        animate_frame(i)
        plt.pause(0.01)

    plt.show()


#This is the general solver for the equations of motion of two variables, x and y (not necessarily the
#Cartesian coordinates, can be generalized coordinates).
#Inputs: t0 and tf - initial and final time
#        x0, y0 - position of the particle at t0, that is, x(t0), y(t0)
#        vx0, vy0 - velocity of the particle at t0, that is, \dot{x}(t0), \dot{y}(t0)
#        n - number of points in which time is divided into
#Output: it returns t, x(t), y(t), vx(t), vy(t)
def solve_eom(t0, tf, x0, y0, vx0, vy0, force_fct, n=None, max_stp=None):
    init_cond = [vx0, vy0, x0, y0]

    def eom(t, p):
        vx, vy, x, y = p
        return force_fct(t, x, y, vx, vy)

    if max_stp is None:
        sol = solve_ivp(eom, [t0, tf], init_cond, method='DOP853', dense_output=True)
    else:
        sol = solve_ivp(eom, [t0, tf], init_cond, method='DOP853', dense_output=True, max_step=max_stp)

    if sol.success == False:
        print("Sorry, there has been an error solving the ODEs!")

    if n is None:
        n = int(np.ceil((tf-t0)*20))

    t = np.linspace(t0, tf, n)
    return [sol.sol(t)[2].T, sol.sol(t)[0].T, sol.sol(t)[3].T, sol.sol(t)[1].T, t]
