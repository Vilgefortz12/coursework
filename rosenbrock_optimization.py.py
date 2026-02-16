import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

trajectory = [] 


def callback(x):
    trajectory.append(x.copy())


result = minimize( rosenbrock, x0 = [-1, 1], method = 'BFGS', tol = 1e-6, callback =callback ,  options = {'disp': True})


print("Optimization trajectory:")
for i, point in enumerate(trajectory):
    print(f"Iteraton {i+1} : x = {point[0]:.6f}, y = {point[1]:.6f}, f = {rosenbrock(point):.6f}")


print("The point found", result.x)
print("Value", result.fun)


trajectory = np.array(trajectory)
x = np.linspace(-2,2,100)
y = np.linspace(-1,3,100)

X,Y = np.meshgrid(x,y)

Z = rosenbrock([X,Y])

plt.contour(X,Y,Z, levels = 20,colors='black', alpha=0.5)
plt.contourf(X,Y,Z, levels = 50, cmap = 'viridis')
plt.colorbar()


plt.title('Optimization trajectory of the Rosenbrock function')
plt.plot(trajectory[:,0], trajectory[:,1], 'r.-', label='Trajectory')
plt.legend()


plt.plot(trajectory[:,0],trajectory[:,1],'r.-')
plt.show()









