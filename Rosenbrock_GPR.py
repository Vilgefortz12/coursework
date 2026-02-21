import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


X_train = np.random.uniform(-2, 2, ( 50 , 2 ))
Y_train = np.array([rosenbrock(x) for x in X_train])

kernel = RBF(length_scale = 1)

gpr = GaussianProcessRegressor(
    kernel = kernel,
    n_restarts_optimizer = 15,
    random_state = 30 )



gpr.fit(X_train,Y_train)

X_test = np.linspace(-2, 2, 50).reshape(-1, 1)
X_test = np.hstack([X_test, np.zeros_like(X_test)])

y_mean = gpr.predict(X_test)


y_true = np.array([rosenbrock([x, 0]) for x in X_test[:, 0]])


plt.plot(X_test,y_mean,'r', label='Prediction')


plt.scatter(
    X_train[:, 0],  
    Y_train,       
    c='blue',
    s=15,
    label='Training points',
    zorder=5
)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('GPR on Rosenbrock functions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.plot(X_test[:, 0], y_true, 'k--', label='True function', alpha=0.7)

plt.show()

