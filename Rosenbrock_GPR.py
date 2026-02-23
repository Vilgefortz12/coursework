import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared

def rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


plt.figure(figsize=(15, 10))


scales = [0.01, 0.05, 0.2, 0.5, 1, 10, 100, 1000, 10000]


for i, scale in enumerate(scales):

    plt.subplot(3, 3, i+1)

    X_train = np.random.uniform(-2, 2, ( 50 , 2 ))
    Y_train = np.array([rosenbrock(x) for x in X_train])


    kernel = RBF(length_scale=scale, length_scale_bounds='fixed')


    gpr = GaussianProcessRegressor(
    kernel = kernel,
    n_restarts_optimizer = 0,
    random_state = 30,
    alpha=1e-10)



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
    plt.title(f'GPR on Rosenbrock functions, scale = {scale}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.plot(X_test[:, 0], y_true, 'k--', label='True function', alpha=0.7)

plt.suptitle('Comparison of different length_scale values', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


n_restarts = [0,1,2,3,5,10]


plt.figure(figsize=(15, 10))


for i, n_restart in enumerate(n_restarts):

    plt.subplot(3, 3, i+1)

    X_train = np.random.uniform(-2, 2, ( 50 , 2 ))
    Y_train = np.array([rosenbrock(x) for x in X_train])


    kernel = RBF(length_scale= 1,  length_scale_bounds=(1e-5, 1e5))


    gpr = GaussianProcessRegressor(
    kernel = kernel,
    n_restarts_optimizer = n_restart,
    random_state = 30,
    alpha=1e-10)



    gpr.fit(X_train,Y_train)

    optimized_scale = gpr.kernel_.length_scale

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
    plt.title(f'GPR on Rosenbrock functions, number of restarts = {n_restart}, scale = {optimized_scale:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.plot(X_test[:, 0], y_true, 'k--', label='True function', alpha=0.7)


plt.suptitle('Effect of n_restarts_optimizer on hyperparameter optimization', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


kernels = [
    RBF(length_scale = 1),
    Matern(length_scale= 1, nu = 0.1),
    Matern(length_scale= 1, nu = 0.5),
    Matern(length_scale= 1, nu = 2),
    RationalQuadratic(length_scale=1.0, alpha=1.0)
    ]

kernel_names = [
    'RBF',
    'Matern (nu=0.1)',
    'Matern (nu=0.5)',
    'Matern (nu=2)',
    'Rational Quadratic'
]

plt.figure(figsize=(15, 10))

X_train = np.random.uniform(-2, 2, (30, 2))
Y_train = np.array([rosenbrock(x) for x in X_train])

for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    plt.subplot(2, 3, i+1)
    
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=30
    )
    
    gpr.fit(X_train, Y_train)
    
    X_test = np.linspace(-2, 2, 200).reshape(-1, 1)
    X_test = np.hstack([X_test, np.zeros_like(X_test)])
    y_mean = gpr.predict(X_test)
    y_true = np.array([rosenbrock([x, 0]) for x in X_test[:, 0]])
    
    plt.plot(X_test[:, 0], y_mean, 'r', label='Prediction')
    plt.scatter(X_train[:, 0], Y_train, c='blue', s=15, alpha=0.5)
    plt.plot(X_test[:, 0], y_true, 'k--', alpha=0.7)
    
    opt_scale = gpr.kernel_.length_scale 
    plt.title(f'{name}\nscale={opt_scale:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)


plt.suptitle('Comparison of different kernels for Gaussian Process Regression', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()
