# %%
from collections import OrderedDict
import csv
import numpy as np
from matplotlib import pyplot as plt

# %%
# plot formatting

plt.rcParams['font.size'] = 10
plt.rcParams['lines.markersize'] = 4
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['figure.dpi'] = 150

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# %% md
# Problem 2

## Parts 2--7
# %%
x_0 = np.zeros(2)
b = np.array([4.5, 6])
n_iter = 5000

f_grad = lambda x: (x - b) / np.linalg.norm(x - b)
g_grad = lambda x: 2 * (x - b)

f_step_sizes = OrderedDict([
    ('1', lambda k: 1),
    ('(5/6)^k', lambda k: (5 / 6) ** k),
    ('1/(k+1)', lambda k: 1 / (k + 1))
])
g_step_sizes = OrderedDict([
    ('0.1', lambda k: 0.1),
    ('(1/6)^k', lambda k: (1 / 6) ** k),
    ('1/(4(k+1))', lambda k: 1 / 4 / (k + 1))
])

f_k_estimates = [None, None, 942]
g_k_estimates = [21, None, 3183]


def run_gradient_descent(grad, x_0, step_size, n_iter=1000):
    xs = np.zeros((n_iter + 1, x_0.size))
    x = x_0.copy()
    xs[0, :] = x

    for k in range(n_iter):
        x = x - step_size(k) * grad(x)
        xs[k + 1, :] = x

    return xs


def relative_distance_from(xs, b):
    squared_dists = (xs ** 2).sum(1) - 2 * (xs @ b) + (b ** 2).sum()
    return np.sqrt(squared_dists) / np.linalg.norm(b)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
for i, (label, step_size) in enumerate(f_step_sizes.items()):
    plt.loglog(relative_distance_from(run_gradient_descent(f_grad, x_0, step_size, n_iter=n_iter), b),
               label=r'$\alpha_k = %s$' % label)
    if f_k_estimates[i] is not None:
        plt.plot([f_k_estimates[i]] * 2, [1e-4, 3], '--', label=r'$k=%d$' % f_k_estimates[i])
plt.plot([0, n_iter], [0.01, 0.01], ':k', label=r'$1\%$')

plt.ylim(1e-4, 3)
plt.title(r'Minimizing $f(\mathbf{x})$')
plt.xlabel('$k$')
plt.ylabel(r'$\|\mathbf{x} - \mathbf{x^*}\|_2 / \|\mathbf{x^*}\|_2$')
plt.legend()

plt.subplot(1, 2, 2)
for i, (label, step_size) in enumerate(g_step_sizes.items()):
    plt.loglog(relative_distance_from(run_gradient_descent(g_grad, x_0, step_size, n_iter=n_iter), b),
               label=r'$\alpha_k = %s$' % label)
    if g_k_estimates[i] is not None:
        plt.plot([g_k_estimates[i]] * 2, [1e-4, 3], '--', label=r'$k=%d$' % g_k_estimates[i])
plt.plot([0, n_iter], [0.01, 0.01], ':k', label=r'$1\%$')

plt.ylim(1e-4, 3)
plt.title(r'Minimizing $g(\mathbf{x})$')
plt.xlabel('$k$')
plt.ylabel(r'$\|\mathbf{x} - \mathbf{x^*}\|_2 / \|\mathbf{x^*}\|_2$')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
# %% md
# Problem 3

## Part 1
# %%
# load feature variables and their names
X = np.loadtxt('hitters.x.csv', delimiter=',', skiprows=1)
with open('hitters.x.csv', 'r') as f:
    X_colnames = next(csv.reader(f))

# load salaries
y = np.loadtxt('hitters.y.csv', delimiter=',', skiprows=1)

X -= X.mean(0)[None, :]
X /= X.std(0)[None, :]

# %% md
## Part 2
# %%
X_aug = np.hstack((np.ones((X.shape[0], 1)), X))


def ridge(X_aug, y, lamda):
    eye_aug = np.eye(X_aug.shape[1])
    eye_aug[0, 0] = 0
    return np.linalg.inv(X_aug.T @ X_aug + lamda * eye_aug) @ (X_aug.T @ y)


# %% md
## Part 3
# %%
lamdas = np.logspace(-3, 7, 100)
theta_hats = np.array([ridge(X_aug, y, lamda) for lamda in lamdas])

norm_penalties = np.sqrt((theta_hats[:, 1:] ** 2).sum(1))

plt.loglog(lamdas, norm_penalties)

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\|\hat{\theta}_\mathrm{ridge}\|_2$')
plt.title('Penalty term versus regularization parameter')
plt.show()
# %% md
## Part 4
# %%
theta_mse = ridge(X_aug, y, 0)

for j, theta in enumerate(theta_mse):
    plt.semilogx(lamdas, np.ones_like(lamdas) * theta, ':', c=plt.cm.tab20(j / 20))
    plt.semilogx(lamdas, theta_hats[:, j], c=plt.cm.tab20(j / 20))

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\hat{\theta}_j$')
plt.show()


# %% md
## Part 5
# %%
def cv_ridge(X_aug, y, lamda, n_folds=5):
    fold_size = X_aug.shape[0] // n_folds
    perm = np.random.permutation(X_aug.shape[0])

    errors = []

    for k in range(n_folds):
        test_mask = np.zeros(X_aug.shape[0], dtype=bool)
        test_mask[k * fold_size:(k + 1) * fold_size] = True
        train_mask = np.logical_not(test_mask)

        X_train, X_test = X_aug[perm[train_mask], :], X_aug[perm[test_mask], :]
        y_train, y_test = y[perm[train_mask]], y[perm[test_mask]]

        theta = ridge(X_train, y_train, lamda)
        errors.append(((y_test - X_test @ theta) ** 2).mean())

    return errors


cv_errors = np.sqrt(np.array([cv_ridge(X_aug, y, lamda, n_folds=5) for lamda in lamdas]).mean(1))

plt.semilogx(lamdas, cv_errors)

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$RMSE_{CV}$')
plt.show()

best_lamda_i = np.argmin(cv_errors)
best_lamda = lamdas[best_lamda_i]
print('Best value of lambda: %g' % best_lamda)
# %% md
## Part 6
# %%
best_theta = ridge(X_aug, y, best_lamda)

for i, feature_name in sorted(enumerate(['bias'] + X_colnames), key=lambda x: best_theta[x[0]]):
    print('%s: %g' % (feature_name, best_theta[i]))


# %% md
# Problem 4

## Part 5
# %%
def simulate_part_4(D, n, w0=1, w1=1, sigma=1):
    alphas = np.random.rand(n) * (1 - (-1)) + (-1)
    y_star = w1 * alphas + w0
    y = y_star + sigma * np.random.rand(n)
    p = np.poly1d(np.polyfit(alphas, y, D))
    return ((p(alphas) - y_star) ** 2).mean()


n_trials = 1000

Ds = list(range(1, 10))
ns = [10, 20, 50, 100, 200]

mse = np.zeros((len(ns), len(Ds)))

for i, n in enumerate(ns):
    for j, D in enumerate(Ds):
        mse[i, j] = np.mean([simulate_part_4(D, n) for _ in range(n_trials)])

plt.figure(figsize=(8, 3.5))

plt.subplot(1, 2, 1)
for i, n in enumerate(ns):
    plt.plot(Ds, mse[i, :], label='$n=%d$' % n)

plt.title('Polynomial fit error of linear function, $D$ scaling')
plt.ylabel(r'$\frac{1}{n}\|\mathbf{X}\hat{\mathbf{w}} - \mathbf{y^*}\|_2^2$')
plt.xlabel('$D$')
plt.legend()

plt.subplot(1, 2, 2)
for j, D in enumerate(Ds):
    plt.plot(ns, mse[:, j], label='$D=%d$' % D)

plt.title('Polynomial fit error of linear function, $n$ scaling')
plt.ylabel(r'$\frac{1}{n}\|\mathbf{X}\hat{\mathbf{w}} - \mathbf{y^*}\|_2^2$')
plt.xlabel('$n$')
plt.legend()

plt.tight_layout()
plt.show()


# %% md
## Part 7
# %%
def simulate_part_6(D, n, sigma=1):
    alphas = np.random.rand(n) * (3 - (-4)) + (-4)
    y_star = np.exp(alphas)
    y = y_star + sigma * np.random.rand(n)
    p = np.poly1d(np.polyfit(alphas, y, D))
    return ((p(alphas) - y_star) ** 2).mean()


n_trials = 1000

Ds = list(range(1, 10))
ns = [10, 20, 50, 120]

mse = np.zeros((len(ns), len(Ds)))

for i, n in enumerate(ns):
    for j, D in enumerate(Ds):
        mse[i, j] = np.mean([simulate_part_6(D, n) for _ in range(n_trials)])

plt.figure(figsize=(8, 3.5))

plt.subplot(1, 2, 1)
for i, n in enumerate(ns):
    plt.semilogy(Ds, mse[i, :], label='$n=%d$' % n)

plt.title('Polynomial fit error of exponential function, $D$ scaling')
plt.ylabel(r'$\frac{1}{n}\|\mathbf{X}\hat{\mathbf{w}} - \mathbf{y^*}\|_2^2$')
plt.xlabel('$D$')
plt.legend()

plt.subplot(1, 2, 2)
for j, D in enumerate(Ds):
    plt.semilogy(ns, mse[:, j], label='$D=%d$' % D)

plt.title('Polynomial fit error of exponential function, $n$ scaling')
plt.ylabel(r'$\frac{1}{n}\|\mathbf{X}\hat{\mathbf{w}} - \mathbf{y^*}\|_2^2$')
plt.xlabel('$n$')
plt.legend()

plt.tight_layout()
plt.show()
# %%
