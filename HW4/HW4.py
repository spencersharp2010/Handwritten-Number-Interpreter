# %%

import csv
from zipfile import ZipFile

import numpy as np
from matplotlib import pyplot as plt

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

# Problem 1

## Part 3 and 4

# %%

with ZipFile('nuclear.zip', 'r') as zf:
    with zf.open('nuclear_x.csv', 'r') as f:
        X = np.loadtxt(f, delimiter=',')

    with zf.open('nuclear_y.csv', 'r') as f:
        y = np.loadtxt(f, delimiter=',', dtype=int)


# %%

def batch_hinge_subgradient(X, y, w, b):
    hinge_mask = (((X @ w + b) * y) <= 1)

    dw = -(y[hinge_mask, None] * X[hinge_mask, :]).sum(0)
    db = -y[hinge_mask].sum()

    return dw, db


def run_stochastic_sgm(X, y, step_size, batch_size=1, lamda=0.001, n_iter=100):
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    ws = [w.copy()]
    bs = [b]

    for j in range(1, n_iter + 1):

        alpha = step_size(j)
        perm = np.random.permutation(n)

        for k in range(0, n, batch_size):
            indices = perm[k * batch_size:(k + 1) * batch_size]

            dw, db = batch_hinge_subgradient(X[indices, :], y[indices], w, b)
            dw += lamda * w * len(indices)
            dw /= n
            db /= n

            w -= alpha * dw
            b -= alpha * db
            ws.append(w.copy())
            bs.append(b)

    return ws, bs


step_size = lambda j: 100 / j
batch_size = X.shape[0]
lamda = 0.001
n_iter = 50

ws, bs = run_stochastic_sgm(X, y, step_size, batch_size=batch_size, lamda=lamda, n_iter=n_iter)
ws_sto, bs_sto = run_stochastic_sgm(X, y, step_size, batch_size=1, lamda=lamda, n_iter=n_iter)


# %%

def compute_Js(X, y, ws, bs, lamda):
    Js = []
    for w, b in zip(ws, bs):
        ts = X @ w + b
        hinges = np.maximum(1 - ts * y, 0)
        Js.append(hinges.mean() + lamda / 2 * np.linalg.norm(w) ** 2)

    return Js


Js = compute_Js(X, y, ws, bs, lamda)
# only evaluate J every 100 iterations for stochastic SGM since it is expensive
sto_spacing = 100
Js_sto = compute_Js(X, y, ws_sto[::sto_spacing], bs_sto[::sto_spacing], lamda)

plt.plot(np.arange(len(Js)), Js, '--', color=colors[2], label='SGM')
plt.plot(np.arange(len(Js_sto)) / X.shape[0] * sto_spacing, Js_sto, ':', color=colors[3], label='Stochastic SGM')

plt.yscale('log')

plt.ylabel('J')
plt.xlabel('\# Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# %%

print('SGM:')
print('w0: %g, w1: %g,' % tuple(ws[-1]), 'b: %g,' % bs[-1], 'J: %g' % Js[-1])
print('Stochastic SGM:')
print('w0: %g, w1: %g,' % tuple(ws_sto[-1]), 'b: %g,' % bs_sto[-1], 'J: %g' % Js_sto[-1])

# %%

plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', facecolors='none', edgecolors=colors[0], alpha=0.1)
plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='x', facecolors=colors[1], alpha=0.1)


def plot_linear_decision_boundary(w, b, x_min, x_max, pattern, color, label):
    plt.plot([x_min, x_max], [-x_min * w[0] / w[1] - b / w[1], -x_max * w[0] / w[1] - b / w[1]], pattern, color=color,
             label=label)


x_min, x_max = X[:, 0].min(), X[:, 0].max()
plot_linear_decision_boundary(ws[-1], bs[-1], x_min, x_max, '--', colors[2], 'SGM')
plot_linear_decision_boundary(ws_sto[-1], bs_sto[-1], x_min, x_max, ':', colors[3], 'Stochastic SGM')
plt.ylabel('y')
plt.xlabel('x')
plt.title('data with learned decision boundary')
plt.legend()
plt.show()

# %% md

# Problem 3

# %%

# X_train = np.loadtxt('train_data.csv', delimiter=',', skiprows=1, dtype=float)[:, 1:]
# y_train = np.loadtxt('train_labels.csv', delimiter=',', skiprows=1, dtype=int)[:, 1]
# X_test = np.loadtxt('test_data.csv', delimiter=',', skiprows=1, dtype=float)[:, 1:]

# %% md

## Logistic regression baseline solution

# %%

# standard_scaler = StandardScaler()
# logistic_regression = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(
#     standard_scaler.fit_transform(X_train), y_train)
# y_hat_logistic_regression = logistic_regression.predict(standard_scaler.transform(X_test))

# with open('logistic_regression_sol.csv', 'w') as f:
#     csvwriter = csv.writer(f)
#     csvwriter.writerow(['Id', 'Category'])
#     csvwriter.writerows(enumerate(y_hat_logistic_regression))

# %% md

## Closest subspace solution

# Here is an
# example
# of
# a
# solution
# that
# can
# beat
# the
# logistic
# regression
# baseline, achieving
# about
# 94 % accuracy.The
# method
# computes
# the
# subspace
# containing
# 90 % of
# the
# energy
# for each class, and to classify an example measures which of these subspaces best preserves the feature vector (image).

# %%


class SubspaceProjector(object):

    def __init__(self, energy_fraction=0.9):
        self.energy_fraction = energy_fraction
        self.V = None

    def fit(self, X):
        # identify subspace containing self.energy_fraction of the energy
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        energy_fractions = np.cumsum(s ** 2)
        energy_fractions /= energy_fractions[-1]
        k = np.argmax(energy_fractions >= self.energy_fraction)

        self.V = Vh[:k, :].T

        return self

    def transform(self, X):
        if self.V is None:
            raise NotFittedError()

        return (X @ self.V) @ self.V.T


class ClosestSubspaceClassifier(object):

    def __init__(self, energy_fraction=0.9):
        self.energy_fraction = energy_fraction
        self.labels = None
        self.subspace_projectors = None

    def fit(self, X, y):
        self.labels = np.array(sorted(set(y)))
        self.subspace_projectors = [SubspaceProjector(self.energy_fraction).fit(X[y == label, :]) for label in
                                    self.labels]

        return self

    def predict(self, X):
        if self.labels is None or self.subspace_projectors is None:
            raise NotFittedError()

        closenesses = np.stack([
            np.linalg.norm(self.subspace_projectors[i].transform(X), axis=1)
            for i in range(len(self.labels))
        ], axis=1) / np.linalg.norm(X, axis=1)[:, None]

        return self.labels[np.argmax(closenesses, 1)]


# %%

# closest_subspace_classifier = ClosestSubspaceClassifier().fit(X_train, y_train)
# y_hat_closest_subspace = closest_subspace_classifier.predict(X_test)

# with open('closest_subspace_sol.csv', 'w') as f:
#     csvwriter = csv.writer(f)
#     csvwriter.writerow(['Id', 'Category'])
#     csvwriter.writerows(enumerate(y_hat_closest_subspace))

# %%


