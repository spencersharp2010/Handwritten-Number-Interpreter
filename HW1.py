#%%
import numpy as np
from matplotlib import pyplot as plt
#%%
# plot formatting

plt.rcParams['font.size'] = 7
plt.rcParams['lines.markersize'] = 4
plt.rcParams['figure.figsize'] = [5, 3]
plt.rcParams['figure.dpi'] = 150

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#%% md
# Problem 4

## Parts 1 and 2
#%%
n = 1000
x = np.random.rand(n)
y = x + 0.5*np.random.randn(n)

plt.scatter(x, y, c=colors[0], label='Data')

a = (x @ y) / (x @ x)
plt.plot([0, 1], [0, a], c=colors[1], label='Best fit line $a = %g$' % a)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Random data and best fit line')
plt.legend()
plt.show()
#%% md
## Part 4
#%%
z = 30 * (x - 0.25)**2 * (x - 0.75)**2 + 0.1 * np.random.randn(n)

plt.scatter(x, z, c=colors[0], label='Data')

mX = x[:, None] ** np.arange(4 + 1)[None, :]
a2 = np.linalg.pinv(mX) @ z
mT = np.linspace(0, 1)[:, None] ** np.arange(4 + 1)[None, :]
polynomial_str = '%.2f' % a2[0]
for i, coeff in enumerate(a2[1:]):
    if i == 0:
        polynomial_str += ' %s %.2fx' % ('+' if coeff >= 0 else '-', abs(coeff))
    else:
        polynomial_str += ' %s %.2fx^%d' % ('+' if coeff >= 0 else '-', abs(coeff), i + 1)
plt.plot(mT[:, 1], mT @ a2, c=colors[1], label='Best fit $p_4(x) = %s$' % polynomial_str)

plt.xlabel('$x$')
plt.ylabel('$z$')
plt.title('Random data and best fit polynomial')
plt.legend()
plt.show()
#%%
