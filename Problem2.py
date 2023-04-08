import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import os


def threshold_plot(ax, x, y, threshv, undercolor, overcolor):
    cmap = ListedColormap([undercolor, overcolor])
    norm = BoundaryNorm([min(0, np.min(y)), threshv, np.max(y)], cmap.N)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)

    ax.add_collection(lc)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(min(0,np.min(y)*1.1), np.max(y)*1.1)

    return lc


def plot_time_series(t, U, Gamma, part):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    ax1.plot(t, np.real(U))
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.set_ylabel(r'Re $u_t$')
    ax2.plot(t, np.imag(U))
    ax2.set_xlim(np.min(t), np.max(t))
    ax2.set_ylabel(r'Im $u_t$')

    lc = threshold_plot(ax3, t, Gamma, 0, 'red', 'blue')
    ax3.axhline(0, color='k', ls='--')
    ax3.set_ylabel(r'$\gamma$')

    OUT_DIR = os.path.join(os.getcwd(), 'pic')
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'p2_part_{}.png'.format(part))

    plt.savefig(path)
    plt.close()
    #plt.show()


def main(T, dt, omega, sigma_u, d_gamma, gamma_hat, sigma_gamma, part):
    t = np.arange(0, T + dt/2, dt)
    Nt = np.size(t)

    print('mean: ', gamma_hat)
    print('std: ', sigma_gamma/(2*d_gamma)**(1/2))

    # generate time series of the parameter gamma
    Gamma = np.zeros(Nt)
    Gamma[0] = np.random.normal(gamma_hat, sigma_gamma)
    for it in range(1, Nt):
        Gamma[it] = Gamma[it-1] - d_gamma*(Gamma[it-1] - gamma_hat)*dt + sigma_gamma*(dt)**(1/2)*np.random.normal(0, 1)
    # generate time series of U
    U = np.zeros(Nt, dtype='complex_')
    U[0] = np.random.normal(0, 1)

    for it in range(1, Nt):
        U[it] = U[it-1] + (-Gamma[it-1] + 1j*omega)*U[it-1]*dt + sigma_u*(dt)**(1/2)*np.random.normal(0, 1)

    plot_time_series(t, U, Gamma, part)

T = 100
dt = 0.01
omega = 1
sigma_u = 2

# mean
#gamma_hat = 3

# variance is sigma_gamma**2/2*d_gamma
d_gamma = 0.5
#sigma_gamma = 2.5

part_a = {'gamma_hat': 0.2, 'sigma_gamma': 0.25, 'part': 'a'}
part_b = {'gamma_hat': 4, 'sigma_gamma': 2.5, 'part': 'b'}
part_c = {'gamma_hat': 3, 'sigma_gamma': 0.1, 'part': 'c'}

parts = [part_a, part_b, part_c]

for part in parts:
    main(T, dt, omega, sigma_u, d_gamma, **part)