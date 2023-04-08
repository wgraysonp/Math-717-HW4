import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os


def main():
    # parameters
    b = 2
    c = 4
    lmbda = 1

    # discretized time domain
    t = np.linspace(0, 100, 1001)
    Nt = np.size(t)

    # Number of Monte-Carlo Trials
    N = 5000

    # array to hold final point in time series for each trial
    X_end = np.zeros(N)

    # run N trials
    for i in range(N):
        # generate a time series
        X_end[i] = gen_time_series(b, c, lmbda, t)[Nt-1]

    # create the histogram
    plot_empirical_pdf(b, c, X_end)


def gen_time_series(b, c, lmbda, t):
    Nt = np.size(t)
    dt = t[1]-t[0]
    X = np.zeros(np.size(t))

    X[0] = np.random.uniform(b, c)
    m = (b+c)/2

    for it in range(1, Nt):
        # set noise to zero outside of interval [b, c]
        if X[it-1] < b or X[it-1] > c:
            sigma = 0
        else:
            sigma = (lmbda*(X[it-1] - b)*(c - X[it-1]))**(1/2)

        # Euler-Maruyama step
        X[it] = X[it-1] - lmbda*(X[it-1] - m)*dt + sigma*dt**(1/2)*np.random.normal(0, 1)
    return X


def plot_empirical_pdf(b, c, X_end):
    N_bins = 40

    fig = plt.figure()
    ax = plt.subplot()
    N, bins, patches = ax.hist(X_end, bins=N_bins, density=True, label='PDF')

    # Setting color
    fracs = ((N ** (1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    #ax.legend()
    #plt.show()

    OUT_DIR = os.path.join(os.getcwd(), 'pic')
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'p1_pdf.png')

    plt.savefig(path)

    plt.close()


def plot_time_series(t, X):
    fig = plt.figure()

    ax = plt.subplot()

    ax.plot(t, X)

    plt.show()

    plt.close()

main()