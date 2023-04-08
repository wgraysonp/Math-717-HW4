import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def gen_time_series(t, a, f, sigma):
    Nt = np.size(t)
    dt = t[1] - t[0]
    X = np.zeros((2, Nt))
    X[0,0] = np.random.normal(0, 1)
    X[1, 0] = f
    dW = dt**(1/2)*np.random.normal(0, 1, Nt)

    for it in range(1, Nt):
        X[0, it] = X[0, it-1] + (-a*X[0, it-1] + f)*dt + sigma[0]*dW[it]
        X[1, it] = f

    return X


def model_update(dt, F, sigma, prev_mean, prev_cov):
    B = np.eye(2) + dt*F
    post_mean = B @ prev_mean
    post_cov = B @ prev_cov @ np.transpose(B) + dt*np.diag(sigma)
    return post_mean, post_cov


def kalman_forecast(t, obs_time_ratio, F, sigma, g1, g2, sigma_01, sigma_02, x0, X_true):
    Nt = np.size(t)
    dt = t[1] - t[0]

    prior_mean = np.zeros((2, Nt - 1))
    post_mean = np.zeros((2, Nt - 1))
    observations = []

    filtered_mean = [0, 0]
    filtered_cov = np.array([[9, 0], [0, 9]])

    R0 = sigma_01
    G = np.array([g1, g2])

    for it in range(1, Nt):
        model_mean, model_cov = model_update(dt, F, sigma, filtered_mean, filtered_cov)
        prior_mean[:, it-1] = model_mean

        if (it+1) % obs_time_ratio == 0:
            print('Yes!')

            true_data = X_true[:, it]

            # observations
            v = G.dot(true_data) + np.random.normal(0, 0.1)
            observations.append(v)

            # Kalman Gain
            K = model_cov.dot(G) * (G.dot(model_cov.dot(G)) + R0)**(-1)

            # posterior statistics
            filtered_mean = model_mean + K*(v - G.dot(model_mean))
            filtered_cov = (np.eye(2) - np.outer(K, G)) @ model_cov

            post_mean[:, it-1] = filtered_mean
        else:
            post_mean[:, it-1] = prior_mean[:, it-1]


    return prior_mean, post_mean, observations


def plot_forecast(mean_prior, mean_post, observations, t_obs, X, t, dir, dt_obs, show_ts):
    if show_ts:
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])

        ax1.plot(t, X[0, :], label=r'true signal', color='green')
        #ax1.plot(t_obs[1:], mean_post[0, :])
        ax1.plot(t_obs[1:], observations, linestyle='none', marker='o', markerfacecolor='none',
                                        color='red', label='observation')
        ax1.set_ylabel(r'$u_t$')

        ax1.legend()

        ax2.plot(t, X[1, :], label=r'true value', color='red', linestyle='--')
        ax2.plot(t[1:], mean_post[1, :], label=r'Estimation', color='green')
        ax2.set_ylabel(r'$f$')

        plt.legend(loc='best')
        fig.align_labels()
        dt_obs = str(dt_obs).replace('.', ',')
        path = os.path.join(dir, 'problem3_timeseries_dt={}.png'.format(dt_obs))

        plt.savefig(path)

        plt.close()

    else:

        fig = plt.figure(tight_layout=True)
        ax2 = plt.subplot()

        ax2.plot(t, X[1, :], label=r'true value', color='red', linestyle='--')
        ax2.plot(t[1:], mean_post[1, :], label=r'Estimation', color='green')
        ax2.set_ylabel(r'$f$')

        plt.legend(loc='best')
        fig.align_labels()
        dt_obs = str(dt_obs).replace('.', ',')
        path = os.path.join(dir, 'problem3_estimation_dt={}.png'.format(dt_obs))

        plt.savefig(path)

        plt.close()


def plot_time_series(X, t, dir):
    fig = plt.figure()
    ax = plt.subplot()

    ax.set_ylabel(r'$u_t$')
    ax.plot(t, X[0])

    path = os.path.join(dir, 'problem3_timeseries.png')
    #plt.savefig(path)
    plt.show()
    plt.close()


def main(dir):
    a = 1
    f = 1
    sigma = np.array([0.5, 0])

    # observation coefficients
    g1 = 1
    g2 = 0

    # observation noise
    sigma0_1 = 0.01
    sigma0_2 = 0

    F = np.array([[-a, 1], [0, 0]])

    # initial conditions used in Kalman filter
    x0 = np.array([0, 0])

    # time array for time series
    t = np.linspace(0, 100, 12001)

    #time array for observations and forecast
    t_obs = np.linspace(0, 100, 401)
    obs_time_ratio = int((t_obs[1] - t_obs[0])/(t[1] - t[0]))
    dt_obs = t_obs[1] - t_obs[0]
    print((t_obs[1] - t_obs[0])/t[1]-t[0])
    print(obs_time_ratio)
    print(dt_obs)

    # generate time series
    X = gen_time_series(t, a, f, sigma)

    # run the data assimilation forecast
    prior_mean, post_mean, observations = kalman_forecast(t, obs_time_ratio, F, sigma, g1,
                                                                               g2, sigma0_1, sigma0_2, x0, X)
    plot_forecast(prior_mean, post_mean, observations, t_obs, X, t, dir, dt_obs, show_ts=True)

    #plot_time_series(X, t, dir)

if __name__=='__main__':
    OUT_DIR = os.path.join(os.getcwd(), 'pic')
    os.makedirs(OUT_DIR, exist_ok=True)
    main(OUT_DIR)