import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal

def sghmc(U, gradU, m, dt, nstep, x, C, V):
## SGHMC using gradU, for nstep, starting at position x
    p = np.random.randn(x.shape) * np.sqrt(m)
    B = 0.5 * V * dt
    D = np.sqrt( 2 * (C-B) * dt )

    for i in range(1, nstep):
        p = p - gradU( x ) * dt  - p * C * dt  + np.random.randn(1)*D
        x = x + p/m * dt
    newx = x
    return newx

def HMC_noised(xstart, pstart, gradU, dt, nstep, niter, m):
    x = xstart
    p = pstart
    xs = np.zeros(nstep)
    ys = np.zeros(nstep)
    for i in range(nstep):
        for j in range(niter):
            p = p - gradU(x) * dt / 2
            x = x + p / m * dt
            p = p - gradU(x) * dt / 2
        xs[i] = x
        ys[i] = p
    return xs, ys

def HMC_noised_resample(xstart, pstart, gradU, dt, nstep, niter, m):
    x = xstart
    p = pstart
    xs = np.zeros(nstep)
    ys = np.zeros(nstep)
    for i in range(nstep):
        p = np.random.randn(1) * np.sqrt(m)
        for j in range(niter):
            p = p - gradU(x) * dt / 2
            x = x + p / m * dt
            p = p - gradU(x) * dt / 2
        xs[i] = x
        ys[i] = p
    return xs, ys


def HMC_noised_friction(xstart, pstart, gradU, dt, nstep, niter, m, sigma, C):
    x = xstart
    p = pstart
    xs = np.zeros(nstep)
    ys = np.zeros(nstep)
    Bhat = 0.5 * (sigma*sigma) * dt
    D = np.sqrt(2 * (C - Bhat) * dt)
    for i in range(nstep):
        #p = np.random.randn(1) * np.sqrt(m)
        for j in range(niter):
            p = p - gradU(x) * dt - p*C*dt/m +np.random.randn(1)*D
            x = x + p / m * dt
        xs[i] = x
        ys[i] = p
    return xs, ys

def perfect_grad(xstart, pstart, gradU, dt, nstep, niter, m):
    x = xstart
    p = pstart
    xs = np.zeros(nstep)
    ys = np.zeros(nstep)
    for i in range(nstep):
        for j in range(niter):
            p = p - gradU(x) * dt / 2
            x = x + p / m * dt
            p = p - gradU(x) * dt / 2
        xs[i] = x
        ys[i] = p
    return xs, ys

def HMC_noised_friction_momentum(xstart, pstart, gradU, dt, nstep, niter, m, sigma, C):
    x = xstart
    p = pstart
    xs = np.zeros(nstep)
    ys = np.zeros(nstep)
    Bhat = 0.5 * (sigma*sigma) * dt
    D = np.sqrt(2 * (C - Bhat) * dt)
    momentum = 0
    for i in range(nstep):
        #p = np.random.randn(1) * np.sqrt(m)
        for j in range(niter):
            momentum = 0.2 * momentum + 0.8* gradU(x)
            p = p - momentum * dt - p*C*dt/m +np.random.randn(1)*D
            x = x + p / m * dt
        xs[i] = x
        ys[i] = p
    return xs, ys

def figure2():
    # U = 1/2 \theta^2
    m, C, dt = 1, 3, 0.1
    fgname = 'figure/trace'
    nstep, niter = 500, 50
    sigma = 0.5

    def gradUPerfect(x):
        return x

    def gradU(x):
        return x + np.random.randn(1) * sigma
    xstart, pstart = 1, 0

    x1,y1 = HMC_noised(xstart, pstart, gradU, dt, nstep, niter, m)
    x2,y2 = HMC_noised_resample(xstart, pstart, gradU, dt, nstep, niter, m)
    x3,y3 = HMC_noised_friction(xstart, pstart, gradU, dt, nstep, niter, m, sigma, C)
    x4,y4 = perfect_grad(xstart, pstart, gradUPerfect, dt, nstep, niter, m)
    x5, y5 = HMC_noised_friction_momentum(xstart, pstart, gradU, dt, nstep, niter, m, sigma, C)

    burn_in = 300
    plt.scatter(x1[burn_in :], y1[burn_in :], c='g', label='Noisy HMC')
    plt.scatter(x2[burn_in :], y2[burn_in :], c='r', label='Noisy HMC(resample)')
    plt.scatter(x3[burn_in :], y3[burn_in :], c='m', label='HMC with friction')
    plt.scatter(x4[burn_in :], y4[burn_in :], c='y', label='HMC')
    plt.scatter(x5[burn_in :], y5[burn_in :], c='c', label='Momentum')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.xlim(xmax=9, xmin=0)
    # plt.ylim(ymax=9, ymin=0)
    plt.legend()
    # plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)
    plt.show()

#################################################3

def figure3():
    V = 1
    # covariance matrix
    rho = 0.9
    covS = np.array([[1, rho], [rho, 1]])
    invS = np.linalg.inv(covS)
    x = np.array([[0],[0]])

    # this is highest value tried so far for SGLD that does not diverge
    etaSGLD = 0.05
    etaSGHMC = 0.05
    alpha = 0.035
    L = 50  # number of steps
    def probUMap(X,Y):
        return np.exp(- 0.5 * (X * X * invS[0, 0] + 2 * X * Y * invS[0, 1] + Y * Y * invS[1, 1])) / (
                2 * math.pi * np.sqrt(np.abs(np.linalg.det(covS))))

    def funcU(X):
        return 0.5 * np.matmul(np.matmul(np.transpose(x), invS), x)

    def gradUTrue(x):
        return np.matmul(invS, x)

    def gradUNoise(x):
        return np.matmul(invS, x) + np.random.randn(2, 1)

    [XX, YY] = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2));
    ZZ = probUMap(XX, YY)
    plt.contour(XX, YY, ZZ)

    def sgld(gradU, eta, L, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > 1:
            raise('too big eta')
            exit(1)
        sigma = np.sqrt(2 * eta * (1 - beta))

        for t in range(L):
            dx = - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + dx
            data[:, t] = x.reshape(-1)

        return data

    def sghmc(gradU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        a = gradU(x)
        for t in range(L):
            p = p * momentum - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + p
            data[:, t] = x.reshape(-1)

        return data

    dsgld = sgld(gradUNoise, etaSGLD, L, x, V)
    dsghmc = sghmc(gradUNoise, etaSGHMC, L, alpha, x, V)
    plt.scatter(dsgld[0,:], dsgld[1,:], c='b', label='sgld')
    plt.scatter(dsghmc[0,:], dsghmc[1,:], c='r', marker='x', label='sghmc')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    # legend([h1 h2], {'SGLD', 'SGHMC'});
    # axis([-2.1 3 - 2.1 3]);
    # len = 4;
    # set(gcf, 'PaperPosition', [0 0 len len / 8.0 * 6.5])
    # set(gcf, 'PaperSize', [len len / 8.0 * 6.5])

def figure3_mod(seed):
    seed = 1
    V = 1
    # covariance matrix
    rho = 0.9
    covS = np.array([[1, rho], [rho, 1]])
    invS = np.linalg.inv(covS)
    x = np.array([[0.0],[0.0]])

    # this is highest value tried so far for SGLD that does not diverge
    etaSGLD = 0.05
    etaSGHMC = 0.05
    alpha = 0.035
    L = 50  # number of steps
    def probUMap(X,Y):
        return np.exp(- 0.5 * (X * X * invS[0, 0] + 2 * X * Y * invS[0, 1] + Y * Y * invS[1, 1])) / (
                2 * math.pi * np.sqrt(np.abs(np.linalg.det(covS))))

    def funcU(X):
        return 0.5 * np.matmul(np.matmul(np.transpose(x), invS), x)

    def gradUTrue(x):
        return np.matmul(invS, x)

    def gradUNoise(x):
        return np.matmul(invS, x) + np.random.randn(2, 1)

    def HessianUNoise(x):
        return invS #+ np.random.randn(2, 1)

    [XX, YY] = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2));
    ZZ = probUMap(XX, YY)
    plt.contour(XX, YY, ZZ)

    def sgld(gradU, eta, L, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > 1:
            raise('too big eta')
            exit(1)
        sigma = np.sqrt(2 * eta * (1 - beta))

        for t in range(L):
            dx = - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + dx
            data[:, t] = x.reshape(-1)

        return data

    def sghmc(gradU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        a = gradU(x)
        for t in range(L):
            p = p * momentum - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + p
            data[:, t] = x.reshape(-1)

        return data

    def sghmc_mod(gradU,HessU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        invH = np.linalg.inv(HessU(x))
        for t in range(L):
            p = p * momentum - np.matmul(invH,gradU(x)) * eta + np.random.randn(2, 1) * sigma
            x = x + p
            data[:, t] = x.reshape(-1)

        return data

    def sghmc_adam(gradU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        # v = 0.0
        # for t in range(L):
        #     rectify_1 = 1.0 / (1.0 - (1.0 - 0.1) ** (t + 1))
        #     rectify_2 = 1.0 / (1.0 - (1.0 - 0.99) ** (t + 1))
        #
        #     v = 0.9 * v + 0.1 * np.square(gradU(x))
        #     p = momentum * p + (-np.sqrt(v * rectify_2) / rectify_1) * (gradU(x)) + np.random.randn(2, 1) * \
        #         sigma * np.sqrt(v * rectify_2) / rectify_1 / eta
        #     x += p * eta * rectify_1 / (np.sqrt(v * rectify_2) + 1e-8)
        #     data[:, t] = x.reshape(-1)

        v = np.square(gradU(x))
        for t in range(L):
            # v = (1-0.01)*v + 0.01* np.square(gradU(x))
            # v = 0.9*v+0.1*np.square(gradU(x))
            # p = p * momentum/(v+1e-8) - (gradU(x)/(np.sqrt(v)+1e-8)) * eta + np.random.randn(2, 1) * sigma
            # x = x+ p
            v = 0.9 * v + 0.1 * np.square(gradU(x))
            p = p * momentum - (gradU(x) * np.sqrt(v))  + np.random.randn(2, 1) * sigma * np.sqrt(v)/eta
            x = x + p * eta/(np.sqrt(v)+1e-8)
            data[:, t] = x.reshape(-1)

        return data

    np.random.seed(seed)
    dsgld = sgld(gradUNoise, etaSGLD, L, x, V)
    np.random.seed(seed)
    dsghmc = sghmc(gradUNoise, etaSGHMC, L, alpha, x, V)
    # np.random.seed(seed)
    # dsghmc_mod = sghmc_mod(gradUNoise, HessianUNoise, etaSGHMC, L, alpha, x, V)
    np.random.seed(seed)
    dsghmc_adam = sghmc_adam(gradUNoise, etaSGHMC, L, alpha, x, V)
    burn_in = 1
    plt.scatter(dsgld[0,burn_in :], dsgld[1,burn_in :], c='b', label='sgld')
    # plt.scatter(dsghmc_mod[0, burn_in :], dsghmc_mod[1, burn_in :], c='c', marker='s', label='sghmc_mod')

    plt.scatter(dsghmc[0, burn_in:], dsghmc[1, burn_in:], c='r', marker='x', label='sghmc')
    plt.scatter(dsghmc_adam[0, burn_in:], dsghmc_adam[1, burn_in:], c='g', marker='^', label='sghmc_adam')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    # legend([h1 h2], {'SGLD', 'SGHMC'});
    # axis([-2.1 3 - 2.1 3]);
    # len = 4;
    # set(gcf, 'PaperPosition', [0 0 len len / 8.0 * 6.5])
    # set(gcf, 'PaperSize', [len len / 8.0 * 6.5])

def figure3a_mod(seed):
    V = 1
    # covariance matrix
    rho = 0.9
    covS = np.array([[1, rho], [rho, 1]])
    invS = np.linalg.inv(covS)
    x = np.array([[0.0], [0.0]])

    # this is highest value tried so far for SGLD that does not diverge
    etaSGLD = 0.18
    etaSGHMC = 0.1
    alpha = 0.05
    L = 5000000 # number of steps
    nset = 5

    def gradUNoise(x):
        return np.matmul(invS, x) + np.random.randn(2, 1)

    def sgld(gradU, eta, L, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > 1:
            raise ('too big eta')
            exit(1)
        sigma = np.sqrt(2 * eta * (1 - beta))

        for t in range(L):
            dx = - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + dx
            data[:, t] = x.reshape(-1)

        return data

    def sghmc(gradU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        a = gradU(x)
        for t in range(L):
            p = p * momentum - gradU(x) * eta + np.random.randn(2, 1) * sigma
            x = x + p
            data[:, t] = x.reshape(-1)

        return data

    def sghmc_adam(gradU, eta, L, alpha, x, V):
        m = len(x)
        data = np.zeros((m, L))
        beta = V * eta * 0.5
        if beta > alpha:
            raise ('too big eta')
            exit(1)

        sigma = np.sqrt(2 * eta * (alpha - beta))
        p = np.random.randn(m, 1) * np.sqrt(eta)
        momentum = 1 - alpha

        # v = np.square(gradU(x))
        # for t in range(L):
        #     v = 0.995 * v + 0.005 * np.square(gradU(x))
        #     p = p * momentum - (gradU(x) * np.sqrt(v)) + np.random.randn(2, 1) * sigma * np.sqrt(v) / eta
        #     x = x + p * eta / (np.sqrt(v) + 1e-8)
        #     data[:, t] = x.reshape(-1)
        v = 0.0
        for t in range(L):
            rectify_1 = 1.0 / (1.0 - (1.0 - 0.1) ** (t+1))
            rectify_2 = 1.0 / (1.0 - (1.0 - 0.99) ** (t+1))

            v = 0.992 * v + 0.008 * np.square(gradU(x))
            p = momentum*p + (-np.sqrt(v * rectify_2) / rectify_1) * (gradU(x)) + np.random.randn(2, 1) * \
                sigma *np.sqrt(v * rectify_2) / rectify_1 / eta
            x+= p * eta* rectify_1 / (np.sqrt(v* rectify_2) + 1e-8)

            # v = 0.995 * v + 0.005 * np.square(gradU(x))
            # p = p * momentum - (gradU(x) * np.sqrt(v)) + np.random.randn(2, 1) * sigma * np.sqrt(v) / eta
            # x = x + p * eta / (np.sqrt(v) + 1e-8)

            data[:, t] = x.reshape(-1)

        return data

    def xcorr(x):
        corr = signal.correlate(x,x, mode='full')
        lags = signal.correlation_lags(len(x), len(x), mode='full')
        return corr, lags

    def aucTime(data, dvar):
       # auto correlation time, the calculation method comes  from Hoffman et.al.No - U - Turn sampeler
       m, L = data.shape
       tau = []

       for i in range(m):
           acorr, lags = xcorr(data[i,:])
           acorr /= dvar
           res = 1
           for j in range(L):
               rpho = 0.5 * (acorr[L + j] + acorr[L - j])
               if rpho < 0.05:
                   break
               res = res + 2 * (1 - j / L)
           tau.append(res)
       return tau


    covESGLD,meanESGLD,SGLDeta,SGLDauc = [],[],[],[]
    np.random.seed(seed)
    for i in range(1,nset+1):
        eta = etaSGLD * (0.8**(i - 1))
        dsgld = sgld(gradUNoise, eta, L, x, V)
        covESGLD.append(np.matmul(dsgld, np.transpose(dsgld)) / L)
        meanESGLD.append(np.mean(dsgld, 1, keepdims=True))
        SGLDeta.append(eta)
        SGLDauc.append(np.mean(aucTime(dsgld, 1)))

    covESGHMC, meanESGHMC, SGHMCDeta, SGHMCDauc = [], [], [], []
    np.random.seed(seed)
    for i in range(1,nset+1):
        dscale =  (0.6 ** (i - 1))
        eta = etaSGHMC * dscale * dscale
        dsghmc = sghmc(gradUNoise, eta, L, alpha*dscale, x, V)
        covESGHMC.append(np.matmul(dsghmc, np.transpose(dsghmc)) / L)
        meanESGHMC.append(np.mean(dsghmc, 1, keepdims=True))
        SGHMCDeta.append(eta)
        SGHMCDauc.append(np.mean(aucTime(dsghmc, 1)))

    covESGHMC_A, meanESGHMC_A, SGHMCDeta_A, SGHMCDauc_A = [], [], [], []
    np.random.seed(seed)
    for i in range(1, nset + 1):
        dscale = (0.6 ** (i - 1))
        eta = etaSGHMC * dscale * dscale
        dsghmc_a = sghmc_adam(gradUNoise, eta, L, alpha * dscale, x, V)
        covESGHMC_A.append(np.matmul(dsghmc_a, np.transpose(dsghmc_a)) / L)
        meanESGHMC_A.append(np.mean(dsghmc_a, 1, keepdims=True))
        SGHMCDeta_A.append(eta)
        SGHMCDauc_A.append(np.mean(aucTime(dsghmc_a, 1)))

    SGLDCovErr, SGHMCCovErr, ASGHMCCovErr = [], [], []
    for i in range(nset):
        SGLDCovErr.append( np.sum(np.abs(covESGLD[i] - np.matmul(meanESGLD[i], np.transpose(meanESGLD[i])) - covS ))/ 4)
        SGHMCCovErr.append(np.sum(np.abs(covESGHMC[i] - np.matmul(meanESGHMC[i], np.transpose(meanESGHMC[i]))- covS )) / 4)
        ASGHMCCovErr.append(
            np.sum(np.abs(covESGHMC_A[i] - np.matmul(meanESGHMC_A[i], np.transpose(meanESGHMC_A[i])) - covS)) / 4)

    plt.plot(np.array(SGLDauc), np.array(SGLDCovErr), 'b-x', label='sgld')
    plt.plot(np.array(SGHMCDauc), np.array(SGHMCCovErr), 'r-o', label='sghmc')
    plt.plot(np.array(SGHMCDauc_A), np.array(ASGHMCCovErr), 'g-s', label='sghmc_adam')

    plt.xlabel('Autocorrelation Time')
    plt.ylabel('Average Absolute Error of Sample Covariance')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    seed = 1234567
    np.random.seed(seed)
    #figure2()
    figure3_mod(seed)
    #figure3a_mod(seed)
