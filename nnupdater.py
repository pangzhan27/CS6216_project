import numpy as np
import sys


# updater that performs SGD update given weight parameter
class SGDUpdater:
    def __init__(self, w, g_w, param):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like(w)

    def print_info(self):
        return

    def update(self):
        param = self.param
        self.m_w[:] *= (1.0 - param.mdecay)
        self.m_w[:] += (-param.eta) * (self.g_w ) #+ self.wd * self.w
        self.w[:] += self.m_w

class ADAMUpdater:
    def __init__(self, w, g_w, s_g_w, param):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like(w)
        self.s_g_w = s_g_w
        self.v = 0.0
        self.i = 0

    def print_info(self):
        return

    def update(self):
        param = self.param
        if np.sum(self.v) == 0:
            self.v = self.s_g_w
        else:
            self.v = 0.99*self.v + 0.01*self.s_g_w

        self.i +=1
        rectify_1 = 1.0/(1.0 - (1.0 - param.mdecay)**self.i)
        rectify_2 = 1.0 / (1.0 - (1.0 - 0.99) ** self.i)

        # Adam
        self.m_w[:] = (1.0 - param.mdecay)*self.m_w[:] +param.mdecay*(self.g_w) #+ self.wd * self.w
        self.w[:] -= param.eta * self.m_w* rectify_1 / (np.sqrt(self.v*rectify_2) + 1e-8)

# updater that performs update given weight parameter using SGHMC/SGLD
class SGHMC_ADAM_Updater:
    def __init__(self, w, g_w, s_g_w, param):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like(w)
        self.s_g_w = s_g_w
        self.v = 0.0
        self.i = 0

    def print_info(self):
        return

    def update(self):
        param = self.param
        if np.sum(self.v) == 0:
            self.v = self.s_g_w
        else:
            self.v = 0.99*self.v + 0.01*self.s_g_w

        self.i +=1
        rectify_1 = 1.0/(1.0 - (1.0 - param.mdecay)**self.i)
        rectify_2 = 1.0 / (1.0 - (1.0 - 0.99) ** self.i)

        self.m_w[:] *= (1.0 - param.mdecay)
        self.m_w[:] += (-np.sqrt(self.v*rectify_2)/rectify_1) * (self.g_w)  #+ self.wd * self.w
        if param.need_sample():
            self.m_w[:] += np.random.randn(self.w.size).reshape(self.w.shape) * param.get_sigma()* np.sqrt(self.v*rectify_2)/rectify_1/param.eta

        self.w[:] += param.eta * self.m_w * rectify_1 / (np.sqrt(self.v*rectify_2) + 1e-8)
        # SGD
        #self.w[:] -= param.eta * (self.g_w)
        # Adam
        # self.m_w[:] = 0.9*self.m_w[:] +0.1*(self.g_w)
        # #self.v = 0.99*self.v + 0.01*self.s_g_w
        # self.w[:] -= param.eta * self.m_w / (np.sqrt(self.v) + 1e-8)

        # self.m_w[:] = 0.9 * self.m_w[:] - 0.1 * (self.g_w)
        # self.v = 0.99 * self.v + 0.01 * self.s_g_w
        # self.w[:] += param.eta * self.m_w / (np.sqrt(self.v) + 1e-8)

class SGHMCUpdater:
    def __init__(self, w, g_w, param):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like(w)

    def print_info(self):
        return

    def update(self):
        param = self.param
        self.m_w[:] *= (1.0 - param.mdecay)
        self.m_w[:] += (-param.eta) * (self.g_w + self.wd * self.w)#
        if param.need_sample():
            self.m_w[:] += np.random.randn(self.w.size).reshape(self.w.shape) * param.get_sigma()
        self.w[:] += self.m_w

# updater that performs NAG(nestrov's momentum) update given weight parameter
class NAGUpdater:
    def __init__(self, w, g_w, param):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like(w)
        self.m_old = np.zeros_like(w)

    def print_info(self):
        return

    def update(self):
        param = self.param
        momentum = 1.0 - param.mdecay
        self.m_old[:] = self.m_w
        self.m_w[:] *= momentum
        self.m_w[:] += (-param.eta) * (self.g_w + self.wd * self.w)
        if param.need_sample():
            self.m_w[:] += np.random.randn(self.w.size).reshape(self.w.shape) * param.get_sigma()
        self.w[:] += (1.0 + momentum) * self.m_w - momentum * self.m_old


# Hyper Parameter Gibbs Gamma sampler for regularizer update
class HyperUpdater:
    def __init__(self, param, updaterlist):
        self.updaterlist = updaterlist
        self.param = param
        self.scounter = 0

    # update hyper parameters
    def update(self):
        param = self.param
        if not param.need_hsample():
            return

        self.scounter += 1
        if self.scounter % param.gap_hcounter() != 0:
            return
        else:
            self.scounter = 0

        sumsqr = sum(np.sum(u.w * u.w) for u in self.updaterlist)
        sumcnt = sum(u.w.size for u in self.updaterlist)
        alpha = param.hyper_alpha + 0.5 * sumcnt
        beta = param.hyper_beta + 0.5 * sumsqr

        if param.temp < 1e-6:
            # if we are doing MAP, take the mode, note: normally MAP adjust is not as well as MCMC
            plambda = max(alpha - 1.0, 0.0) / beta
        else:
            plambda = np.random.gamma(alpha, 1.0 / beta)

        # set new weight decay
        wd = plambda / param.num_train

        for u in self.updaterlist:
            u.wd = wd

        ss = ','.join(str(u.w.shape) for u in self.updaterlist)
        print('hyperupdate[%s]:plambda=%f,wd=%f' % (ss, plambda, wd))
        sys.stdout.flush()

    def print_info(self):
        return