import sys
import os, struct
from array import array as pyarray
from numpy import array, int8, uint8, zeros
import sys
import nnupdater
import numpy as np
import nnet

# 1. set hyper-parameters
class NNParam:
    def __init__(self):
        # network type
        self.net_type = 'mlp2'
        self.node_type = 'sigmoid'
        self.out_type = 'softmax'
        # ------------------------------------
        self.eta = 0.01 # learning rate
        self.mdecay = 0.1 # momentum decay
        self.wd = 0.0 # weight decay,
        self.num_burn = 1000 # number of burn-in round, start averaging after num_burn round
        self.batch_size = 500 # mini-batch size used in training
        self.init_sigma = 0.001  # initial gaussian standard deviation used in weight init
        self.seed = 123  # random number seed
        self.updater = 'sgd' # weight updating method
        self.temp = 1.0 # temperature: temp=0 means no noise during sampling(MAP inference)
        self.start_sample = 1  # start sampling weight after this round
        # ----------------------------------
        self.hyperupdater = 'none' # hyper parameter sampling
        self.start_hsample = 1  # when to start sample hyper parameter
        # Gamma(alpha, beta) prior on regularizer
        self.hyper_alpha = 1.0
        self.hyper_beta = 1.0
        # sample hyper parameter each gap_hsample over training data
        self.gap_hsample = 1
        # -----------------------------------
        # adaptive learning rate and momentum. By default, no need to set these settings
        self.delta_decay = 0.0
        self.start_decay = None
        self.alpha_decay = 1.0
        self.decay_momentum = 0
        self.init_eta = None
        self.init_mdecay = None
        # -----------------------
        # following things are not set by user
        self.wsample = 1.0 # sample weight
        self.rcounter = 0 # round counter

        # how many steps before resample hyper parameter

    def gap_hcounter(self):
        return int(self.gap_hsample * self.num_train / self.batch_size)

    # adapt learning rate and momentum, if necessary
    def adapt_decay(self, rcounter):
        # adapt decay ratio
        if self.init_eta == None:
            self.init_eta = self.eta
            self.init_mdecay = self.mdecay
        self.wsample = 1.0
        if self.start_decay == None:
            return

        d_eta = 1.0 * np.power(1.0 + max(rcounter - self.start_decay, 0) * self.alpha_decay, - self.delta_decay)
        assert d_eta - 1.0 < 1e-6 and d_eta > 0.0

        if self.decay_momentum != 0:
            d_mom = np.sqrt(d_eta)
            self.wsample = d_mom
        else:
            d_mom = 1.0
            self.wsample = d_eta

        self.eta = d_eta * self.init_eta
        self.mdecay = d_mom * self.init_mdecay

    # set current round
    def set_round(self, rcounter):
        self.rcounter = rcounter
        self.adapt_decay(rcounter)
        if self.updater == 'sgld':
            assert np.abs(self.mdecay - 1.0) < 1e-6

    # get noise level for sampler
    def get_sigma(self):
        if self.mdecay - 1.0 > -1e-5 or self.updater == 'sgld':
            scale = self.eta / self.num_train
        else:
            scale = self.eta * self.mdecay / self.num_train
        return np.sqrt(2.0 * self.temp * scale)

        # whether we need to sample weight now

    def need_sample(self):
        if self.start_sample == None:
            return False
        else:
            return self.rcounter >= self.start_sample

    # whether we need to sample hyper parameter now
    def need_hsample(self):
        if self.start_hsample == None:
            return False
        else:
            return self.rcounter >= self.start_hsample

    # whether the network need to provide second moment of gradient
    def rec_gsqr(self):
        return True

def cfg_param():
    param = NNParam()
    param.init_sigma = 0.01
    param.input_size = 28 * 28
    param.num_class = 10
    param.eta = 0.1
    param.mdecay = 0.1
    param.wd = 0.00002
    param.batch_size = 500
    param.num_round = 800
    param.num_hidden = 100
    param.path_data = 'dataset'
    param.net_type = 'mlp2'
    param.updater = 'adam'#'sghmc_adam'#'sghmc' #'sgld' #'sgld' #'sghmc' #'sgd'#'sghmc' #
    param.hyperupdater = 'gibbs-sep'
    param.num_burn = 50
    param.mdecay = 0.01
    return param


# load MNIST dataset
def create_batch( images, labels, nbatch, doshuffle=False, scale=1.0 ):
    if labels.shape[0] % nbatch != 0:
        print( '%d data will be dropped during batching' % (labels.shape[0] % nbatch))
    nsize = labels.shape[0] // nbatch * nbatch
    assert images.shape[0] == labels.shape[0]

    if doshuffle:
        ind = list(range( images.shape[0] ))
        np.random.shuffle( ind )
        images, labels = images[ind], labels[ind]

    images = images[ 0 : nsize ];
    labels = labels[ 0 : nsize ];
    xdata = np.float32( images.reshape( labels.shape[0]//nbatch, nbatch, images[0].size ) ) * scale
    ylabel = labels.reshape( labels.shape[0]//nbatch, nbatch )
    return xdata, ylabel


def load(digits, dataset="training", path="."):
    """
    Loads MNIST files into 3D numpy arrays
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

# Networks
class NNFactory:
    def __init__(self, param):
        self.param = param

    def create_updater(self, w, g_w, sg_w):
        if self.param.updater == 'sgd':
            return nnupdater.SGDUpdater(w, g_w, self.param)
        elif self.param.updater == 'sghmc' or self.param.updater == 'sgld':
            if self.param.updater == 'sgld':
                self.param.mdecay = 1.0
            return nnupdater.SGHMCUpdater(w, g_w, self.param)
        elif self.param.updater == 'nag':
            return nnupdater.NAGUpdater(w, g_w, self.param)
        elif self.param.updater == 'adam':
            return nnupdater.ADAMUpdater(w, g_w, sg_w, self.param)
        elif self.param.updater == 'sghmc_adam':
            return nnupdater.SGHMC_ADAM_Updater(w, g_w, sg_w, self.param)
        else:
            raise('NNConfig', 'unknown updater')

    def create_hyperupdater(self, updaterlist):
        if self.param.hyperupdater == 'none':
            return []
        elif self.param.hyperupdater == 'gibbs-joint':
            return [nnupdater.HyperUpdater(self.param, updaterlist)]
        elif self.param.hyperupdater == 'gibbs-sep':
            return [nnupdater.HyperUpdater(self.param, [u]) for u in updaterlist]
        else:
            raise('NNConfig', 'unknown hyperupdater')

    def create_olabel(self):
        param = self.param
        if param.out_type == 'softmax':
            return np.zeros((param.batch_size), 'int8')
        else:
            return np.zeros((param.batch_size), 'float32')

    def create_outlayer(self, o_node, o_label):
        param = self.param
        if param.out_type == 'softmax':
            return nnet.SoftmaxLayer(o_node, o_label)
        elif param.out_type == 'linear':
            return nnet.RegressionLayer(o_node, o_label, param)
        elif param.out_type == 'logistic':
            return nnet.RegressionLayer(o_node, o_label, param)
        else:
            raise('NNConfig', 'unknown out_type')

def softmax(param):
    factory = NNFactory(param)
    # setup network for softmax
    i_node = np.zeros((param.batch_size, param.input_size), 'float32')
    o_node = np.zeros((param.batch_size, param.num_class), 'float32')
    o_label = factory.create_olabel()

    nodes = [i_node, o_node]
    layers = [nnet.FullLayer(i_node, o_node, param.init_sigma, param.rec_gsqr())]

    layers += [factory.create_outlayer(o_node, o_label)]
    net = nnet.NNetwork(layers, nodes, o_label, factory)
    return net

def mlp2layer(param):
    factory = NNFactory(param)
    # setup network for 2 layer perceptron
    i_node = np.zeros((param.batch_size, param.input_size), 'float32')
    o_node = np.zeros((param.batch_size, param.num_class), 'float32')
    h1_node = np.zeros((param.batch_size, param.num_hidden), 'float32')
    h2_node = np.zeros_like(h1_node)
    o_label = factory.create_olabel()

    nodes = [i_node, h1_node, h2_node, o_node]
    layers = [nnet.FullLayer(i_node, h1_node, param.init_sigma, param.rec_gsqr())]
    layers += [nnet.ActiveLayer(h1_node, h2_node, param.node_type)]
    layers += [nnet.FullLayer(h2_node, o_node, param.init_sigma, param.rec_gsqr())]
    layers += [factory.create_outlayer(o_node, o_label)]

    net = nnet.NNetwork(layers, nodes, o_label, factory)
    return net

def mlp3layer(param):
    factory = NNFactory(param)
    # setup network for 2 layer perceptron
    i_node = np.zeros((param.batch_size, param.input_size), 'float32')
    o_node = np.zeros((param.batch_size, param.num_class), 'float32')
    h1_node = np.zeros((param.batch_size, param.num_hidden), 'float32')
    h2_node = np.zeros_like(h1_node)

    h1_node2 = np.zeros((param.batch_size, param.num_hidden2), 'float32')
    h2_node2 = np.zeros_like(h1_node2)

    o_label = factory.create_olabel()

    nodes = [i_node, h1_node, h2_node, h1_node2, h2_node2, o_node]

    layers = [nnet.FullLayer(i_node, h1_node, param.init_sigma, param.rec_gsqr())]
    layers += [nnet.ActiveLayer(h1_node, h2_node, param.node_type)]

    layers += [nnet.FullLayer(h2_node, h1_node2, param.init_sigma, param.rec_gsqr())]
    layers += [nnet.ActiveLayer(h1_node2, h2_node2, param.node_type)]

    layers += [nnet.FullLayer(h2_node2, o_node, param.init_sigma, param.rec_gsqr())]

    layers += [factory.create_outlayer(o_node, o_label)]

    net = nnet.NNetwork(layers, nodes, o_label, factory)
    return net

def create_net( param ):
    if param.net_type == 'mlp2':
        return mlp2layer( param )
    if param.net_type == 'mlp3':
        return mlp3layer( param )
    elif param.net_type == 'softmax':
        return softmax( param )
    else:
        raise('NNConfig', 'unknown net_type')

def run_exp(param):
    np.random.seed(param.seed)
    net = create_net(param)
    print('network configure end, start loading data ...')

    # load in data
    train_images, train_labels = load(range(10), 'training', param.path_data)
    test_images, test_labels = load(range(10), 'testing', param.path_data)

    # create a batch data, nbatch: batch size. doshuffle: True, shuffle the data. scale: 1.0/256 scale by this factor so all features are in [0,1]
    train_xdata, train_ylabel = create_batch(train_images, train_labels, param.batch_size, True, 1.0 / 256.0)
    test_xdata, test_ylabel = create_batch(test_images, test_labels, param.batch_size, True, 1.0 / 256.0)

    # split validation set
    ntrain = train_xdata.shape[0]
    nvalid = 10000
    assert nvalid % param.batch_size == 0
    nvalid = nvalid // param.batch_size
    valid_xdata, valid_ylabel = train_xdata[0:nvalid], train_ylabel[0:nvalid]
    train_xdata, train_ylabel = train_xdata[nvalid:ntrain], train_ylabel[nvalid:ntrain]

    # setup evaluator
    evals = []
    evals.append(nnet.NNEvaluator(net, train_xdata, train_ylabel, param, 'train'))
    evals.append(nnet.NNEvaluator(net, valid_xdata, valid_ylabel, param, 'valid'))
    evals.append(nnet.NNEvaluator(net, test_xdata, test_ylabel, param, 'test'))

    # set parameters
    param.num_train = train_ylabel.size
    print('loading end,%d train,%d valid,%d test, start update ...' %(train_ylabel.size, valid_ylabel.size, test_ylabel.size))

    errors, likelihoods = [], []
    for it in range(param.num_round):
        param.set_round(it)
        net.update_all(train_xdata, train_ylabel)
        sys.stderr.write('[%d]' % it)
        error, likelihood = [], []
        for ev in evals:
            err, llik = ev.eval(it, sys.stderr)
            error.append(err)
            likelihood.append(llik)
        errors.append(error)
        likelihoods.append(likelihood)
        sys.stderr.write('\n')
    print('all update end')
    errors = np.array(errors)
    print(errors[:, 0])
    print(errors[:, 1])
    print(errors[:, 2])
    np.save(param.updater, errors)


if __name__ == '__main__':
    # a = np.load('sghmc.npy')
    param = cfg_param()
    run_exp(param)