import numpy as np
import sys
import math


# Full connected layer
# note: all memory are pre-allocated, always use a[:]= instead of a= in assignment
class FullLayer:
    def __init__(self, i_node, o_node, init_sigma, rec_gsqr=False):
        assert i_node.shape[0] == o_node.shape[0]
        self.rec_gsqr = rec_gsqr
        # node value
        self.i_node = i_node
        self.o_node = o_node
        # weight
        self.o2i_edge = np.float32(np.random.randn(i_node.shape[1], o_node.shape[1]) * init_sigma)
        self.o2i_bias = np.zeros(o_node.shape[1], 'float32')
        # gradient
        self.g_o2i_edge = np.zeros_like(self.o2i_edge)
        self.g_o2i_bias = np.zeros_like(self.o2i_bias)
        # gradient square
        self.sg_o2i_edge = np.zeros_like(self.o2i_edge)
        self.sg_o2i_bias = np.zeros_like(self.o2i_bias)
        if self.rec_gsqr:
            self.i_square = np.zeros_like(self.i_node)
            self.o_square = np.zeros_like(self.o_node)

    def forward(self, istrain=True):
        # forward prop, node value to o_node
        self.o_node[:] = np.dot(self.i_node, self.o2i_edge) + self.o2i_bias

    def backprop(self, passgrad=True):
        # backprop, gradient is stored in o_node
        # divide by batch size
        bscale = 1.0 / self.o_node.shape[0]
        #bscale = 1
        self.g_o2i_edge[:] = bscale * np.dot(self.i_node.T, self.o_node)
        self.g_o2i_bias[:] = np.mean(self.o_node, 0)

        # record second moment of gradient if needed
        if self.rec_gsqr:
            self.o_square[:] = np.square(self.o_node)
            self.i_square[:] = np.square(self.i_node)
            self.sg_o2i_edge[:] = bscale *np.square(np.dot(self.i_node.T, self.o_node))  #np.dot(self.i_square.T, self.o_square)
            self.sg_o2i_bias[:] = np.mean(self.o_square, 0)

        # backprop to i_node if necessary
        if passgrad:
            self.i_node[:] = np.dot(self.o_node, self.o2i_edge.T)

    def params(self):
        # return a reference list of parameters
        return [(self.o2i_edge, self.g_o2i_edge, self.sg_o2i_edge), (self.o2i_bias, self.g_o2i_bias, self.sg_o2i_bias)]


class ActiveLayer:
    def __init__(self, i_node, o_node, n_type='relu'):
        assert i_node.shape[0] == o_node.shape[0]
        # node value
        self.n_type = n_type
        self.i_node = i_node
        self.o_node = o_node

    def forward(self, istrain=True):
        # also get gradient ready in i node
        if self.n_type == 'relu':
            self.o_node[:] = np.maximum(self.i_node, 0.0)
            self.i_node[:] = np.sign(self.o_node)
        elif self.n_type == 'tanh':
            self.o_node[:] = np.tanh(self.i_node)
            self.i_node[:] = (1.0 - np.square(self.o_node))
        elif self.n_type == 'sigmoid':
            self.o_node[:] = 1.0 / (1.0 + np.exp(- self.i_node))
            self.i_node[:] = self.o_node * (1.0 - self.o_node)
        else:
            raise('NNConfig', 'unknown node_type')

    def backprop(self, passgrad=True):
        if passgrad:
            self.i_node[:] *= self.o_node;

    def params(self):
        return []


class SoftmaxLayer:
    def __init__(self, i_node, o_label):
        assert i_node.shape[0] == o_label.shape[0]
        assert len(o_label.shape) == 1
        self.i_node = i_node
        self.o_label = o_label

    def forward(self, istrain=True):
        nbatch = self.i_node.shape[0]
        self.i_node[:] = np.exp(self.i_node - np.max(self.i_node, 1).reshape(nbatch, 1))
        self.i_node[:] = self.i_node / np.sum(self.i_node, 1).reshape(nbatch, 1)

    def backprop(self, passgrad=True):
        if passgrad:
            nbatch = self.i_node.shape[0]
            for i in range(nbatch):
                self.i_node[i, self.o_label[i]] -= 1.0

    def params(self):
        return []


class RegressionLayer:
    def __init__(self, i_node, o_label, param):
        assert i_node.shape[0] == o_label.shape[0]
        assert i_node.shape[0] == o_label.size
        assert i_node.shape[1] == 1
        self.i_tmp = np.zeros_like(i_node)
        self.n_type = param.out_type
        self.i_node = i_node
        self.o_label = o_label
        self.param = param
        self.base_score = None

    def init_params(self):
        if self.base_score != None:
            return
        param = self.param
        self.scale = param.max_label - param.min_label;
        self.min_label = param.min_label
        self.base_score = (param.avg_label - param.min_label) / self.scale
        if self.n_type == 'logistic':
            self.base_score = - math.log(1.0 / self.base_score - 1.0);
        print('range=[%f,%f], base=%f' % (self.min_label, param.max_label, param.avg_label))

    def forward(self, istrain=True):
        self.init_params()
        nbatch = self.i_node.shape[0]
        self.i_node[:] += self.base_score
        if self.n_type == 'logistic':
            self.i_node[:] = 1.0 / (1.0 + np.exp(-self.i_node))

        self.i_tmp[:] = self.i_node[:]

        # transform to approperiate output
        self.i_node[:] = self.i_node * self.scale + self.min_label

    def backprop(self, passgrad=True):
        if passgrad:
            nbatch = self.i_node.shape[0]
            label = (self.o_label.reshape(nbatch, 1) - self.min_label) / self.scale
            self.i_node[:] = self.i_tmp[:] - label
            # print np.sum( np.sum( (label - self.i_tmp[:])**2 ) )

    def params(self):
        return []


class NNetwork:
    def __init__(self, layers, nodes, o_label, factory):
        self.nodes = nodes
        self.o_label = o_label
        self.i_node = nodes[0]
        self.o_node = nodes[-1]
        self.layers = layers
        self.weights = []
        self.updaters = []
        for l in layers:
            self.weights += l.params()
        for w, g_w, sg_w in self.weights:
            assert w.shape == g_w.shape and w.shape == sg_w.shape
            self.updaters.append(factory.create_updater(w, g_w, sg_w))

        self.updaters = factory.create_hyperupdater(self.updaters) + self.updaters

    def update(self, xdata, ylabel):
        self.i_node[:] = xdata
        for i in range(len(self.layers)):
            self.layers[i].forward(True)

        self.o_label[:] = ylabel
        for i in reversed(range(len(self.layers))):
            self.layers[i].backprop(i != 0)
        for u in self.updaters:
            u.update()

    def update_all(self, xdatas, ylabels):
        for i in range(xdatas.shape[0]):
            self.update(xdatas[i], ylabels[i])
        for u in self.updaters:
            u.print_info()

    def predict(self, xdata):
        self.i_node[:] = xdata
        for i in range(len(self.layers)):
            self.layers[i].forward(False)
        return self.o_node


# evaluator to evaluate results
class NNEvaluator:
    def __init__(self, nnet, xdatas, ylabels, param, prefix=''):
        self.nnet = nnet
        self.xdatas = xdatas
        self.ylabels = ylabels
        self.param = param
        self.prefix = prefix
        nbatch, nclass = nnet.o_node.shape
        assert xdatas.shape[0] == ylabels.shape[0]
        assert nbatch == xdatas.shape[1]
        assert nbatch == ylabels.shape[1]
        self.o_pred = np.zeros((xdatas.shape[0], nbatch, nclass), 'float32')
        self.rcounter = 0
        self.sum_wsample = 0.0

    def __get_alpha(self):
        if self.rcounter < self.param.num_burn:
            return 1.0
        else:
            self.sum_wsample += self.param.wsample
            return self.param.wsample / self.sum_wsample

    def eval(self, rcounter, fo):
        self.rcounter = rcounter
        alpha = self.__get_alpha()
        self.o_pred[:] *= (1.0 - alpha)
        sum_bad = 0.0
        sum_loglike = 0.0

        for i in range(self.xdatas.shape[0]):
            self.o_pred[i, :] += alpha * self.nnet.predict(self.xdatas[i])
            y_pred = np.argmax(self.o_pred[i, :], 1)
            sum_bad += np.sum(y_pred != self.ylabels[i, :])
            for j in range(self.xdatas.shape[1]):
                sum_loglike += np.log(self.o_pred[i, j, self.ylabels[i, j]])

        ninst = self.ylabels.size
        fo.write(' %s-err:%f %s-nlik:%f' % (self.prefix, sum_bad / ninst, self.prefix, -sum_loglike / ninst))
        return (sum_bad / ninst, -sum_loglike / ninst)