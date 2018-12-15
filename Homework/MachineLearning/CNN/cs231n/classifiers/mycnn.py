from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class CNN_old(object):
    """
    My CNN architecture:
    conv1 - relu - 2x2 max pool - conv2 - relu - 2x2 max pool 
    conv3 - relu - 2x2 max pool - fc1 - relu - fc2 - softmax
    """
    def __init__(self, input_dim=(3, 32, 32), conv1_N_filters=32, 
                conv1_filter_size=5, conv2_N_filters=32, conv2_filter_size=5,
                conv3_N_filters=64, conv3_filter_size=5,
                fc1_dim=64, fc2_dim=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float32):
        self.params = {}
        self.reg = reg 
        self.dtype = dtype

        # Suppose stride of conv = 1, stride of maxpool = 2
        C1, H1, W1 = input_dim                              # 3x32x32   -> 32x16x16
        C2, H2, W2 = conv1_N_filters, int(H1/2), int(W1/2)  # 32x16x16  -> 32x8x8
        C3, H3, W3 = conv2_N_filters, int(H2/2), int(W2/2)  # 32x8x8    -> 64*4*4
        fc1_input = conv3_N_filters * int(H3/2) * int(W3/2)
        fc2_input = fc1_dim

        """ Initialize the parameters """
        # Conv1
        self.params['W1'] = weight_scale * np.random.randn(conv1_N_filters, C1, 
                                    conv1_filter_size, conv1_filter_size)
        self.params['b1'] = np.zeros(conv1_N_filters)
        
        # Conv2 
        self.params['W2'] = weight_scale * np.random.randn(conv2_N_filters, C2,
                                    conv2_filter_size, conv2_filter_size)
        self.params['b2'] = np.zeros(conv2_N_filters)

        # Conv3
        self.params['W3'] = weight_scale * np.random.randn(conv3_N_filters, C3,
                                    conv3_filter_size, conv3_filter_size)
        self.params['b3'] = np.zeros(conv3_N_filters)

        # Fc1 
        self.params['W4'] = weight_scale * np.random.randn(fc1_input, fc1_dim)
        self.params['b4'] = np.zeros(fc1_dim)

        # Fc2 
        self.params['W5'] = weight_scale * np.random.randn(fc2_input, fc2_dim)
        self.params['b5'] = np.zeros(fc2_dim)


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # set conv1_param 
        filter1_size = W1.shape[2]
        conv1_param = {'stride': 1, 'pad': (filter1_size - 1) // 2}

        # set pool1_param
        pool1_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # set conv2_param
        filter2_size = W2.shape[2]
        conv2_param = {'stride': 1, 'pad': (filter2_size - 1) // 2}

        # set pool2_param
        pool2_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # set conv3_param
        filter3_size = W3.shape[2]
        conv3_param = {'stride': 1, 'pad': (filter3_size - 1) // 2}

        # set pool3_param
        pool3_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # Forward
        out, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv1_param, pool1_param)
        out, conv2_cache = conv_relu_pool_forward(out, W2, b2, conv2_param, pool2_param)
        out, conv3_cache = conv_relu_pool_forward(out, W3, b3, conv3_param, pool3_param)
        out, fc1_cache = affine_relu_forward(out, W4, b4)
        out, fc2_cache = affine_forward(out, W5, b5)
        scores = out

        if y is None:
            return scores 

        loss, grads = 0, {}

        # Backward
        data_loss, dout = softmax_loss(scores, y)

        # L2 Reg
        W_square_sum = 0
        for layer in range(5):
            Wi = self.params['W%d' % (layer + 1)]
            W_square_sum += (np.sum(Wi**2))
        reg_loss = 0.5 * self.reg * W_square_sum
        loss = data_loss + reg_loss

        # Calculate grandients
        dout, dW5, db5 = affine_backward(dout, fc2_cache)
        dout, dW4, db4 = affine_relu_backward(dout, fc1_cache)
        dout, dW3, db3 = conv_relu_pool_backward(dout, conv3_cache)
        dout, dW2, db2 = conv_relu_pool_backward(dout, conv2_cache)
        dout, dW1, db1 = conv_relu_pool_backward(dout, conv1_cache)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['W4'] = dW4
        grads['W5'] = dW5
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['b4'] = db4
        grads['b5'] = db5

        return loss, grads

class CNN(object):
    """
    My CNN architecture:
    conv1 - relu - 2x2 max pool - conv2 - 2x2 maxpool - relu - fc - softmax
    """
    def __init__(self, input_dim=(3, 32, 32), conv1_N_filters=16, 
                conv1_filter_size=5, conv2_N_filters=32, conv2_filter_size=5,
                weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg 
        self.dtype = dtype

        # Suppose stride of conv = 1, stride of maxpool = 2
        C1, H1, W1 = input_dim                              # 3x32x32   -> 32x16x16
        C2, H2, W2 = conv1_N_filters, int(H1/2), int(W1/2)  # 32x16x16  -> 32x8x8

        """ Initialize the parameters """
        # Conv1
        self.params['W1'] = weight_scale * np.random.randn(conv1_N_filters, C1, 
                                    conv1_filter_size, conv1_filter_size)
        self.params['b1'] = np.zeros(conv1_N_filters)
        
        # Conv2 
        self.params['W2'] = weight_scale * np.random.randn(conv2_N_filters, C2,
                                    conv2_filter_size, conv2_filter_size)
        self.params['b2'] = np.zeros(conv2_N_filters)

        # Fc
        self.params['W3'] = weight_scale * np.random.randn(conv2_N_filters*int(H2/2)*int(W2/2), 10)
        self.params['b3'] = np.zeros(10)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # set conv1_param 
        filter1_size = W1.shape[2]
        conv1_param = {'stride': 1, 'pad': (filter1_size - 1) // 2}

        # set pool1_param
        pool1_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # set conv2_param
        filter2_size = W2.shape[2]
        conv2_param = {'stride': 1, 'pad': (filter2_size - 1) // 2}

        # set pool2_param
        pool2_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # Forward
        out, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv1_param, pool1_param)
        out, conv2_cache = conv_relu_pool_forward(out, W2, b2, conv2_param, pool2_param)
        out, fc_cache = affine_forward(out, W3, b3)
        scores = out

        if y is None:
            return scores 

        loss, grads = 0, {}

        # Backward
        loss, dout = softmax_loss(scores, y)

        # L2 Reg
        # W_square_sum = 0
        # for layer in range(5):
        #     Wi = self.params['W%d' % (layer + 1)]
        #     W_square_sum += (np.sum(Wi**2))
        # reg_loss = 0.5 * self.reg * W_square_sum
        # loss = data_loss + reg_loss

        # Calculate grandients
        dout, dW3, db3 = affine_backward(dout, fc_cache)
        dout, dW2, db2 = conv_relu_pool_backward(dout, conv2_cache)
        dout, dW1, db1 = conv_relu_pool_backward(dout, conv1_cache)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        return loss, grads