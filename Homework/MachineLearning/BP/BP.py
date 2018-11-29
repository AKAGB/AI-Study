import random
import math
import numpy as np
import pandas as pd

class NeuralNetwork:
    LEARNING_RATE = 0.5
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, output_layer_weights = None):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        self.hidden_layer = NeuronLayer(num_hidden)
        self.output_layer = NeuronLayer(num_outputs)

        if hidden_layer_weights is None:
            hidden_layer_weights = np.random.randn(num_hidden, num_inputs + 1)
        if output_layer_weights is None:
            output_layer_weights = np.random.randn(num_outputs, num_hidden + 1)

        self.init_h_w = hidden_layer_weights
        self.init_o_w = output_layer_weights

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)


    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        for i in range(self.num_hidden):
            self.hidden_layer.neurons[i].weights = hidden_layer_weights[i] 

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):    
        for i in range(self.num_outputs):
            self.output_layer.neurons[i].weights = output_layer_weights[i]

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_output = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_output)


    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        deltas = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            each_n = self.output_layer.neurons[i]
            deltas[i] = each_n.calculate_pd_error_wrt_total_net_input(training_outputs[i])

        pd_err_zj = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            weights = np.array([each_n.weights[i] for each_n in self.output_layer.neurons])
            pd_err_yj = np.dot(deltas, weights)
            pd_err_zj[i] = pd_err_yj * self.hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_input()

        for i in range(self.num_outputs):
            for j in range(self.num_hidden):
                pd_err_wij = self.output_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j) * deltas[i]
                self.output_layer.neurons[i].weights[j] -= pd_err_wij * NeuralNetwork.LEARNING_RATE
            
        for i in range(self.num_hidden):
            for j in range(self.num_inputs):
                pd_err_wij = self.hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j) * pd_err_zj[i]
                self.hidden_layer.neurons[i].weights[j] -= pd_err_wij * NeuralNetwork.LEARNING_RATE
                
    def calculate_total_error(self, training_sets):
        result = 0
        for example in training_sets:
            inputs, outputs = example[:-1], np.array([example[-1]])
            self.feed_forward(inputs)
            all_err = [self.output_layer.neurons[i].calculate_error(outputs[i]) for i in range(self.num_outputs)]
            result += sum(all_err)
        return result

    def saveNet(self):
        hidden_layer_weights = []
        for each_n in self.hidden_layer.neurons:
            hidden_layer_weights.append(each_n.weights)
        output_layer_weights = []
        for each_n in self.output_layer.neurons:
            output_layer_weights.append(each_n.weights)
        hidden_layer_weights = np.array(hidden_layer_weights)
        output_layer_weights = np.array(output_layer_weights)
        np.savetxt('hidden_layer.txt', hidden_layer_weights)
        np.savetxt('output_layer.txt', output_layer_weights)

    def loadNet(self):
        hidden_layer_weights = np.loadtxt('net/hidden_layer.txt')
        output_layer_weights = np.loadtxt('net/output_layer.txt')
        output_layer_weights = output_layer_weights.reshape(self.num_outputs, self.num_hidden+1)
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)


class NeuronLayer:
    def __init__(self, num_neurons):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron())

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])

    def feed_forward(self, inputs):
        outputs = [neuron.calculate_output(inputs) for neuron in self.neurons]
        return np.array(outputs)

    def get_outputs(self):
        outputs = [neuron.output for neuron in self.neurons]
        return np.array(outputs)

class Neuron:
    def __init__(self, weights=[]):
        self.weights = weights
        self.inputs = []
        self.total_net_input = 0
        self.output = 0

    def calculate_output(self, inputs):
        self.inputs = np.hstack((inputs, np.array(1)))
        self.calculate_total_net_input()
        self.output = self.squash(self.total_net_input)
        return self.output

    def calculate_total_net_input(self):
        self.total_net_input = np.dot(self.weights, self.inputs)

    def squash(self, total_net_input):
        ex = math.e**(-total_net_input)
        return 1 / (1 + ex)

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        pd_err_output = self.calculate_pd_error_wrt_output(target_output)
        pd_output_total = self.calculate_pd_total_net_input_wrt_input()
        return pd_err_output * pd_output_total

    def calculate_error(self, target_output):
        return (self.output - target_output)**2

    def calculate_pd_error_wrt_output(self, target_output):
        return  self.output - target_output

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)
    
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

def preProcess(filepath):
    Data = pd.read_csv(filepath, header=None, sep='\s+', dtype=np.object)
    Data = Data.replace('?', '-1')
    Data = pd.DataFrame(Data, dtype=np.float)
    Data[2] /= 1000000
    Data[3] /= 100
    Data[4] /= 100
    Data[5] /= 100
    Data[18] /= 100
    Data[19] /= 100
    Data[24] /= 10000
    Data[27] -= 1
    return Data

if __name__ == "__main__":
    train_sets = preProcess('horse-colic.data')
    test_sets = preProcess('horse-colic.test')
    train_length = len(train_sets)
    test_length = len(test_sets)
    hidden_layer_weights = np.loadtxt('good_init/5/1/hidden_weight.txt')
    output_layer_weights = np.loadtxt('good_init/5/1/output_weight.txt')
    output_layer_weights = output_layer_weights.reshape(1, 6)
    # hidden_layer_weights = None
    # output_layer_weights = None
    nn = NeuralNetwork(27, 5, 1, hidden_layer_weights=hidden_layer_weights, output_layer_weights=output_layer_weights)
    
    # np.savetxt('hidden_weight.txt', nn.init_h_w)
    # np.savetxt('output_weight.txt', nn.init_o_w)

    # train
    for k in range(20):
        for i in range(10):
            for j in range(train_length):
                ex = np.array(train_sets.iloc[j])
                # print(ex)
                nn.train(np.array(ex[:27]), [ex[27]])
        err = nn.calculate_total_error(np.array(train_sets))
        err_test = nn.calculate_total_error(np.array(test_sets))
        NeuralNetwork.LEARNING_RATE *= 0.8
        print('Train%d: \t%.9f' % (k+1, err))
        if (k+1) % 3 == 0:
            acc = 0
            for i in range(test_length):
                ex = np.array(test_sets.iloc[i])
                inputs, output = ex[:27], ex[27]
                predict = round(nn.feed_forward(inputs)[0])
                # print(abs(predict - output))
                if abs(predict - output) <= 0.5:
                    acc += 1
            # print(acc)
            print('Accuracy rate:', acc / test_length)
    
    print('Finally:')
    acc = 0
    for i in range(test_length):
        ex = np.array(test_sets.iloc[i])
        inputs, output = ex[:27], ex[27]
        predict = round(nn.feed_forward(inputs)[0])
        # print(abs(predict - output))
        if abs(predict - output) <= 0.5:
            acc += 1
    # print(acc)
    print('Accuracy rate:', acc / test_length)

    nn.saveNet()

