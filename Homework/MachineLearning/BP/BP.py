import random
import math
import numpy as np
import pandas as pd

labels = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]

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
                self.output_layer.neurons[i].weights[j] -= NeuralNetwork.LEARNING_RATE * pd_err_wij


        for i in range(self.num_hidden):
            for j in range(self.num_inputs):
                pd_err_wij = self.hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j) * pd_err_zj[i]
                self.hidden_layer.neurons[i].weights[j] -= NeuralNetwork.LEARNING_RATE * pd_err_wij 

    def calculate_total_error(self, test_x, test_y):
        result = 0
        l = len(test_y)
        for i in range(l):
            
            inputs, outputs = test_x[i], labels[int(test_y[i])-1]
            # inputs, outputs = test_x[i], test_y[i]
            self.feed_forward(inputs)
            all_err = [self.output_layer.neurons[i].calculate_error(outputs[i]) for i in range(self.num_outputs)]
            result += sum(all_err)
        return result / l

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
        hidden_layer_weights = np.loadtxt('hidden_layer.txt')
        output_layer_weights = np.loadtxt('output_layer.txt')
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
        return 0.5 * (self.output - target_output)**2

    def calculate_pd_error_wrt_output(self, target_output):
        return  self.output - target_output

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)
        # return 1 - self.output**2
    
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

def preProcess(filepath):
    Data = pd.read_csv(filepath, header=None, sep='\s+', dtype=np.object)
    Data = Data.replace('?', '-1')
    Data = Data[Data[22] != '-1']
    Data = Data.reset_index(drop=True)
    Data = pd.DataFrame(Data, dtype=np.float)
    Data[2] /= 1000000
    Data[3] /= 100
    Data[4] /= 100
    Data[5] /= 100
    Data[18] /= 100
    Data[19] /= 100
    Data[24] /= 10000
    
    return Data

hidden_layer_weights = [[-2.47994546, -1.72878695, -4.70167989, -1.26332175, 7.66559532,
1.74543817, 0.82899178, 1.44391721, -4.2170834, 0.73136789,
-2.84525751, -0.5489623, -1.59265368, -1.47319714, 1.94006368,
-1.32619166, -3.9610622, 3.32860915, 3.32389767, 1.00907623,
1.44507713, -0.39202461, -6.04042209, 5.32888781, 0.54867852,
0.32861444, 2.44740317, 0.74419261],
[4.28151211, -1.72360323, 2.88816389, 1.98837274, -6.77756924,
4.35239887, 1.6293941, 2.3183484, -9.4852417, -6.76695101,
-1.88594888, 6.09585021, -1.79893637, 4.65500456, 2.43254806,
-1.37526359, 3.51241027, -4.12409032, -2.13661766, -1.79749007,
6.15917109, -5.88965477, 10.17071267, -0.14663357, 1.36626511,
-1.31419288, 1.73826709, 1.07220033],
[-0.42565379, 1.08077323, 5.53682016, 3.24621265, -3.43654701,
-2.99627512, -1.18866588, 1.67848722, -2.83942602, 2.63511049,
-2.8905248, 3.56790927, 0.78574986, 0.14769629, -5.28555796,
-0.15609974, -0.56601163, 6.14973798, -2.81728901, 2.30205876,
2.82141887, -0.15548144, 3.43711503, -2.1748885, 0.01858783,
-0.40674915, 5.26944495, -0.9621446, ],
[0.36621336, 0.14258277, -0.28550016, 0.41941569, -1.76134587,
-0.16540766, 0.6137982, -2.02383993, 0.96876868, -0.7674184,
2.74389118, 5.83244205, 1.86428335, 1.94630439, -3.36613854,
-1.40129608, -2.53698329, -1.28960526, 0.66981239, -1.79804658,
-1.84108127, 1.12611013, -0.11678514, -0.13069725, 0.44439874,
-0.90969348, 5.31959731, -0.13675369],
[-6.76327677, -0.88071882, 9.40836171, -0.70581635, -1.499864,
-2.08670718, 4.65184202, 7.09913677, -1.85534158, -0.02894694,
3.87650114, -2.6012522, -2.2976541, -3.15466105, -1.66705634,
0.91376745, 2.39784926, 0.03241444, -4.71098092, -6.93344944,
3.4455057, -5.52176212, -12.65069051, 4.42387866, -0.22119728,
-0.22811309, -3.65447742, 1.22490599]]

output_layer_weights = [[-3.43353398, 3.74995531, 4.54970726, -6.32790263, -3.71117364, 1.58595847],
[0.92768146, -3.97868024, -5.60058011, 2.62454381, 5.12627207, 1.39663096],
[2.80691169, -3.8362228, 4.37087495, -5.96201697, -2.93301269, 0.74001448]]

if __name__ == "__main__":
    train_sets = preProcess('horse-colic.data')
    test_sets = preProcess('horse-colic.test')

    train_length = len(train_sets)
    test_length = len(test_sets)

    # Random Initialize
    # hidden_layer_weights = None
    # output_layer_weights = None
    nn = NeuralNetwork(27, 5, 3, hidden_layer_weights=hidden_layer_weights, output_layer_weights=output_layer_weights)

    train_x = np.array(train_sets.drop([22], axis=1))
    train_y = np.array(train_sets[22])
    test_x = np.array(test_sets.drop([22], axis=1))
    test_y = np.array(test_sets[22])

    NeuralNetwork.LEARNING_RATE = 0.1

    # train
    for k in range(10):
        for i in range(10):
            for j in range(train_length):
                x, y = np.array(train_x[j]), labels[int(train_y[j])-1]
                nn.train(x, y)
        err = nn.calculate_total_error(train_x, train_y)
        
        NeuralNetwork.LEARNING_RATE *= 0.5

        print('Train%d Loss: \t%.9f' % (k+1, err))
    
    print('Test:')
    acc = 0
    for i in range(test_length):
        inputs, output = np.array(test_x[i]), test_y[i]
        predict = np.argmax(nn.feed_forward(inputs)) + 1
        # print('predict')
        # print(abs(predict - output))
        if abs(predict - output) <= 0.1:
            acc += 1
    # print(acc)
    print('Accuracy rate:', acc / test_length)

    nn.saveNet()