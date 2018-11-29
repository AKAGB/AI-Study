import random
import math
import numpy as np
import pandas as pd

# Shorthand:
# "pd_" as a variable prefix means "partial derivative"
# "d_" as a variable prefix means "derivative"
# "_wrt_" is shorthand for "with respect to"
# "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively

class NeuralNetwork:
    LEARNING_RATE = 0.5
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, output_layer_weights = None):
    #Your Code Here
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
        #Your Code Here
        for i in range(self.num_hidden):
            self.hidden_layer.neurons[i].weights = hidden_layer_weights[i] 

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):    
        #Your Code Here
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
        #Your Code Here
        hidden_output = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_output)


    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas        
        #Your Code Here
        deltas = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            each_n = self.output_layer.neurons[i]
            deltas[i] = each_n.calculate_pd_error_wrt_total_net_input(training_outputs[i])

        # 2. Hidden neuron deltas        
        # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
        # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
        #Your Code Here
        # pd_err_yj = []
        pd_err_zj = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            weights = np.array([each_n.weights[i] for each_n in self.output_layer.neurons])
            pd_err_yj = np.dot(deltas, weights)
            pd_err_zj[i] = pd_err_yj * self.hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ             
        # Δw = α * ∂Eⱼ/∂wᵢ
        #Your Code Here        
        for i in range(self.num_outputs):
            for j in range(self.num_hidden):
                pd_err_wij = self.output_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j) * deltas[i]
                self.output_layer.neurons[i].weights[j] -= pd_err_wij * NeuralNetwork.LEARNING_RATE
            
        # 4. Update hidden neuron weights
        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ    
        # Δw = α * ∂Eⱼ/∂wᵢ
        #Your Code Here
        for i in range(self.num_hidden):
            for j in range(self.num_inputs):
                pd_err_wij = self.hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j) * pd_err_zj[i]
                self.hidden_layer.neurons[i].weights[j] -= pd_err_wij * NeuralNetwork.LEARNING_RATE
                
    def calculate_total_error(self, training_sets):
        #Your Code Here
        result = 0
        for example in training_sets:
            inputs, outputs = example[:-1], np.array([example[-1]])
            self.feed_forward(inputs)
            all_err = [self.output_layer.neurons[i].calculate_error(outputs[i]) for i in range(self.num_outputs)]
            result += sum(all_err)
        return result


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
        #Your Code Here
        self.inputs = np.hstack((inputs, np.array(1)))
        self.calculate_total_net_input()
        self.output = self.squash(self.total_net_input)
        return self.output

    def calculate_total_net_input(self):
        #Your Code Here
        self.total_net_input = np.dot(self.weights, self.inputs)

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
    #Your Code Here
        ex = math.e**(-total_net_input)
        return 1 / (1 + ex)

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
    #Your Code Here
        pd_err_output = self.calculate_pd_error_wrt_output(target_output)
        pd_output_total = self.calculate_pd_total_net_input_wrt_input()
        return pd_err_output * pd_output_total

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
    #Your Code Here
        return (self.output - target_output)**2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
    #Your Code Here
        return  self.output - target_output

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
    #Your Code Here
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
    #Your Code Here
        return self.inputs[index]

# An example:
# nn = NeuralNetwork(2, 2, 1, hidden_layer_weights=np.array([[0.15, 0.2], [0.25, 0.3]]), 
#             hidden_layer_bias=0.35, output_layer_weights=np.array([[0.4, 0.45], [0.5, 0.55]]), output_layer_bias=0.6)
# nn = NeuralNetwork(2, 5, 1, hidden_layer_weights=None, 
#             hidden_layer_bias=0.35, output_layer_weights=None, output_layer_bias=0.6)
# for i in range(10000):
#     nn.train([0.05, 0.1], [0.01])
#     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# nn.inspect()

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
    #hidden_layer_weights = np.loadtxt('good_init/5/1/hidden_weight.txt')
    #output_layer_weights = np.loadtxt('good_init/5/1/output_weight.txt')
    #output_layer_weights = output_layer_weights.reshape(1, 6)
    hidden_layer_weights = None
    output_layer_weights = None
    nn = NeuralNetwork(27, 5, 1, hidden_layer_weights=hidden_layer_weights, output_layer_weights=output_layer_weights)
    
    np.savetxt('hidden_weight.txt', nn.init_h_w)
    np.savetxt('output_weight.txt', nn.init_o_w)

    # train
    for k in range(100):
        for i in range(10):
            for j in range(train_length):
                ex = np.array(train_sets.iloc[j])
                # print(ex)
                nn.train(np.array(ex[:27]), [ex[27]])
        err = nn.calculate_total_error(np.array(train_sets))
        err_test = nn.calculate_total_error(np.array(test_sets))
        # NeuralNetwork.LEARNING_RATE *= 0.9
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
            print(acc)
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
    print(acc)
    print('Accuracy rate:', acc / test_length)

