from Neuron_bp.neuron_layer import Neuron_layer
from Neuron_bp.neuron import Neuron

class Neuron_network:
    """
    Een functie die het netwerk kan aanmaken. Dat bestaat uit een of meerdere perceptronLayer
    die verveolgens uit een of meerdere perceptrons bestaat.
    """
    def __init__(self, layers: [[Neuron_layer]]):
        """
        Init waar de layers input meegegeven wordt. Er wordt een lege
        lijst aangemaakt voor het opslaan van de netwerk output.
        """
        self.layers = layers
        self.outputs = []
        self.loss = []
        self.MSE = 0

    def feed_forward(self, input: [float]):
        """
        Een functie die de activatie van de netwerk berekend. Er wordt hier rekening gemaakt met het
        feit dat de input van de volgnde laag de output van de vorige laag moet zijn.

        :param input: een lijst met de input voor de eerste laag
        :return: een int met de activatie van de netwerk tussen 0 en 1
        """
        self.outputs = [input]
        for i in self.layers:
            self.outputs.append(i.predict_layer(self.outputs[-1]))
        self.outputs = self.outputs[-1]
        return self.outputs

    def train(self, inputs, target, epochs, learning_rate=0.1):
        for epoch in range(epochs):
            layersRange, inputsRange = range(len(self.layers)), range(len(inputs))
            # for x, index in zip(inputsRange, layersRange):
            for input in inputsRange:
                for index in layersRange:
                    self.feed_forward(inputs[input])
                    index = index * -1
                    if index != 0: #dus wel een hidden layer
                        self.layers[index-1].calculate_error(target[input], self.layers[index].weights,
                                                             self.layers[index].errors)
                    # geen hidden layer
                    else: self.layers[index-1].calculate_error(target[input], [], [])
            self.update_netwerk(inputs, target)
        self.calculate_total_loss()

    def update_netwerk(self, inputs, target):
        for x in range(len(inputs)):
            for index in range(len(self.layers)):
                index = index * -1
                self.layers[index-1].change_weight_layer()
                self.layers[index-1].update_layer()
            self.calculate_loss(target[x])

    def calculate_loss(self, target, loss_sum= 0):
        for index in range(len(target)):
            loss_sum += (target[index] - self.outputs[index])**2
        self.loss.append(loss_sum/len(target))

    def calculate_total_loss(self):
        MSE = sum(self.loss) / len(self.loss)
        self.MSE = MSE
        self.loss = []
        return MSE

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.outputs}'

n1 = Neuron(weights=[24, 24], bias=-12)
n2 = Neuron(weights=[-12, -12], bias=18)
n3 = Neuron(weights=[12, 12], bias=-18)
layer1 = Neuron_layer([n1, n2, n3])

n4 = Neuron(weights=[12, 12, 0], bias=-18)
n5 = Neuron(weights=[12, 12, 0], bias=-18)
layer2 = Neuron_layer([n4, n5])

netwerk = Neuron_network(layers=[layer1, layer2])

inputs = [[0, 1], [1, 1], [1, 0], [0, 0]]
outputs = [[0, 1], [1, 0], [0, 1], [0, 0]]
for input, output in zip(inputs, outputs):
    antw_list = netwerk.feed_forward(input)
    # print(antw_list)
netwerk.train(inputs, outputs, 10)
