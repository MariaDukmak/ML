from Neuron.neuron_layer import Neuron_layer
from Neuron.neuron import Neuron

class Neuron_network:
    """
    Een functie die het netwerk kan aanmaken. Dat bestaat uit een of meerdere perceptronLayer
    die verveolgens uit een of meerdere perceptrons bestaat.
    """
    def __init__(self, layers:[[Neuron_layer]]):
        """
        Init waar de layers input meegegeven wordt. Er wordt een lege
        lijst aangemaakt voor het opslaan van de netwerk output.
        """
        self.layers = layers
        self.outputs = []

    def feed_forward(self, input:[float]):
        """
        Een functie die de activatie van de netwerk berekend. Er wordt hier rekening gemaakt met het
        feit dat de input van de volgnde laag de output van de vorige laag moet zijn.

        :param input: een lijst met de input voor de eerste laag
        :return: een int met de activatie van de netwerk tussen 0 en 1
        """
        self.outputs = [input]
        for i in self.layers:
            self.outputs.append(i.predict_layer(self.outputs[-1]))
        return self.outputs[-1]

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.outputs}'
#
orNeuron = Neuron(weights=[100, 100], bias=-50)
nandNeuron = Neuron(weights=[-80, -80], bias=100)
andNeuron = Neuron(weights=[80, 80], bias=-100)
firstLayer = Neuron_layer([orNeuron, nandNeuron, andNeuron])

andNeuron = Neuron(weights=[50, 50, -10], bias=-80)
extraNeuron = Neuron(weights=[-100, -100, 1000], bias=0)
secondLayer = Neuron_layer([andNeuron, extraNeuron])

halfAdderNetwork = Neuron_network([firstLayer, secondLayer])

inputs = [[0,1], [1,1], [1,0], [0,0]]
outputs = [[0, 1], [1, 0], [0, 1], [0, 0]]
for input, output in zip(inputs, outputs):
    print("input", input, halfAdderNetwork.feed_forward(input))
