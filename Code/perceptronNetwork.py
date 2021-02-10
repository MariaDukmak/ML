import Code.perceptronLayer as PerceptronLayer
import perceptron
import perceptronLayer


class PerceptronNetwork(object):
    """
    Een functie die het netwerk kan aanmaken. Dat bestaat uit een of meerdere perceptronLayer
    die verveolgens uit een of meerdere perceptrons bestaat.
    """
    def __init__(self, layers=[[PerceptronLayer]]):
        """
        Init waar de layers input meegegeven wordt. Er wordt een lege
        lijst aangemaakt voor het opslaan van de netwerk output.
        """
        self.layers = layers
        self.output_layers = []

    def feed_forward(self, input:[float]):
        """
        Een functie die de activatie van de netwerk berekend. Er wordt hier rekening gemaakt met het
        feit dat de input van de volgnde laag de output van de vorige laag moet zijn.
        :param input: een lijst met de input voor de eerste laag
        :return: een int met de activatie van de netwerk tussen 0 en 1
        """
        outputs = [input]
        for i in self.layers:
            outputs.append(i.predict_layer(outputs[-1]))
        return outputs[-1]

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.output_layers}'

#
# p1 = perceptron.Perceptron(weights=[1, 1], bias=0, threshold=1)
# p2 = perceptron.Perceptron(weights=[-1,-1], bias=0, threshold=-1.5)
# p3 = perceptron.Perceptron(weights=[1, 1], bias=0, threshold=2)
# layer1 = perceptronLayer.PerceptronLayer(perceptron=[p1, p2, p3])
#
# p4 = perceptron.Perceptron(weights=[1,1,0], bias=0, threshold=2)
# p5 = perceptron.Perceptron(weights=[0,0,1], bias=0, threshold=1)
#
# layer2 = perceptronLayer.PerceptronLayer(perceptron=[p4, p5])
#
# netwerk = PerceptronNetwork(layers=[layer1, layer2])
# print(netwerk.feed_forward([1,0]))
# print(netwerk.feed_forward([0,0]))
# print(netwerk.feed_forward([1,1])) #0,1

