import Code.perceptronLayer as PerceptronLayer


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

    def froward_feed(self, input:[float]):
        """
        Een functie die de activatie van de netwerk berekend. Er wordt hier rekening gemaakt met het
        feit dat de input van de volgnde laag de output van de vorige laag moet zijn.
        :param input: een lijst met de input voor de eerste laag
        :return: een int met de activatie van de netwerk tussen 0 en 1
        """
        next_layer_input = input
        for i in self.layers:
            self.output_layers.append(i.predict_layer(next_layer_input))
            next_layer_input = self.output_layers[-1]
        self.output_layers = next_layer_input
        return self.output_layers

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.output_layers}'


