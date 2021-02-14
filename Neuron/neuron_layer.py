from Neuron.neuron import Neuron

class Neuron_layer:
    """
    Een class waar de lagen van de netwerk aangemaakt kunnen worden.
    """
    def __init__(self, perceptron=[[Neuron]]):
        """
        init waar de verwachte onderdelen van de laag gedefineerd worden.
        Een laag kan en of meerdere perceptrons bevaten. Er wordt ook een
        leege lijst aangeemkt waar de output van de laag wordt opgeslagen.
         """
        self.perceptron = perceptron

    def predict_layer(self, input: [float]):
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        return [i.predict(input) for i in self.perceptron]

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.perceptron}'
