from typing import List
from Neuron.neuron import Neuron


class Neuron_layer:
    """
    Een class waar de lagen van de netwerk aangemaakt kunnen worden.
    """
    def __init__(self, neurons: [[Neuron]]):
        """
        init waar de verwachte onderdelen van de laag gedefineerd worden.
        Een laag kan en of meerdere perceptrons bevaten. Er wordt ook een
        leege lijst aangeemkt waar de output van de laag wordt opgeslagen.
         """
        self.neurons = neurons
        self.errors = []

    def predict_layer(self, input: List[float]) -> list:
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        return [i.predict(input) for i in self.neurons]

    def error(self, target: List[int or float]) -> None:
        for index, output_neuron in enumerate(self.neurons):
            output_neuron.cal_error_output(output_neuron.antwoord, target[index])

    def error_hidden(self, next_layerW:List[List[float]], next_layerE:List[float]) -> None:
        for index, neuron in enumerate(self.neurons):
            next_weight, next_error = [], []
            for weight in next_layerW:
                next_weight.append(weight[index])
            self.neurons[index].cal_error_hidden(neuron.antwoord, next_weight, next_layerE)
            next_error.append(self.neurons[index].error)
            self.errors = next_error

    def update(self, leariningrate: float = 0.1) -> None:
        for neuron in self.neurons:
            neuron.update(leariningrate)

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.neurons}'
