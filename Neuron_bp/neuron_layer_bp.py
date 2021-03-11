from typing import List
from Neuron_bp.neuron_bp import Neuron


class Neuron_layer:
    def __init__(self, neurons: [[Neuron]]):
        self.neurons = neurons
        self.errors = []

    def predict_layer(self, input: List[float]) -> list:
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        return [i.predict(input) for i in self.neurons]

    def error_output(self, target: List[int or float]) -> None:
        """
        Een functie die de eror van de output laag berekent
        :param target: een lijst met verwachte outputs van de laag
        :return: None
        """
        for index, output_neuron in enumerate(self.neurons):
            output_neuron.cal_error_output(output_neuron.antwoord, target[index])

    def error_hidden(self, next_layerW:List[List[float]], next_layerE:List[float]) -> None:
        """
        Een functie die de error vann een hidden layer berekent
        :param next_layerW: een lijst met de weights van de volgende laag
        :param next_layerE: een lijst met de errors van de volgende laag
        :return:None
        """
        for index, neuron in enumerate(self.neurons):
            next_weight, next_error = [], []
            for weight in next_layerW:
                next_weight.append(weight[index])
            self.neurons[index].cal_error_hidden(neuron.antwoord, next_weight, next_layerE)
            next_error.append(self.neurons[index].error)
            self.errors = next_error

    def update(self, leariningrate: float = 0.1) -> None:
        """
        Een functie die de weights en bias van de laag update
        :param leariningrate: een float
        :return: None
        """
        for neuron in self.neurons:
            neuron.update(leariningrate)

    def __str__(self) -> str:
        return f'Layer met deze neuron(s) {self.neurons}en deze errors {self.errors}'
