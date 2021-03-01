from typing import List


class Neuron:
    def __init__(self, weights: List[float], bias: float):
        """
        init perceptron class met de weights, bias, threshold ,
        een lege lijst voor de input en een vriabele voor het opslaan van het antwoord
        """
        self.weights = weights
        self.bias = bias
        self.antwoord = 0

        self.iter = 0
        self.error_sum = 0
        self.mean_squared_error = 0

    def check_input(self, input: List[float]) -> None:
        """
        Een functie die de lengte van de input checkt.
        :param input: de input van de perceptron
        :return: None
        """
        assert len(self.weights) == len(input), "ongeldige input lengte"

    def predict(self, input: List[float]) -> float:
        """
        Een functie die perceptron runt
        :return: de predict voor de input(0 of 1)
        """
        self.check_input(input)
        antwoord = 0
        for index in range(len(self.weights)):
            antwoord += input[index] * self.weights[index]
        antwoord += self.bias
        antwoord = self.sigmoid(antwoord)
        return antwoord

    def sigmoid(self, predict: float or int) -> float:
        """
        Een activatie functie
        :return een activatie tussen 0 en 1
        """
        e = 2.7182
        return 1 / (1 + e**(-predict))

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de perceptron netjes uitprint
        """
        return f'Perceptron: weights={self.weights},' \
               f' bias={self.bias}'
