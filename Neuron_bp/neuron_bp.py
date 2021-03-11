from typing import List


class Neuron:
    def __init__(self, weights: List[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.antwoord = 0
        self.error = 0
        self.input = []

    def check_input(self, input: List[float]) -> None:
        """
        Een functie die de lengte van de input checkt.
        :param input: de input van de perceptron
        :return: None
        """
        assert len(self.weights) == len(input), "ongeldige input lengte"

    def predict(self, input: List[float]) -> float:
        """
        Een functie die neuron runt
        :return: de predict voor de gegeven input,
        oftwel de uitkomst van de sigmooid functie
        """
        self.check_input(input)
        self.input = input
        antwoord = 0
        for index in range(len(self.weights)):
            antwoord += input[index] * self.weights[index]
        antwoord += self.bias
        antwoord = self.sigmoid(antwoord)
        self.antwoord = antwoord
        return antwoord

    def sigmoid(self, predict: float or int) -> float:
        """
        Een activatie functie
        :return een activatie tussen 0 en 1
        """
        e = 2.7182
        return 1 / (1 + e**(-predict))

    def cal_derivative(self, output: float) -> float:
        """
        Een functie die een stukje van de vergelijking
        oplost voor zowel de output als de hidden neuron
        :return een float
        """
        # outputj ∙ (1 – outputj)
        return output * (1 - output)

    def cal_error_output(self, output: float,  expected_output: int) -> float:
        """
        Een functie die de error van een output neuron berekent
        :return de error van de neuron
        """
        # Δj = σ'(inputj) ∙ – (targetj – outputj)
        error = self.cal_derivative(output) * -(expected_output - output)
        self.error = error
        return self.error

    def cal_error_hidden(self, output: float, next_weight: List[float], next_error: List[float]) -> float:
        """
        Een functie die de error voor een hidden neuron berekent.
        :return de error van de neuron
        """
        # Δi = σ'(input) ∙ Σj wi,j ∙ Δj
        sum_error = 0
        for index in range(len(next_weight)):
            sum_error += next_weight[index] * next_error[index]
        self.error = self.cal_derivative(output) * sum_error
        return self.error

    def update(self, learning_rate: float = 0.1) -> None:
        """
        Een functie die de weights en bias update door het
        te aftereken van de delta.
        :return None
        """
        for i in range(len(self.weights)):
            # w'i,j = wi,j – Δwi,j
            self.weights[i] -= (learning_rate * self.error * self.input[i])
        # b'j = bj – Δbj
        self.bias -= (learning_rate * self.error)

    def __str__(self) -> str:
        return f'Neuron: weights={self.weights},' \
               f' bias={self.bias} en de error {self.error}'
