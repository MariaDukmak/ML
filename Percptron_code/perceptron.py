from typing import List


class Perceptron(object):
    """
     Een class waar de perceptron wordt aangemaakt.
    """
    def __init__(self, weights:List[float], bias: float, threshold: float):
        """
        init perceptron class met de weights, bias, threshold ,
        een lege lijst voor de input en een vriabele voor het opslaan van het antwoord
        """
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.antwoord = 0

    def check_input(self, input: List[float]) -> None:
        """
        Een functie die de lengte van de input checkt.
        :param input: de input van de perceptron
        :return: None
        """
        assert len(self.weights) == len(input), "ongeldige input lengte"

    def predict(self, input: List[float]) -> int:
        """
        Een functie die perceptron runt
        :return: de predict voor de input(0 of 1)
        """
        self.check_input(input)
        self.antwoord = 0
        for index in range(len(self.weights)):
            self.antwoord += input[index] * self.weights[index]
        self.antwoord += self.bias
        self.antwoord = self.activation(self.antwoord)
        return self.antwoord

    def activation(self, predict: float) -> int:
        """
        Een functie die een activatie teruggeeft

        :param predict: de som van(weight* input)+bias
        :return: een activatie van 0 of 1
        """
        return 1 if predict >= self.threshold else 0

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de perceptron netjes uitprint
        """
        return f'Perceptron: \n weights={self.weights},' \
               f' bias={self.bias}\n en the predict is:{self.antwoord}'
