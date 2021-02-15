from typing import List
import Percptron_code.perceptron as perceptron


class PerceptronLayer(object):
    """
    Een class waar de lagen van de netwerk aangemaakt kunnen worden.
    """
    def __init__(self, perceptron:[[perceptron]]):
        """
        init waar de verwachte onderdelen van de laag gedefineerd worden.
        Een laag kan en of meerdere perceptrons bevaten. Er wordt ook een
        leege lijst aangeemkt waar de output van de laag wordt opgeslagen.
         """
        self.perceptron = perceptron

    def predict_layer(self, input: List[float]) -> list:
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        return [i.predict(input) for i in self.perceptron]

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.perceptron}'
