from typing import List

import Percptron_code.perceptronLayer as PerceptronLayer


class PerceptronNetwork(object):
    """
    Een functie die het netwerk kan aanmaken. Dat bestaat uit een of meerdere perceptronLayer
    die verveolgens uit een of meerdere perceptrons bestaat.
    """
    def __init__(self, layers: [[PerceptronLayer]]):
        """
        Init waar de layers input meegegeven wordt. Er wordt een lege
        lijst aangemaakt voor het opslaan van de netwerk output.
        """
        self.layers = layers
        self.outputs = []

    def feed_forward(self, input: List[float]) -> float:
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

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.outputs}'
