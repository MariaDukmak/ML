from Neuron_bp.neuron_layer import Neuron_layer
from Neuron_bp.neuron import Neuron
import time, random
from typing import List


class Neuron_network:
    """
    Een functie die het netwerk kan aanmaken. Dat bestaat uit een of meerdere perceptronLayer
    die verveolgens uit een of meerdere perceptrons bestaat.
    """
    def __init__(self, layers: [[Neuron_layer]]):
        """
        Init waar de layers input meegegeven wordt. Er wordt een lege
        lijst aangemaakt voor het opslaan van de netwerk output.
        """
        self.layers = layers
        self.outputs = []
        self.loss = []

    def feed_forward(self, input: List[float]) -> list:
        """
        Een functie die de activatie van de netwerk berekend. Er wordt hier rekening gemaakt met het
        feit dat de input van de volgnde laag de output van de vorige laag moet zijn.

        :param input: een lijst met de input voor de eerste laag
        :return: een int met de activatie van de netwerk tussen 0 en 1
        """
        self.outputs = [input]
        for i in self.layers:
            self.outputs.append(i.predict_layer(self.outputs[-1]))
        self.outputs = self.outputs[-1]
        return self.outputs

    def calculate_loss(self, target: List[int or float], loss_sum: int = 0) -> None:
        for index in range(len(target)):
            loss_sum += (target[index] - self.outputs[index])**2
        self.loss.append(loss_sum/len(target))

    def calculate_total_loss(self) -> float:
        total_loss = sum(self.loss) / len(self.loss)
        self.loss = []
        return total_loss

    def train(self, inputs:List[List[float]], targets:List[List[float]], epoches:int=10000, max_time:int=200) -> None:
        start_time, epoch = time.time(), 0

        while epoches > epoch and time.time()-start_time < max_time:
            # for epoch in range(epoches):
            # inputs = random.shuffle(list(range(inputs)))
            for index, input_list in enumerate(inputs):
                self.feed_forward(input_list)
                target = targets[index]

                for i in range(len(self.layers[::-1])):
                    if i == 0:self.layers[i-1].error(target)
                    else:
                        next_weights = [neuron.weights for neuron in self.layers[i].neurons]
                        next_errors = [neuron.error for neuron in self.layers[i].neurons]
                        self.layers[i-1].error_hidden(next_weights, next_errors)

                for i in range(len(self.layers[::-1])):
                    self.layers[i-1].update()
                self.calculate_loss(target)
            epoch += 1
            print("loss", self.calculate_total_loss())

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de netwerk netjes uitprint
        """
        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.outputs}'
