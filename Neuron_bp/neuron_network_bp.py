from Neuron_bp.neuron_layer_bp import Neuron_layer
import time, random
from typing import List


class Neuron_network:
    def __init__(self, layers: [[Neuron_layer]]):
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
            self.outputs.append(i.predict_layer (list((self.outputs[-1]))))
        self.outputs = self.outputs[-1]
        return self.outputs

    def calculate_loss(self, target: List[int or float], loss_sum: int = 0) -> None:
        """
        Een functie die de loss van de netwerk berekent voor 1 traning voorbeeld
        :param target: verwachte output
        :param loss_sum: de sum van de loss
        :return: None
        """
        for index in range(len(target)):
            loss_sum += (target[index] - self.outputs[index])**2
        self.loss.append(loss_sum/len(target))

    def MSE(self, inputs:[List[float]], targets:[List[float]]) -> float:
        """
        Een functie die de MSE voor de hele netewerk berekent.
        :return: de totale loss van de hele netwerk
        """
        for i in range(len(inputs)):
            self.feed_forward(inputs[i])
            self.calculate_loss(targets[i])

        total_loss = sum(self.loss) / len(self.loss)
        self.loss = []
        return total_loss

    def train(self, inputs:List[List[float]], targets:List[List[float]], learning_rate:float = 0.1,
              epoches:int=10000, max_time:int=200) -> None:
        """
        Een functie die de tranien door de hele netwerk doet.
        :param inputs: een lijst met de inputs lijsten
        :param targets: een lijst met lijst van de verwachte outputs
        :param learning_rate: een getal ?
        :param epoches: een int van aantaal epoches
        :param max_time: een int van de max train tijd
        :return: None
        """
        start_time, epoch = time.time(), 0
        while epoches > epoch and time.time()-start_time < max_time:
            ## ADD RANDOM SHIFFEL INPUT!!!!!
            # for epoch in range(epoches):
            # inputs = random.shuffle(list(range(inputs)))
            for index, input_list in enumerate(inputs):
                self.feed_forward(input_list)
                target = targets[index]

                for i in range(len(self.layers[::-1])):
                    if i == 0:self.layers[i-1].error_output(target)
                    else:
                        next_weights = [neuron.weights for neuron in self.layers[i].neurons]
                        next_errors = [neuron.error for neuron in self.layers[i].neurons]
                        self.layers[i-1].error_hidden(next_weights, next_errors)

                for i in range(len(self.layers[::-1])):
                    self.layers[i-1].update(learning_rate)
            epoch += 1
        print("total loss", self.MSE(inputs, targets))
        print("epoches", epoch)


    def __str__(self) -> str:

        return f'Deze netwerk bestaat uit deze lagen {self.layers} en heeft een output van {self.outputs}'
