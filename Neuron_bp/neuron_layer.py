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
        self.weights = [i.weights for i in self.neurons]

    def predict_layer(self, input: [float]):
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        return [i.predict(input) for i in self.neurons]

    def calculate_error(self, verwachte_output:list, weight_volgende_layer: List[int] = [], error_volgende_layer:List[int] = []):
        if len(weight_volgende_layer) == 0 and len(error_volgende_layer) == 0:
            for index in range(len(self.neurons)):
                self.neurons[index].calculate_error(verwachte_output[index], [], [])
                self.errors.append(self.neurons[index].error)
        else:
            for index in range(len(self.neurons)):
                weights = [weight[index] for weight in weight_volgende_layer]
                self.neurons[index].calculate_error(verwachte_output, weights,error_volgende_layer)
                self.errors.append(self.neurons[index].error)

    def calculate_error_layer(self, v_ouput, next_lw =[], next_le=[]):
        neuronsRange = range(len(self.neurons))
        for index in neuronsRange:
            self.neurons[index].calculate_error(verwachte_output=v_ouput[index], weight_volgende_n=[], error_volgende_n=[])

    def change_weight_layer(self, learningrate=0.1):
        for neuron in self.neurons:
            neuron.change_weights(learningrate)

    def update_layer(self):
        for neuron in self.neurons:
            neuron.update()

        self.errors = []
        self.weights = [i.weights for i in self.neurons]



    def __str__(self):
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.neurons}'
