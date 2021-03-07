import unittest
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
        self.neuron_input = []

        self.iter = 0
        self.error = 0
        self.updated_weights = []
        self.updated_bias = 0

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
        self.neuron_input = input
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

    def calculate_error(self, verwachte_output, weight_volgende_n, error_volgende_n):
        #Check eerst of het een end/output neuron is
        #(die geen andere inputs nodig voor het berekenen van de error)
        if len(weight_volgende_n) == 0 and len(error_volgende_n) == 0:
            #neuron erorr =  outputs ∙ (1 – outputs) × –(targets – outputs)
            self.error = self.antwoord * (1 - self.antwoord) * -(verwachte_output - self.antwoord)
        # (die heeft dan een lijst van de wights en de errors van de output neuron nodig)
        else:
            # hidden_neuron_error = output ∙ (1 – output) ∑i (w,i ∙ Δi)
            segma_neurons= self.calculate_sum_erros(weight_volgende_n, error_volgende_n)
            self.error = self.antwoord * (1 - self.antwoord) * segma_neurons

    def calculate_sum_erros(self, weight_volgende_n, error_volgende_n):
        # ∑i (wf,i ∙ Δi) voor hidden neuron
        segma_neurons = 0
        for index in range(len(weight_volgende_n)):
            segma_neurons += weight_volgende_n[index] * error_volgende_n[index]
        return segma_neurons

    def change_weights(self, learingrate=0.1):
        self.updated_weights = []
        # self.updated_bias= 0
        for index in range(len(self.weights)):
            # w'i = wi −∆wi = wi −η∂C/∂wi
            self.updated_weights.append(self.weights[index] - (learingrate * self.error * self.neuron_input[index]))
        # b'j = bj – Δbj
        self.updated_bias = self.bias - (learingrate * self.error)

    def get_update(self):
        self.weights = self.updated_weights
        self.bias = self.updated_bias

        # reset de updateweight en de update bias lijsten
        self.updated_weights = []
        self.updated_bias = 0

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de perceptron netjes uitprint
        """
        return f'Perceptron: weights={self.weights},' \
               f' bias={self.bias}'


class TestNeuron(unittest.TestCase):

    def test_AND(self):
        """
        Hier wordt de werking van een AND gate getest
        """
        # Maak de neuron aan
        p1 = Neuron(weights=[-0.5, 0.5], bias=-1.5)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        # Vergelijk de output met de verwachte output
        # for input, output in zip(inputs, outputs):
        #     self.assertNotEqual(output, p1.predict(input))

        for _ in range(100000):
            for input, output in zip(inputs, outputs):
                p1.predict(input)
                p1.calculate_error(output, [], [])
                p1.change_weights(1)
                p1.get_update()
                # print(p1)

        # DIT WERKT!
        for input, output in zip(inputs, outputs):
            p1.predict(input)
            print(p1.antwoord, output)
            self.assertEqual(round(p1.antwoord), output)
