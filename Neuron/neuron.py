from typing import List


class Neuron:
    def __init__(self, weights:List[float], bias: float):
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

    def predict(self, input: List[float]):
        """
        Een functie die perceptron runt
        :return: de predict voor de input(0 of 1)
        """
        antwoord = 0
        for index in range(len(self.weights)):
            antwoord += input[index] * self.weights[index]
        antwoord += self.bias
        antwoord = self.sigmoid(antwoord)
        # self.antwoord = antwoord
        return antwoord

    def sigmoid(self, z):
        return 1 / (1 + 2.7182**(-z))

    def update(self, input, verwachte_output, learning_rate=0.1):
        # y = f(w ∙ x)
        output = self.antwoord
        # e = d – y
        error = verwachte_output - output
        for i in range(len(self.weights)):
            # Δw = η ∙ e ∙ x
            weight = learning_rate * error * input[i]
            # Δwj = η (target(i) – output(i)) xj(i)
            self.weights[i] = self.weights[i] + weight
        # Δb = η ∙ e
        bias = learning_rate * error
        # b' = b + Δb
        self.bias = self.bias + bias
        # n
        self.iter += 1
        # Σ | d – y |
        self.error_sum += abs(error)

    def error(self):
        # MSE = Σ | d – y |2 / n
        self.mean_squared_error = (self.error_sum ** 2) / self.iter

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de perceptron netjes uitprint
        """
        return f'Perceptron: weights={self.weights},' \
               f' bias={self.bias}'

# print("AND")
# # AND
# x = Neuron([0.5,0.5], bias= -1)
# print(x.predict([1,1]))
# print(x.predict([1,0]))
# print(x.predict([0,1]))
# print(x.predict([0,0]))
#
# # x = Neuron([0.3, 0.5], bias= -0.7)
# # print(x.predict([1,1]))
# # print(x.predict([1,0]))
# # print(x.predict([0,1]))
# # print(x.predict([0,0]))
#
#
# print("OR")
#
# # OR
# x = Neuron([1, 1], bias= -1)
# print(x.predict([1,1]))
# print(x.predict([1,0]))
# print(x.predict([0,1]))
# print(x.predict([0,0]))
#
#
# print("NOT")
# x = Neuron([-1], 0)
# print(x.predict([0]))
# print(x.predict([1]))
#
#
# print("NOR")
# x = Neuron([-100, -100, -100], 0)
# inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
# outputs = [1, 0, 0, 0, 0, 0, 0, 0]
#
# for input, output in zip(inputs, outputs):
#     print("input", input,"Verwachte output:",output, "output", x.predict(input))