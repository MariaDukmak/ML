import Code.perceptron as perceptron


class PerceptronLayer(object):
    """
    Een class waar de lagen van de netwerk aangemaakt kunnen worden.
    """
    def __init__(self, perceptron=[[perceptron]]):
        """
        init waar de verwachte onderdelen van de laag gedefineerd worden.
        Een laag kan en of meerdere perceptrons bevaten. Er wordt ook een
        leege lijst aangeemkt waar de output van de laag wordt opgeslagen.
         """
        self.perceptron = perceptron
        self.output = []

    def predict_layer(self, input: [float]):
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        for i in self.perceptron:
            # hier wordt de activatie per perceptron berekend
            self.output.append(i.predict(input))
        return self.output

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.perceptron} heeft de output van {self.output}'
# p1 = pr.Perceptron([0.5,0.5], 0, 1)
# p2 = pr.Perceptron([0.5, 0.5], 0, 1)
# x = PerceptronLayer([p1, p2])
# x1 = PerceptronLayer([p1, p2])
#
# print(x.set_p([0,0]))
#
# class PerceptronLayer(object):
# #     """
# #     Een class die de Layer van Perceptron aanmaakt
# #     """
# #     def __init__(self, weights: list, bias: float, threshold: float):
# #         """Een init waar de weights, bias, threshold ,een lege lijst voor de antwoorden
# #          en een lege lijst voor de Perceptron inputs"""
# #         self.weights = weights
# #         self.bias = bias
# #         self.threshold = threshold
# #         self.antwoord = []
# #         self.perceptron_inputs = []
# #         # Aanmaken van een Perceptron dat dat vervolgens de class gebruikt kan worden
# #         self.per = pr.Perceptron(self.weights, self.bias, self.threshold)
# #
# #     def set_input(self, inputs: list):
# #         """
# #         Een functie die de input van de layer set
# #         :param inputs: een lijst met de inputs
# #         """
# #         self.perceptron_inputs = inputs
# #
# #     def predict(self):
# #         """
# #         Een functie die de predicts van de layer maakt
# #         :return: een lijst met de activaties
# #         """
# #         for input in self.perceptron_inputs:
# #             self.per.set_input(input)
# #             self.antwoord.append(self.per.predict())
# #         return self.antwoord
# #
# #     def __str__(self) :
# #         return f'Deze laag heeft als uitkomst:{self.predict()}'