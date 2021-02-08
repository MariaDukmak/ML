import Code.perceptron as pr


class PerceptronLayer(object):
    """
    Een class die de Layer van Perceptron aanmaakt
    """
    def __init__(self, weights: list, bias: float, threshold: float):
        """Een init waar de weights, bias, threshold ,een lege lijst voor de antwoorden
         en een lege lijst voor de Perceptron inputs"""
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.antwoord = []
        self.perceptron_inputs = []
        # Aanmaken van een Perceptron dat dat vervolgens de class gebruikt kan worden
        self.per = pr.Perceptron(self.weights, self.bias, self.threshold)

    def set_input(self, inputs: list):
        """
        Een functie die de input van de layer set
        :param inputs: een lijst met de inputs
        """
        self.perceptron_inputs = inputs

    def predict(self):
        """
        Een functie die de predicts van de layer maakt
        :return: een lijst met de activaties
        """
        for input in self.perceptron_inputs:
            self.per.set_input(input)
            self.antwoord.append(self.per.predict())
        return self.antwoord

    def __str__(self) :
        return f'Deze laag heeft als uitkomst:{self.predict()}'
