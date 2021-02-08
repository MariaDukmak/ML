class Perceptron(object):
    """
     Een class waar de Perceptron aangemaakt wordt
    """
    def __init__(self, weights: list, bias: float, threshold: float):
        """
        init Perceptron class met de weights, bias, threshold ,
        een lege lijst voor de input en een vriabele voor het opslaan van het antwoord
        """
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.antwoord = 0
        self.input = []

    def set_input(self, perceptron_input: list):
        """
        Een fuctie waar de input vab de Perceptron geset kan worden.
        :param perceptron_input: list van de inputs
        :return: None
        """
        if len(perceptron_input) == len(self.weights):
            self.input = perceptron_input
        else:return None

    def predict(self):
        """
        Een functie die Perceptron runt
        :return: de predict voor de input(0 of 1)
        """
        for index in range(len(self.weights)):
            self.antwoord += self.input[index] * self.weights[index]
        self.antwoord += self.bias
        self.antwoord = self.activation(self.antwoord)
        return self.antwoord

    def activation(self, predict):
        """
        Een functie die een activatie teruggeeft
        :param predict: de som van(weight* input)+bias
        :return: een activatie van 0 of 1
        """
        return 1 if predict >= self.threshold else 0

    def __str__(self) -> str:
        return f'Perceptron: \n input =, weights={self.weights},' \
               f' bias={self.bias}\n en the predict is:{self.predict()}'

