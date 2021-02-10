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


    def predict_layer(self, input: [float]):
        """
        Een functie waar de activatie van de laag wordt berekend.
        :param input: de input van de laag
        :return: een lijst met de activatie tussen 0 en 1
        """
        self.output = []
        for i in self.perceptron:
            # hier wordt de activatie per perceptron berekend
            self.output.append(i.predict(input))
        return self.output

    def __str__(self):
        """
        Een fucntie die de eigenschappen van de laag netjes uitprint
        """
        return f'Layer met deze perceptron(s) {self.perceptron} heeft de output van {self.output}'
