from typing import List


class Perceptron(object):
    """
     Een class waar de perceptron wordt aangemaakt.
    """
    def __init__(self, weights: List[float], bias: float, threshold: float):
        """
        init perceptron class met de weights, bias, threshold ,
        een lege lijst voor de input en een vriabele voor het opslaan van het antwoord
        """
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.antwoord = 0

        self.iter = 0
        self.error_sum = 0
        self.root_mean_error = 0

    def predict(self, input: List[float]) -> float:
        """
        Een functie die perceptron runt
        :return: de predict voor de input(0 of 1)
        """
        antwoord = 0
        for index in range(len(self.weights)):
           antwoord += input[index] * self.weights[index]
        antwoord += self.bias
        antwoord = self.activation(antwoord)
        self.antwoord = antwoord
        return antwoord

    def activation(self, predict: float) -> int:
        """
        Een functie die een activatie teruggeeft

        :param predict: de som van(weight* input)+bias
        :return: een activatie van 0 of 1
        """
        return 1 if predict >= self.threshold else 0

    def update(self, input: List[int], verwachte_output: [int], learning_rate: float = 0.1) -> None:
        """
        Een functie die de weight en de bias voor perceptron blijft updaten tot dat we op de goede uitput komen.
        :param input: een list van ints met de input voor de perceptron.
        :param verwachte_output: een int van de goede uitkomst voor de input.
        :param learning_rate: een float tussen de 0.1 en 1.0 .
        """
        # y = f(w ∙ x)
        self.predict(input)
        output = self.antwoord
        # e = d – y
        error = verwachte_output - output
        for index in range(len(self.weights)):
            # Δw = η ∙ e ∙ x
            delta_weight = learning_rate * error * input[index]
            # Δwj = η (target(i) – output(i)) xj(i)
            self.weights[index] = self.weights[index] + delta_weight
        # Δb = η ∙ e
        delta_bias = learning_rate * error
        # b' = b + Δb
        self.bias = self.bias + delta_bias
        # Σ | d – y |
        self.error_sum += abs(error)
        # n
        self.iter += 1

    def error(self) -> None:
        # MSE = Σ | d – y |2 / n
        self.root_mean_error = (self.error_sum**2)/self.iter

    def __str__(self) -> str:
        """
        Een fucntie die de eigenschappen van de perceptron netjes uitprint
        """
        return f'Perceptron: weights={self.weights},' \
               f' bias={self.bias}, output ={self.antwoord}'


# p1 = Perceptron(weights=[0.5,0.5], bias=-1, threshold=0)
# p1.predict([1,1])
# p1.update([1,1],1)
# print(p1.weights)
