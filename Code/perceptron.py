class Perceptron(object):
    def __init__(self, weights:list, inputs:list, bias:int, t: float):
        self.weights = weights
        self.inputs = inputs
        self.bias = bias
        self.t = t
        self.ant = 0
    # def check_length(self):
    #     if len(self.weights) != len(self.inputs):
    #         raise DimensionError("Vectors hebben niet hetzelfde formaat")
    #     else:
    #         return True
    #
    # def matrix_product(self):
    #     # if self.check_length() is True:
    #     for i in self.inputs:
    #         res = 0
    #         for weight in range(len(self.weights)):
    #             res += (i * self.weights[weight])
    #         self.lijst.append(res+self.bias)
    #     return self.lijst
    # def matrix_product(self):
    #     for i in self.inputs:
    #         for j in self.weights:
    #             ant = (i*j) + self.bias
    #         self.lijst.append(ant)
    #     self.lijst = sum(self.lijst)
    #     return self.lijst

    def product(self):
        for index in range(len(self.weights)):
            self.ant += self.inputs[index] * self.weights[index]
        self.ant += self.bias
        return self.ant

    def activation(self, input):
        return 1 if input >= self.t else 0

    def predict(self):
        pred = self.activation(self.product())
        return pred

    def __str__(self) -> str:
        return f'Perceptron: \n input = {self.inputs}, weights={self.weights},' \
              f' bias={self.bias}\n en the predict is:{self.predict()}'

