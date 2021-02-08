class DimensionError(Exception):
    pass

class Perceptron(object):
    def __init__(self, weights:list, inputs:list, bias:int, t: float):
        self.weights = weights
        self.inputs = inputs
        self.bias = bias
        self.t = t
        self.pred = []
        self.lijst = []

    def get_length(self):
        if len(self.weights) != len(self.inputs[0]):
            raise DimensionError("Vectors hebben niet hetzelfde formaat")
        else:
            return True

    def matrix_product(self):
        if self.get_length() is True:
            for i in self.inputs:
                res = 0
                for weight in range(len(self.weights)):
                    res += (i[weight] * self.weights[weight])
                self.lijst.append(res+self.bias)
        return self.lijst

    def activation(self, input):
        return 1 if input >= self.t else 0

    def predict(self):
        self.pred = [self.activation(i) for i in self.matrix_product()]

    def __str__(self) -> str:
        return f'Perceptron: \n input = {self.inputs}, weights={self.weights},' \
               f' bias={self.bias}\n en the predict is:{self.pred}'


x = Perceptron([[0.5, 0.5],[0.5,0.5]],[[0,0], [0,1], [1,0], [1,1]], 0,1)
x.predict()
print(x.__str__())
