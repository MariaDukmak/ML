import Code.perceptron as pr

class PerceptronLayer(object):
    def __init__(self, inputs:[pr], weights, bias, t):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.t = t
        self.ant = []

    def verzamel_perdict(self):
        for input in self.inputs:
            # ant = pr.Perceptron(self.weights,input,self.bias, self.t).predict()
            self.ant.append()
        return self.ant

    def __str__(self) :
        return f'Deze laag heeft als uitkomst:{self.verzamel_perdict()}'



x = PerceptronLayer([[0,0], [0,1], [1,0], [1,1]], [0.5, 0.5], 0, 1)

print(x)