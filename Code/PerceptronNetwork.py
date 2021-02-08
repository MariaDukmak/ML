import Code.PerceptronLayer as pl

class PerceptronNetwork(object):
    def __init__(self, perceptron_list:[pl]):
        self.perceptron_list = perceptron_list
        self.outputs = []

    def feed_forward(self):
        for p in self.perceptron_list:
            self.outputs.append()
        return self.outputs

    def __str__(self) -> str:
        return f'Network feed forward {self.feed_forward()}'

