from unittest import TestCase
from Code.perceptron import Perceptron
from Code.perceptronNetwork import PerceptronNetwork
from Code.perceptronLayer import PerceptronLayer

class TestPerceptron(TestCase):
    def testPAND(self):
        p1 = Perceptron(weights=[0.5, 0.5], bias=0, threshold=1)
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        outputs = [0, 0, 0, 1]

        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p1.predict(input))

    def testPOR(self):
        p2 = Perceptron(weights=[0.5, 0.5], bias=0.5, threshold=1)
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        outputs = [0, 1, 1, 1]

        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p2.predict(input))

    def testPNOT(self):
        p3 = Perceptron(weights=[-1], bias=0, threshold=-0.5)

        self.assertEqual(p3.predict([1]), 0)
        self.assertEqual(p3.predict([0]), 1)

    def testNOR(self):
        p4 = Perceptron(weights=[-1, -1, -1], bias=0, threshold=0)
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [1, 0, 0, 0, 0, 0, 0, 0]
        for input, output in zip(inputs, outputs):
            print(input)
            print(output)
            self.assertEqual(output, p4.predict(input))

    def testPARTY(self):
        p5 = Perceptron(weights= [0.6, 0.3, 0.2],  bias=0, threshold= 0.4)
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [0, 0, 0, 1, 1, 1,1, 1]
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p5.predict(input))


class TestNewtwek(TestCase):

    def testXOR(self):
        p1 = Perceptron(weights=[1, 1], bias=0, threshold=0.5)
        p2 = Perceptron(weights=[-1,-1], bias=0, threshold=-1.5)

        layer1 = PerceptronLayer(perceptron=[p1, p2])

        p3 = Perceptron(weights=[1,3], bias=0, threshold=1.5)
        layer2 = PerceptronLayer(perceptron=[p3])

        network = PerceptronNetwork(layers=[layer1, layer2])

        input = [1, 1]
        self.assertEqual([0], network.feed_forward(input))

    def testHALFADDER(self):

        p1 = Perceptron(weights=[1, 1], bias=0, threshold=1)
        p2 = Perceptron(weights=[-1, -1], bias=0, threshold=-1.5)
        p3 = Perceptron(weights=[1, 1], bias=0, threshold=2)
        layer1 = PerceptronLayer(perceptron=[p1, p2, p3])

        p4 = Perceptron(weights=[1, 1, 0], bias=0, threshold=2)
        p5 = Perceptron(weights=[0, 0, 1], bias=0, threshold=1)

        layer2 = PerceptronLayer(perceptron=[p5, p4])

        netwerk = PerceptronNetwork(layers=[layer1, layer2])
        inputs = [[0,1], [1,1], [1,0], [0,0]]
        outputs = [[0,1], [1, 0], [0,1], [0,0]]

        for input, output in zip(inputs, outputs):
            self.assertEqual(output, netwerk.feed_forward(input))
