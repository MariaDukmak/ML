import unittest
from Percptron_code.perceptron import Perceptron
from Percptron_code.perceptronNetwork import PerceptronNetwork
from Percptron_code.perceptronLayer import PerceptronLayer


class TestNewtwek(unittest.TestCase):
    """
    Een class waar de werking van de netwerk met meerdere layers wordt getest.
    Deze class bevat testen voor de XOR logic gate en de half adder.
    """

    def testXOR(self):
        """
        Hier worden de perceptrons, de layers em de netwerk voor een XOR gate aangemaakt
        en vervolgens de output van de netwerk verglijken met de actuele output.
        """
        # Maak de onderdelen van het netwerk aan
        p1 = Perceptron(weights=[1, 1], bias=0, threshold=0.5)
        p2 = Perceptron(weights=[-1,-1], bias=0, threshold=-1.5)
        layer1 = PerceptronLayer(perceptron=[p1, p2])

        p3 = Perceptron(weights=[1,1], bias=0, threshold=2)
        layer2 = PerceptronLayer(perceptron=[p3])

        network = PerceptronNetwork(layers=[layer1, layer2])

        # Maak alle mogelijke inputs en de verwachte outputs aan
        inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        outputs = [[0], [1], [1], [0]]

        # Vergelijk de output van de netwerk met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, network.feed_forward(input))

    def testHALFADDER(self):
        """
        Hier worden de perceptrons, de layers em de netwerk voor een half adder aangemaakt
        en vervolgens de output van de netwerk verglijken met de actuele output.
        """
        # Maak de onderdelen van het netwerk aan
        p1 = Perceptron(weights=[1, 1], bias=0, threshold=1)
        p2 = Perceptron(weights=[-1, -1], bias=0, threshold=-1.5)
        p3 = Perceptron(weights=[1, 1], bias=0, threshold=2)
        layer1 = PerceptronLayer(perceptron=[p1, p2, p3])

        p4 = Perceptron(weights=[1, 1, 0], bias=0, threshold=2)
        p5 = Perceptron(weights=[0, 0, 1], bias=0, threshold=1)
        layer2 = PerceptronLayer(perceptron=[p5, p4])

        netwerk = PerceptronNetwork(layers=[layer1, layer2])

        # Maak alle mogelijke inputs en de verwachte outputs aan
        inputs = [[0,1], [1,1], [1,0], [0,0]]
        outputs = [[0,1], [1,0], [0,1], [0,0]]

        # Vergelijk de output van de netwerk met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, netwerk.feed_forward(input))


if __name__ == '__main__':
    unittest.main()