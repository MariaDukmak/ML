import unittest
from Neuron import neuron, neuron_layer, neuron_network


class TestNetwerk(unittest.TestCase):

    # TODO: fix dit
    def testHALFADDER(self):
        """
        Hier worden de perceptrons, de layers em de netwerk voor een half adder aangemaakt
        en vervolgens de output van de netwerk verglijken met de actuele output.
        """
        # Maak de onderdelen van het netwerk aan
        p1 = neuron.Neuron(weights=[1, 1], bias=0)
        p2 = neuron.Neuron(weights=[-1, -1], bias=0)
        p3 = neuron.Neuron(weights=[1, 1], bias=0)
        layer1 = neuron_layer.Neuron_layer(perceptron=[p1, p2, p3])

        p4 = neuron.Neuron(weights=[1, 1, 0], bias=0)
        p5 = neuron.Neuron(weights=[0, 0, 1], bias=0)
        layer2 = neuron_layer.Neuron_layer(perceptron=[p5, p4])

        netwerk = neuron_network.Neuron_network(layers=[layer1, layer2])

        # Maak alle mogelijke inputs en de verwachte outputs aan
        inputs = [[0,1], [1,1], [1,0], [0,0]]
        outputs = [[0,1], [1,0], [0,1], [0,0]]

        # Vergelijk de output van de netwerk met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, netwerk.feed_forward(input))


if __name__ == '__main__':
    unittest.main()