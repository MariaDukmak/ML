import unittest
from Neuron import neuron, neuron_layer, neuron_network


class TestNetwerk(unittest.TestCase):
    def test_HALFADDER(self):
        """
        Hier worden de perceptrons, de layers em de netwerk voor een half adder aangemaakt
        en vervolgens de output van de netwerk verglijken met de actuele output.
        """
        # Maak de onderdelen van het netwerk aan
        n1 = neuron.Neuron(weights=[24, 24], bias=-12)
        n2 = neuron.Neuron(weights=[-12, -12], bias=18)
        n3 = neuron.Neuron(weights=[12, 12], bias=-18)
        layer1 = neuron_layer.Neuron_layer(perceptron=[n1, n2, n3])

        n4 = neuron.Neuron(weights=[12, 12, 0], bias=-18)
        n5 = neuron.Neuron(weights=[0, 0, 24], bias=-12)
        layer2 = neuron_layer.Neuron_layer(perceptron=[n5, n4])

        netwerk = neuron_network.Neuron_network(layers=[layer1, layer2])

        # Maak alle mogelijke inputs en de verwachte outputs aan
        inputs = [[0,1], [1,1], [1,0], [0,0]]
        outputs = [[0,1], [1,0], [0,1], [0,0]]

        # Vergelijk de output van de netwerk met de verwachte output
        for input, output in zip(inputs, outputs):
            antw_list = netwerk.feed_forward(input)
            self.assertEqual(output, [round(antw_list[0]), round(antw_list[1])])


if __name__ == '__main__':
    unittest.main()