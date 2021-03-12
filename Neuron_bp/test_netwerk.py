import unittest, numpy
from Neuron_bp.neuron_bp import Neuron
from Neuron_bp.neuron_layer_bp import Neuron_layer
from Neuron_bp.neuron_network_bp import Neuron_network


class TestNetwerk(unittest.TestCase):
    def test_XOR(self):
        #Maak de onderdelen van het netwerk aan
        n1 = Neuron(weights=[0.2, -0.4], bias=0)
        n2 = Neuron(weights=[0.7, 0.1], bias=0)
        layer1 = Neuron_layer([n1, n2])

        n4 = Neuron(weights=[0.6, 0.9], bias=0.0)

        layer2 = Neuron_layer([n4])

        netwerk = Neuron_network(layers=[layer1, layer2])

        inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        outputs = [[0], [1], [1], [0]]
        for input, output in zip(inputs, outputs):
            antw_list = netwerk.feed_forward(input)
            print(antw_list)
        netwerk.train(inputs, outputs)

        print("na train")

        for input, output in zip(inputs, outputs):
            antw_list = netwerk.feed_forward(input)
            print(antw_list, output)
            numpy.testing.assert_array_almost_equal(antw_list, output, 1)

    def test_HALFADDER(self):

        #Maak de onderdelen van het netwerk aan

        n1 = Neuron(weights=[0.0, 0.1], bias=0)
        n2 = Neuron(weights=[0.2, 0.3], bias=0)
        n3 = Neuron(weights=[0.4, 0.5], bias=0)
        layer1 = Neuron_layer([n1, n2, n3])

        n4 = Neuron(weights=[0.6, 0.7, 0.8], bias=0)
        n5 = Neuron(weights=[0.9, 1.0, 1.1], bias=0)
        layer2 = Neuron_layer([n4, n5])

        netwerk = Neuron_network(layers=[layer1, layer2])

        inputs = [[0, 1], [1, 1], [1, 0], [0, 0]]
        outputs = [[0, 1], [1, 0], [0, 1], [0, 0]]
        for input, output in zip(inputs, outputs):
            antw_list = netwerk.feed_forward(input)
            print(antw_list)
        netwerk.train(inputs, outputs)

        print("na trainen")

        for input, output in zip(inputs, outputs):
            antw_list = netwerk.feed_forward(input)
            print(antw_list, output)
            numpy.testing.assert_array_almost_equal(antw_list, output, 1)


if __name__ == '__main__':
    unittest.main()