import unittest
from Neuron import neuron


class TestNeuron(unittest.TestCase):
    # alles met dezelfde parameters als bij perceptron
    def test_AND(self):
        """
        Hier wordt de werking van een AND gate getest
        """
        # Maak de neuron aan
        p1 = neuron.Neuron(weights=[0.5, 0.5], bias=-1)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, p1.predict(input))

        p1 = neuron.Neuron(weights=[80, 80], bias=-100)#alles keer 100
        # Maak de inputs en de outputs aan
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            print(input, output,p1.predict(input))
            # ik weet niet hoe het anders moet vergelijken?
            self.assertEqual(output, int(p1.predict(input)))

    def test_NOT(self):
        """
        Hier wordt de werking van een NOT gate getest
        """
        # Maak de neuron aan
        p3 = neuron.Neuron(weights=[-1], bias=0)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[1], [0]], [0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, p3.predict(input))

        p3 = neuron.Neuron(weights=[-1110], bias=1000) #alles keer 1000
        inputs, outputs = [[1], [0]], [0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            # ik weet niet hoe het anders moet vergelijken?
            self.assertEqual(output, int(p3.predict(input)))


    def test_OR(self):
        """
        Hier wordt de werking van een OR gate getest
        """
        # Maak de neuron aan
        p2 = neuron.Neuron(weights=[0.5, 0.5], bias=-0.5)
        # andere optie

        # Maak de inputs en de outputs aan
        inputs, outputs= [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, p2.predict(input))

        p2 = neuron.Neuron(weights=[2000, 2000], bias=-500)#alles keer 1000
        # Maak de inputs en de outputs aan
        inputs, outputs= [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            # ik weet niet hoe het anders moet vergelijken?
            self.assertEqual(output, int(p2.predict(input)))

    # Waarom werkt een neuron niet met dezelde installatie van een perceptron?
    # Dat komt omdat de actievatie functie oftewel de sigmoid functie anders werkt. Die geeft een getal tussen de 0 en 1
    # in tegen stelling tot de activatie functie die een een int van 1 of 0 terug geeft.

    def test_NOR(self):
        """
        Hier wordt de werking van een NOR gate getest
        """
        # Maak de neuron aan
        p4 = neuron.Neuron(weights=[-1, -1, -1], bias=0)
        # Maak de inputs en de outputs aan
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [1, 0, 0, 0, 0, 0, 0, 0]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, p4.predict(input))

        p4= neuron.Neuron(weights=[-200, -200, -200], bias=200)#alles keer 100
        # Maak de inputs en de outputs aan
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [1, 0, 0, 0, 0, 0, 0, 0]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, int(p4.predict(input)))


if __name__ == '__main__':
    unittest.main()