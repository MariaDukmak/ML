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
            # ik weet niet hoe het anders moet vergelijken? #TODO
            self.assertNotEqual(output, p1.predict(input))

        p1 = neuron.Neuron(weights=[80, 80], bias=-100)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            print(input, output,p1.predict(input))
            # ik weet niet hoe het anders moet vergelijken? #TODO
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
            # ik weet niet hoe het anders moet vergelijken? #TODO
            self.assertNotEqual(output, p3.predict(input))

        p3 = neuron.Neuron(weights=[-110], bias=100) #alles keer 100
        inputs, outputs = [[1], [0]], [0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            # ik weet niet hoe het anders moet vergelijken? #TODO
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
            # ik weet niet hoe het anders moet vergelijken? #TODO
            self.assertNotEqual(output, p2.predict(input))

        p2 = neuron.Neuron(weights=[200, 200], bias=-50)
        # Maak de inputs en de outputs aan
        inputs, outputs= [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            # ik weet niet hoe het anders moet vergelijken? #TODO
            self.assertEqual(output, int(p2.predict(input)))

   #TODO: leg beter uit!
    """"Verklaar waarom dit (niet) werkt?
        Omdat we de simgoid functie geen 0 of 1 krijgen kunnen de output wel afronden maar dat is 
        nog steeds niet hoe het bij een neuron is

    """

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

        p4= neuron.Neuron(weights=[-200, -200, -200], bias=200)
        # Maak de inputs en de outputs aan
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [1, 0, 0, 0, 0, 0, 0, 0]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, int(p4.predict(input)))
