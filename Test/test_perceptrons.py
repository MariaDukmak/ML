import unittest
from Code.perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    """
    Een class waar de werking van de perceptron wordt getest.
    Door de werking van de AND, OR, NOT, NOR, PARTY te testen.
    """
    def testAND(self):
        """
        Hier wordt de werking van een AND gate getest
        """
        # Maak de perceptron aan
        p1 = Perceptron(weights=[0.5, 0.5], bias=0, threshold=1)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p1.predict(input))

    def testOR(self):
        """
        Hier wordt de werking van een OR gate getest
        """
        # Maak de perceptron aan
        p2 = Perceptron(weights=[0.5, 0.5], bias=0.5, threshold=1)
        # Maak de inputs en de outputs aan
        inputs, outputs= [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p2.predict(input))

    def testNOT(self):
        """
        Hier wordt de werking van een NOT gate getest
        """
        # Maak de perceptron aan
        p3 = Perceptron(weights=[-1], bias=0, threshold=-0.5)
        # Maak de inputs en de outputs aan
        inputs, outputs = [[1], [0]], [0, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p3.predict(input))

    def testNOR(self):
        """
        Hier wordt de werking van een NOR gate getest
        """
        # Maak de perceptron aan
        p4 = Perceptron(weights=[-1, -1, -1], bias=0, threshold=0)
        # Maak de inputs en de outputs aan
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [1, 0, 0, 0, 0, 0, 0, 0]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p4.predict(input))

    def testPARTY(self):
        """
        Hier wordt de werking van een PARTY gate getest
        """
        # Maak de perceptron aan
        p5 = Perceptron(weights=[0.6, 0.3, 0.2],  bias=0, threshold= 0.4)
        # Maak de inputs en de outputs aan
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        outputs = [0, 0, 0, 1, 1, 1, 1, 1]
        # Vergelijk de output met de verwachte output
        for input, output in zip(inputs, outputs):
            self.assertEqual(output, p5.predict(input))


if __name__ == '__main__':
    unittest.main()
