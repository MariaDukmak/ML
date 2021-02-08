
class PerceptronNetwork(object):
    def __init__(self, layer_list:list):
        self.layer_list = layer_list
        self.input_network = []
        self.outputs = []
        # self.layer = pl.PerceptronLayer()

    def set_input(self, netwerk_list: list):
        self.input_network = netwerk_list

    # fix this dat de output ven layer 1 de input van de next layer is

    def feed_forward(self):
    #     for layer in self.layer_list:
    #         layer.set_input(self.input_network)
    #         layer.verzamel_predict()
    #         self.outputs.append(pl.predict())
    #     return self.outputs
        pass

    def __str__(self) -> str:
        return f'Network feed forward {self.feed_forward()}'

