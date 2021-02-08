import Perceptron.perceptron as pe
# Perceptron unit
# #TODO: maak unittest dat je dit daar doet ipv hier "ziet er iets strakker uit hihi"
# Initialiseer een Perceptron voor elk van de INVERT-, AND- en OR-poorten en test of ze op de juiste manier werken.
inputs = [[0,0], [0,1], [1,0], [1,1]]
bias = 0

AND = pe.Perceptron(weights=[0.5, 0.5], inputs=inputs,bias=bias, t=1)
AND.predict()
print("AND", AND.__str__())

OR = pe.Perceptron(weights=[0.5, 0.5], inputs=inputs,bias=bias, t=0.5)
OR.predict()
print("OR",OR.__str__())


NOT = pe.Perceptron(weights=[-1], inputs=[[0], [1]], bias=0, t=-0.5)
NOT.predict()
print("NOT",NOT.__str__())

#Initialiseer een Perceptron voor een NOR-poort met drie ingangen en test of deze op de juiste manier werkt.

inputs_3 = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
NOR = pe.Perceptron(weights=[-1,-1,-1], inputs =inputs_3, bias=bias, t=0)
NOR.predict()
print("NOR",NOR.__str__())

#Initialiseer ook een PerAceptron voor een uitgebreider beslissysteem
# (minimaal 3 inputs, zie bijvoorbeeld Figuur 2.8 uit de reader)
# en test of deze naar verwachting werkt.

figuur_28= pe.Perceptron(weights= [0.6, 0.3, 0.2], inputs= inputs_3, bias=bias,t= 0.4)
figuur_28.predict()
print("Figuur_28",figuur_28.__str__())