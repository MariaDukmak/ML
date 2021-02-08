import Code.perceptron as pe
import Code.PerceptronLayer as pl




# Code unit
# #TODO: maak unittest dat je dit daar doet ipv hier "ziet er iets strakker uit hihi"
# Initialiseer een Code voor elk van de INVERT-, AND- en OR-poorten en test of ze op de juiste manier werken.
input1, input2, input3, input4= [0,0], [0,1], [1,0], [1,1]
bias = 0
# x = pe.Code(weights=[0.5, 0.5], inputs =[[0,0], [0,1], [1,0], [1,1]], bias = bias, t =1)
# print(x)

print("AND")
AND1 = pe.Perceptron(weights=[0.5, 0.5], inputs=input1,bias=bias, t=1)
AND2 = pe.Perceptron(weights=[0.5, 0.5], inputs=input2,bias=bias, t=1)
AND3 = pe.Perceptron(weights=[0.5, 0.5], inputs=input3,bias=bias, t=1)
AND4 = pe.Perceptron(weights=[0.5, 0.5], inputs=input4,bias=bias, t=1)
AND1.predict()
AND2.predict()
AND3.predict()
AND4.predict()
print(AND1, AND2, AND3, AND4)

print("OR")
OR1 = pe.Perceptron(weights=[0.5, 0.5], inputs=input1,bias=bias, t=0.5)
OR2 = pe.Perceptron(weights=[0.5, 0.5], inputs=input2,bias=bias, t=0.5)
OR3 = pe.Perceptron(weights=[0.5, 0.5], inputs=input3,bias=bias, t=0.5)
OR4 = pe.Perceptron(weights=[0.5, 0.5], inputs=input4,bias=bias, t=0.5)
OR1.predict()
OR2.predict()
OR3.predict()
OR4.predict()
print(OR1, OR2, OR3, OR4)

print("NOT")
NOT1 = pe.Perceptron(weights=[-1], inputs=[0], bias=0, t=-0.5)
NOT2 = pe.Perceptron(weights=[-1], inputs=[1], bias=0, t=-0.5)
NOT1.predict()
NOT2.predict()
print(NOT1, NOT2)

# #Initialiseer een Code voor een NOR-poort met drie ingangen en test of deze op de juiste manier werkt.

# inputs_3 = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
# NOR = pe.Code(weights=[-1,-1,-1], inputs =inputs_3, bias=bias, t=0)
# NOR.predict()
# print("NOR",NOR.__str__())

# #Initialiseer ook een PerAceptron voor een uitgebreider beslissysteem
# # (minimaal 3 inputs, zie bijvoorbeeld Figuur 2.8 uit de reader)
# # en test of deze naar verwachting werkt.
#
# figuur_28= pe.Code(weights= [0.6, 0.3, 0.2], inputs= inputs_3, bias=bias,t= 0.4)
# figuur_28.predict()
# print("Figuur_28",figuur_28.__str__())