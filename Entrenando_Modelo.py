#---- Importar las librerias----
import cv2
import numpy as np
import os

#Importamos las fotos tomadas anteriormente
direccion = 'D:/Clase2/Fotos'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
cont = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir #leer las fotos de los rostros

    for fileName in os.listdir(nombre):
        etiquetas.append(cont) #asignamos las etiquetas
        rostros.append(cv2.imread(nombre + '/' + fileName, 0) )

    cont = cont + 1

#----Creamos el modelo-----
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#----Entrenamos el modelo----
reconocimiento.train(rostros, np.array(etiquetas))

#----Guardamos el modelo----
reconocimiento.write("ModeloEntrenado.xml")
print("Modelo creado")
