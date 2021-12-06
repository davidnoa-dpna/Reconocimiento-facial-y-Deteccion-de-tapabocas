#----importar las librerias----
import cv2
import os
import mediapipe as mp

#----importamos los nombres de las carpetas----
direccion = 'D:/Clase2/Fotos'
etiquetas = os.listdir(direccion)
print("Nombres: ", etiquetas)

#---- Llamar el modelo entrenado----
modelo = cv2.face.LBPHFaceRecognizer_create()

#Leer el modelo
modelo.read('ModeloEntrenado.xml')

#----Declaracion del detector----
detector = mp.solutions.face_detection #Detector
dibujo = mp.solutions.drawing_utils #funcion de Dibujo

#----Realizar la VideoCaptura----
cap = cv2.VideoCapture(0)

#----Inicializamos los parametros de la deteccion----
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    #Inicializamos While True
    while True:
        #Lectura de VideoCaptura
        ret, frame = cap.read()
        copia = frame.copy()



        #Eliminar el error de espejo
        frame = cv2.flip(copia, 1)


        #Eliminar el erro de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia2 = rgb.copy()

        #Deteccion de rostros
        resultado = rostros.process(copia2)

        #Filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro)
                #print(rostro)

                #Extraemos el ancho y el alto de la ventana
                al, an, _ = frame.shape

                #Extraer x inicial & y inicial
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin

                #Extraer el ancho y el alto
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                #Coversion a pixeles
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)

                #Hallamos xfinal y yfinal
                xf = xi + ancho
                yf = yi + alto

                #Extraccion de pixeles
                cara = copia2[yi:yf, xi:xf]

                #Redimencionar las fotos
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)
                cara = cv2.cvtColor(cara, cv2.COLOR_BGRA2GRAY)

               #Realizar la prediccion
                prediccion = modelo.predict(cara)

                #Mostrar los resultados en pantalla
                if prediccion[0] == 0:
                    cv2.putText(frame, '{}'.format(etiquetas[0]), (xi, yi - 5), 1, 1.3, (255,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (255,0,0), 2)

                elif prediccion[0] == 1:
                    cv2.putText(frame, '{}'.format(etiquetas[1]), (xi, yi - 5), 1, 1.3, (0,0,255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (0,0,255), 2)





        #Mostrar los fotogramas
        cv2.imshow("Reconocimiento facial con reconocimiento de Tapabocas", frame)
        #Leyendo una tecla del teclado
        t = cv2.waitKey(1)
        if t == 27: #codigo assci de esc
            break

cap.release()
cv2.destroyAllWindows()

