#----importamos las librerias-----
import cv2
import mediapipe as mp
import os

#-----Creacion de la carpeta donde almacenaremos las fotos----
nombre = 'Tania_Con_Tapabocas'
direccion = 'D:/Clase2/Fotos'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print("Carpeta creada")
    os.makedirs(carpeta)

#Inicializamos nuestro contador
cont = 0

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


        #Eliminar el error de espejo
        frame = cv2.flip(frame, 1)

        #Eliminar el erro de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Deteccion de rostros
        resultado = rostros.process(rgb)

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
                cara = frame[yi:yf, xi:xf]

                #Redimencionar las fotos
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                #Almacenar nuestras imagenes
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(cont), cara)
                cont = cont + 1





        #Mostrar los fotogramas
        cv2.imshow("Reconocimiento facial con reconocimiento de Tapabocas", frame)
        #Leyendo una tecla del teclado
        t = cv2.waitKey(1)
        if t == 27 or cont > 300: #codigo assci de esc
            break

cap.release()
cv2.destroyAllWindows()







