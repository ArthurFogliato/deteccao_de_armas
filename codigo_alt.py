# pendencias:
# incluir streamlit
# arrumar a fonte do label
# ultima atualizacao: 18/11 por Bernardo

#Link do modelo utilizado-----
#Link: https://universe.roboflow.com/marcel-gigena-qbwck/gun-detection-uoebv/
#-----------------------------

#documentação de referência---
#GPU:https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html
#Integração:https://eranfeit.net/real-time-youtube-video-stream-extraction-and-object-detection/#Here_is_the_video_tutorial
#-----------------------------

# Videos para teste do yt----
#Ladrao armamento pesado: 'https://www.youtube.com/watch?v=A46ppMPwhKE'
#Gato disparando fuzil: 'https://www.youtube.com/watch?v=vMF-BfwQyZM'
#Assassino armamento leve: 'https://www.youtube.com/watch?v=irQMaSFR6CI'
#BAZUCA: 'https://www.youtube.com/watch?v=yAJAoUQ3gyI'
#----------------------------

import cv2 as cv
from ultralytics import YOLO
import os
from vidgear.gears import CamGear
import streamlit as st

#Pegar video----
stream = CamGear(source = "https://www.youtube.com/watch?v=vMF-BfwQyZM", stream_mode = True, logging = True  )

#Carregar modelo----
modelo = YOLO("runs.v1/detect/dataset/runs/Deteccao_de_armas-8/weights/best.pt") #modelo antigo
threshold = 0.6 #minimo para a detecção
#----------------

#Loop principal----
while True:
    frame = stream.read()

    if frame is None:
        break

    results = modelo(frame)[0]

    for result in results.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = result

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y1 = int(y2)

        if score > threshold:
            cv.rectangle(frame,(x1,y1),(x2,y2), (0, 255, 0), 1)
            cv.putText(frame, results.names[int(class_id)].upper(), (x1,y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA) 

#------------------
 
#Mostragem das imagens anotadas-----
    cv.imshow("Video",frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
#--------------------






