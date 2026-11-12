#Link do modelo utilizado-----
#Link: https://universe.roboflow.com/marcel-gigena-qbwck/gun-detection-uoebv/
#-----------------------------


import streamlit
import cv2
from ultralytics import YOLO

streamlit.set_page_config(layout="wide")

#Configuaracao modelo-----
model = YOLO("best.pt") #trocar pelo modelito certo
cap = cv2.VideoCapture(0)
#def label():
frame_window = streamlit.image([])
#-------------------------

#Configuracao pagina-----
camera, dados = streamlit.columns(2, border=True)
#------------------------
#Pagina principal--------
with camera:
    streamlit.title("Vídeo")
    frame_window.image(frame)
        
with dados:
    streamlit.title("Dados")
    streamlit.text("Tipo de arma:")
#------------------------

