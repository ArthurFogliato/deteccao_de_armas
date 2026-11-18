from streamlit_webrtc import webrtc_streamer #video
from vidgear.gears import Camgear            #video          
from ultralytics import YOLO
import streamlit as st
import yt_dlp                                #baixar video do youtube //possívelmente remover
import cv2
import os


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


#Pegar video----
stream= Camgear(source = "https://www.youtube.com/watch?v=vMF-BfwQyZM", stream_mode = True, logging = True  )
ydl_opts = {'format': 'best[ext=mp4]'}


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
 info = ydl.extract_info(video_url, download=False)
 stream_url = info['url']
#----------------


#Configuaracao modelo-----
model = YOLO("yolov8n.pt") #trocar pelo modelito certo
cap = cv2.VideoCapture(stream_url)


#def label():
 


frame_window = st.image([])
#----------------


#Configuracao pagina-----


st.set_page_config(layout="wide")


camera, dados = st.columns(2, border=True)


#------------------------


#Pagina principal---
with camera:
   st.title("Vídeo")
   st.empty()
      
with dados:
   st.title("Dados")
   st.empty("Tipo de arma:")
#-------------------





