import streamlit as st
import yt_dlp #baixar video do youtube
import cv2
import glob
from ultralytics import YOLO

# Videos para teste do yt----
#Ladrao armamento pesado: 'https://www.youtube.com/watch?v=A46ppMPwhKE'
#Gato disparando fuzil: 'https://www.youtube.com/watch?v=vMF-BfwQyZM'
#Assassino armamento leve: 'https://www.youtube.com/watch?v=irQMaSFR6CI'
#BAZUCA: 'https://www.youtube.com/watch?v=yAJAoUQ3gyI'
#----------------------------

#Baixar do yt----
video_url = 'https://www.youtube.com/watch?v=vMF-BfwQyZM'
ydl_opts = {'format': 'best[ext=mp4]'}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
  info = ydl.extract_info(video_url, download=False)
  stream_url = info['url']
#----------------

#Configuaracao modelo-----
model = YOLO("yolov8n.pt") #trocar pelo modelito certo
cap = cv2.VideoCapture(stream_url)

def label():
   

frame_window = st.image([])
#----------------

#Configuracao pagina-----

st.set_page_config(layout="wide")

camera, dados = st.columns(2, border=True)

#------------------------

#Pagina principal---
with camera:
    st.title("Vídeo")
    st.video()
        
with dados:
    st.title("Dados")
    st.text("Tipo de arma:")
#-------------------

