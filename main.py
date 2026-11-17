# pendencias:
# trocar o dataset por um de cameras de seguranca
# ultima atualizacao: 17/11 por arthur

import streamlit
import cv2
from ultralytics import YOLO
import imageio
from collections import deque
import os

# se nao existir, cria uma pasta gravacoes pra salvar os videos
os.makedirs("gravacoes", exist_ok=True)

FPS = 12 # ajustar conforme testes

streamlit.set_page_config(layout="wide")
model = YOLO("runs/detect/dataset/runs/Deteccao_de_armas-8/weights/best.pt")

def fechar_video():
    streamlit.session_state.video_salvo = False

# salvando na session state
if "buffer" not in streamlit.session_state:
    streamlit.session_state.buffer = deque(maxlen=FPS * 15)
buffer = streamlit.session_state.buffer

if "video_salvo" not in streamlit.session_state:
    streamlit.session_state.video_salvo = False

# sidebar
with streamlit.sidebar:
    streamlit.title("Controles")
    ligada = streamlit.toggle("on/off camera", value=True)
    habilitar_salvamento = streamlit.toggle("habilitar salvamento")
    salvar = streamlit.button("salvar últimos 15 segundos", disabled=not habilitar_salvamento)

if not habilitar_salvamento:
    buffer.clear()

# colunas
camera, dados = streamlit.columns(2, border=True)
with camera:
    streamlit.title("Vídeo")
    badge = streamlit.empty()
    frame_window = streamlit.empty()

with dados:
    streamlit.title("Dados")
    texto_arma = streamlit.empty()

# criando area de video salvo
area_video = streamlit.empty()

if salvar and len(buffer) > 0: # se clicar no botao salvar e tiver algum frame:
    imageio.mimsave("gravacoes/ultimos_15s.mp4", list(buffer), fps=FPS)
    streamlit.session_state.video_salvo = True

if streamlit.session_state.video_salvo: # assim que tiver um video salvo, aparece
    with area_video.container():
        streamlit.subheader("Gravação salva")
        streamlit.video("gravacoes/ultimos_15s.mp4")
        streamlit.button("Fechar vídeo", on_click = fechar_video)
else:
    area_video.empty()

# se o toggle de camera estiver habilitado:
if ligada:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        badge.markdown(":red-badge[● AO VIVO]")
        ok, frame = cap.read()
        if not ok:
            break

        resultados = model(frame, conf=0.75, verbose=False)
        frame_anotado = resultados[0].plot()
        frame_anotado = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
        if habilitar_salvamento:
            buffer.append(frame_anotado)

        frame_window.image(frame_anotado, width="stretch")

        deteccoes = []
        for caixa in resultados[0].boxes:
            deteccoes.append({
                "tipo": model.names[int(caixa.cls[0])],
                "confiança": float(caixa.conf[0]) * 100,
            })
        texto_arma.dataframe(
            deteccoes,
            hide_index=True,
            column_config={
                "confiança": streamlit.column_config.ProgressColumn(
                    "Confiança",
                    min_value=0.75 * 100,
                    max_value=100,
                    format="%.0f%%",
                )
            },
        )

    cap.release()
else:
    badge.markdown(":gray-badge[● CAMERA DESLIGADA]")