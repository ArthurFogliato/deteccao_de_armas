# pendencias:
#  - está salvando os ultimos 15 segundos, a ideia é clicar em salvar(na sidebar) e ficar disponivel
#  um video dos ultimos 15 segundos, sendo possivel pausar e dar play/voltar
#  - trazer o modelo treinado com fine tunning para as pastas e fazer o main utilizar ele, e
#  não o yolo26 padrão. para isso, é preciso organizar as pastas do modelo junto ao projeto
# ultima atualizacao: sexta feira meio dia por arthur



import streamlit
import cv2
from ultralytics import YOLO

#do python pra salvar os ultimos 15 segundos
from collections import deque
import tempfile

streamlit.set_page_config(layout="wide")
model = YOLO("yolo26n.pt")  #trocar pelo certo
#-------------------------------------------------
# pagina
with streamlit.sidebar:
    streamlit.title("Controles")
    ligada = streamlit.toggle("on/off camera", value=True)
    salvar = streamlit.button("Salvar últimos 15 segundos") 

## colunas
camera, dados = streamlit.columns(2, border=True)
with camera:
    streamlit.title("Vídeo")
    badge = streamlit.empty()
    frame_window = streamlit.empty()   # espaço que o loop vai atualizando

with dados:
    streamlit.title("Dados")
    
    texto_arma = streamlit.empty()

# se botao camera ligada, comeca capturar video e entra no while de mostrar
buffer = deque(maxlen=450) #parte de capturar ultimos segundos
if ligada:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        badge.markdown(":red-badge[● AO VIVO]")
        ok, frame = cap.read()
        if not ok:
            break

        buffer.append(frame.copy())
        resultados = model(frame, verbose=False)
        frame_anotado = resultados[0].plot()                            # frame com as caixas desenhadas
        frame_anotado = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR, streamlit usa RGB

        frame_window.image(frame_anotado, width="stretch")

        deteccoes = []
        for caixa in resultados[0].boxes:
            deteccoes.append({ # dicionario de python
                "tipo": model.names[int(caixa.cls[0])],
                "confiança": float(caixa.conf[0]),
            })
        texto_arma.dataframe(deteccoes, hide_index=True)

    cap.release()
else:
    badge.markdown(":gray-badge[● CAMERA DESLIGADA]")