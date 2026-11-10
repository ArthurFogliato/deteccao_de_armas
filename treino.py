import os
from ultralytics import YOLO
import torch

#documentação de referência
#GPU:https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html
#YOLO Ultralitics:
#--------------------------


BASE = './dataset'

yaml_path = os.path.join(BASE, 'data.yaml')
runs_dir = os.path.join(BASE, 'runs')

if not os.path.exists(yaml_path):
    print("Não foi encontrado o arquivo %s", yaml_path);
    exit(0)



print("Treinamento YOLO")

model = YOLO('yolo26n.pt')

model.train(
    data = yaml_path,
    epochs = 80,
    imgsz = 320,
    patience = 20,
    device = 'cpu',
    project = runs_dir,
    name = 'Deteccao_de_armas',
    save = True
)

print("Acabou o treinamento.")

best_weights = os.path.join(runs_dir, "/weights")

if not os.path.exists(best_weights):
    print("Não foi encontrado o arquivo %s", best_weights);
    exit(0)

model_trained = YOLO(best_weights)

model_trained.export(format='ncnn', imgsz=320)

print("Feitoria!!!")