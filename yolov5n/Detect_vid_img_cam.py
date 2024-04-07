# Comando executado para fazer a detecção dos frames do vídeo para Yolov5n foi este abaixo
# Baixe algum vídeo use o drive para conexão com colaboratory, conecte-se com o drive, faça o upload do arquivo de vídeo e faça a detecção.

# Vídeo
!python /content/yolov5/detect.py --source /content/drive/MyDrive/VideoTestVessel.mp4 --weights /content/atech/yolov5/runs/train/yolov5n_results_850/weights/best.pt --conf-thres 0.45 --iou-thres 0.5

# Camera
!python /content/yolov5/detect.py --source 0 --weights /content/atech/yolov5/runs/train/yolov5n_results_850/weights/best.pt --conf-thres 0.45 --iou-thres 0.5

# Imagem 0.45 conf ou 0.25 como mostra a pasta multiplas detecções
!python /content/yolov5/detect.py --weights /content/ATECH_DEVELOPMENT/yolov5/runs/train/yolov5n_results_850/weights/best.pt --source /content/MultipleDetections/ --conf-thres 0.45 --iou-thres 0.5
