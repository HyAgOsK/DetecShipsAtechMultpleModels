import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
from tracker import *


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
  
        p{
            font-size: 22px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


cfg_model_path = 'models/best.pt'
model = None
confidence = .25
tracker = Tracker()
global_ids_list = []

def image_input(data_src):
    st.markdown("### Detecção de embarcações em imagens")

    img_file = None
    if data_src == 'Dado de amostra':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("➤➤➤➤➤   ➤➤➤➤➤", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Fazer upload de imagem", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Imagem selecionada")
        with col2:
            img, class_name, id = infer_image(img_file)
            st.image(img, caption="Predição da imagem")
            if class_name == '':
                st.markdown(f'### Objetos detectados:\n #### Nenhum objeto detectado')
            else:
                st.markdown(f'### Objetos detectados:\n {class_name}')
                st.markdown(f'Total de objetos: {len(id)}')

                

def video_input(data_src):
    st.markdown("### Detecção de embarcações em vídeo")

    vid_file = None
    if data_src == 'Dado de amostra':
        vid_file = "data/sample_videos/yacht2.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Fazer upload de um vídeo", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Tamanho customizável")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        class_name = 0
        st1, st2, st3, st4 = st.columns(4)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        with st4:
            st.markdown(' ## Objetos Detectados \n')
            st4_text = st.markdown(f"#### {class_name}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Fim do vídeo? Saíndo ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, class_name, _ = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"### **{height}**")
            st2_text.markdown(f"### **{width}**")
            st3_text.markdown(f"### **{fps:.2f}**")
            if class_name == '':
                st4_text.markdown(f"### **Nenhum objeto detectado**")
            else:
                st4_text.markdown(f"### {class_name}")

        cap.release()


def camera_input(camera):
        
        st.markdown("### Detecção de embarcações em tempo realamente")
        cap = cv2.VideoCapture(camera)
        custom_size = st.sidebar.checkbox("Tamanho customizável")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        class_name = "Nenhum objeto detectado"
        st1, st2, st3, st4 = st.columns(4)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        with st4:
            st.markdown(' ## Objetos Detectados \n')
            st4_text = st.markdown(f"{class_name}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Não existe mais frames, fim da captura. Saíndo ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, class_name, _ = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"### **{height}**")
            st2_text.markdown(f"### **{width}**")
            st3_text.markdown(f"### **{fps:.2f}**")
            st4_text.markdown(f"### **{class_name}**")

        cap.release()

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)

    global global_ids_list

    list_ = []
    for index, row in result.pandas().xyxy[0].iterrows():
        x1=int(row['xmin']) 
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        b=str(row['name'])
        list_.append([x1,y1,x2,y2])

    boxes_ids = tracker.update(list_)
    global_ids_list.extend([box_id[-1] for box_id in boxes_ids])
    result.render()
    image = Image.fromarray(result.ims[0])

    class_names = result.names

    object_classes = {}
    for detection in result.xyxy[0]:
        class_index = int(detection[5]) 
        class_name = class_names[class_index]  
        object_classes[class_name] = object_classes.get(class_name, 0) + 1 

    class_names_list = list(object_classes.keys())
    counts_list = list(object_classes.values())

    for  i in range(len(class_names_list)):
        class_names_list[i] += " " + str(counts_list[i]) + str("\n")
    
    class_names_result = "\n".join(class_names_list)
    
 

    
    return image, class_names_result, global_ids_list
    

@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Tipo", ["Enviar arquivo", "url"])
    model_file = None
    if model_src == "Enviar arquivo":
        model_bytes = st.sidebar.file_uploader("Enviar modelo", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file
    
def main():
    global model, confidence, cfg_model_path

    st.title("Dashboard")

    st.sidebar.title("Configurações")
    
    model_src = st.sidebar.radio("Selecione o tipo de arquivo de pesos do yolov5", ["Modelo padrão yolov5n", "Faça upload do seu modelo"])

    if model_src == "Faça upload do seu modelo":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    if not os.path.isfile(cfg_model_path):
        st.warning("Modelo não disponível!! por favor faça o upload do seu modelo.", icon="⚠️")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Selecione o dispositivo para inferência", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Selecione o dispositivo para inferência", ['cpu', 'cuda'], disabled=True, index=0)

        model = load_model(cfg_model_path, device_option)

        confidence = st.sidebar.slider('Confiança', min_value=0.1, max_value=1.0, value=.45)

        if st.sidebar.checkbox("Classes customizadas"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Selecione as classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Selecione o tipo de entrada de dados: ", ['imagem', 'video', 'webcam'])

        data_src = st.sidebar.radio("Adicione seu arquivo: ", ['Dado de amostra', 'Envie seu arquivo'])

        if input_option == 'imagem':
            image_input(data_src)
        elif input_option == 'video':
            video_input(data_src)
        else:  
            camera_input(0)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
