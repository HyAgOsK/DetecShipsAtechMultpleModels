import cv2
import numpy as np
import tensorflow as tf

# Função para carregar o modelo TFLite
def load_model_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Função para realizar a detecção de objetos em uma imagem
def detect_objects(image, interpreter, input_details, output_details, threshold=0.5):
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
    image_input = np.expand_dims(image_resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    for i in range(num_detections):
        if scores[0, i] > threshold:
            box = boxes[0, i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            y_min, x_min, y_max, x_max = box.astype(np.int)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            class_id = int(classes[0, i])
            cv2.putText(image, f"Class {class_id}, {scores[0, i]}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main():
    # Carregar o modelo TFLite
    model_path = "yolov5s-fp16.tflite"  # Defina o caminho para o modelo TFLite
    interpreter, input_details, output_details = load_model_tflite(model_path)

    # Carregar a imagem
    image_path = "example.jpg"  # Defina o caminho para a imagem de entrada
    image = cv2.imread(image_path)

    # Realizar a detecção de objetos
    result_image = detect_objects(image, interpreter, input_details, output_details)

    # Exibir a imagem com os objetos detectados
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
