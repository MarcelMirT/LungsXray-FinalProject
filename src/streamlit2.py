import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

def predict_imagen(imagen, model):
    imagen = imagen.reshape((1, 150, 150, 3))
    predictions = model.predict(imagen)
    predicted_class = tf.argmax(predictions[0]).numpy()
    class_names = ['COVID', 'SANO', 'PNEUMONIA', 'TUBERCULOSIS']
    return class_names[predicted_class]

def main():
    st.title('Bienvenido al servicio de neumología de 4geeks')
    st.write('**Por favor tenga sus radiografias a mano (en el ordenador)**')
    
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    #Carga el modelo
    model = load_model('/workspaces/LungsFPruebaStreamlit/models/modelo_softmax.keras')

    if uploaded_file is not None:

        #Augmenta el brillo y el contraste de la imagen añadida
        # Convert image to RGB mode if not already in RGB
        if uploaded_file.mode != "RGB":
            uploaded_file = uploaded_file.convert("RGB")

        image = Image.open(uploaded_file)
        image = ImageEnhance.Brightness(image).enhance(0.7)
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image.thumbnail((299, 299), Image.BICUBIC)  # Use BICUBIC instead of ANTIALIAS



        if image is not None:
            image = Image.open(uploaded_file).resize((150, 150))
            st.image(image, caption='Imagen cargada', use_column_width=True)

            image = np.array(image) / 255.0

            if st.button('Realizar Predicción de la categoría de la imagen'):
                pred = predict_imagen(image, model)
                st.success('Éxito al realizar la predicción!')
                st.write('La categoría predicha para la imagen:')
                st.write(pred)
    
if __name__ == "__main__":
    main()