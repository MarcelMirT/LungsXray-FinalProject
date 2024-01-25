import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

def predict_imagen(imagen):
    imagen = imagen.reshape((1, 150, 150, 3))
    model_covid = load_model('Modelos_binarios/covid')
    model_sano = load_model('Modelos_binarios/normal')
    model_pneumonia = load_model('Modelos_binarios/pneumonia')
    model_tuberculosis = load_model('Modelos_binarios/tuberculosis')

    pred_covid = model_covid.predict(imagen)[0][0]
    pred_sano = model_sano.predict(imagen)[0][0]
    pred_pneumonia = model_pneumonia.predict(imagen)[0][0]
    pred_tuberculosis = model_tuberculosis.predict(imagen)[0][0]

    #Invierte las probabilidades

    pred_covid = 1 - pred_covid
    pred_sano = 1 - pred_sano
    pred_pneumonia = 1 - pred_pneumonia
    pred_tuberculosis = 1 - pred_tuberculosis

    # Crea un diccionario de predicciones
    predictions = {
        'COVID': pred_covid,
        'SANO': pred_sano,
        'PNEUMONIA': pred_pneumonia,
        'TUBERCULOSIS': pred_tuberculosis
    }
    return predictions

def main():
    st.markdown("<h1 style='text-align: center; color: black; font-size: 40px;'>Servicio de Neumología de 4geeks</h1>", unsafe_allow_html=True)
    st.write('**Por favor tenga sus radiografias a mano (en el ordenador)**')
    st.write('**Las radigrafías deben de ser frontales**')

    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    #Carga el modelo
    
    
    if uploaded_file is not None:
        #Augmenta el brillo y el contraste de la imagen añadida
        # Convert image to RGB mode if not already in RGB
       #if uploaded_file.mode != "RGB":
        #    uploaded_file = uploaded_file.convert("RGB")

        image = Image.open(uploaded_file)
        image = ImageEnhance.Brightness(image).enhance(0.7)
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image.thumbnail((299, 299), Image.BICUBIC)  # Use BICUBIC instead of ANTIALIAS


        if image is not None:
            image = Image.open(uploaded_file).resize((150, 150))
            st.image(image, caption='Imagen cargada', use_column_width=True)

            image = np.array(image) / 255.0

            if st.button('Realizar Predicción de la categoría de la imagen'):
                pred = predict_imagen(image)

                # Ordena las predicciones por su valor de confianza
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                top_prediction, top_confidence = sorted_predictions[0]
                second_prediction, second_confidence = sorted_predictions[1]
                # Verifica si la mejor predicción tiene una confianza >= 90%
                if top_confidence >= 0.9:
                    st.success(‘Éxito al realizar la predicción!‘)
                    st.write(f’La categoría predicha para la imagen es **{top_prediction}** con una confianza del {top_confidence * 100:.2f}%.’)
                    st.write(‘Por favor contraste los resultados con un profesional’)
                else:
                    st.warning(‘Predicciones múltiples debido a confianza baja:‘)
                    st.write(f'1. **{top_prediction}**: {top_confidence * 100:.2f}%‘)
                    st.write(f'2. **{second_prediction}**: {second_confidence * 100:.2f}%’)
                    st.write(‘Por favor contraste los resultados con un profesional’)
    
if __name__ == "__main__":
    main()