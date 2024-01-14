import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
from keras.models import load_model

# https://docs.streamlit.io/library/get-started/multipage-apps
# Local: streamlit run aplicacion_streamlit.py
# Streamlit Sharing 
# render, heroku, AWS EC2
def predict_imagen(imagen):
    # Añadir una dimensión extra (lote)
    imagen = imagen.reshape((1, 150, 150, 3))
    # Cargar el modelo desde el archivo
    model = load_model('models/modelo_softmax.keras')
    # Realizar la predicción
    predictions = model.predict(imagen)
    predicted_class = tf.argmax(predictions[0]).numpy()
    # Obtener el nombre de la clase predicha
    class_names = ['COVID', 'SANO', 'PNEUMONIA', 'TUBERCULOSIS']
    return class_names[predicted_class]

def main():
    # Bienvenida y selección del servicio
    st.title('Bienvenido al servicio de neumología de 4geeks')
    st.write('**Por favor tenga sus radiografias a mano (en el ordenador)**')
    # Redirigir a la página de predicción de imagen
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Mostrar la imagen
        st.write('**Vista Previa de la imagen cargada:**')
        image = Image.open(uploaded_file).resize((150, 150))
        st.image(image, caption='Imagen cargada', use_column_width=True)

    # Convertir la imagen a una matriz de valores de píxeles
    image = np.array(image) / 255.0  # Normalizar los valores de píxeles
    
    # Botón para realizar la predicción con las columnas seleccionadas
    if st.button('Realizar Predicción de la categoría de la imagen'):
        # Predecirla
        pred = predict_imagen(image)

        # Mostrar los resultados de la predicción
        st.success('Éxito al realizar la predicción!')
        st.write('La categoría predicha para la imagen:')
        st.write(pred)
    
if __name__ == "__main__":
    main()