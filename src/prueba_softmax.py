import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import random
import os

dataset_dir = '/workspaces/LungsXray-FP/data/raw/MULTIPLE'
image_size = (150, 150)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
num_classes = len(train_generator.class_indices)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Agrega Dropout para reducir el sobreajuste
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Puedes experimentar con diferentes tasas de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# También puedes agregar ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[reduce_lr]  # Agrega ReduceLROnPlateau como callback
)
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print('\nExactitud en el conjunto de validación:', test_acc)
model.save('modelo_mejorado.keras')
model = load_model('modelo_mejorado.keras')  # Ajusta el nombre de tu modelo
# Preprocesamiento de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Ajusta si tu modelo espera etiquetas categóricas
    shuffle=False
)
# Obtener las predicciones del modelo para el conjunto de prueba
predictions = model.predict(test_generator)
# Convertir las predicciones a clases (si es necesario)
predicted_classes = np.argmax(predictions, axis=1)
# Obtener las etiquetas verdaderas
true_classes = test_generator.classes
# Calcular la precisión y el recall por clase
classification_rep = classification_report(true_classes, predicted_classes)
accuracy = accuracy_score(true_classes, predicted_classes)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Informe de clasificación:")
print(classification_rep)
print("Exactitud general:")
print(accuracy)
print("Matriz de confusión:")
print(conf_matrix)