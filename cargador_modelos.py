"""
Ejemplo de como podemos cargar y salvar diferentes modelos
"""

from tensorflow.keras.applications import MobileNetV2

modelo = MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
nombre = 'clasificador.hdf5'

modelo.save('clasificador.hdf5')