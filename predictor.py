from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

import warnings
#No suele ser buena práctica, lo hago por ser un proyecto de prueba.
warnings.filterwarnings("ignore")

def predictor(ruta_imagen, modelo):
    """
    Función que recibe la ruta de la imagen y un modelo de clasificación y genera
    la predicción.

    Args:
        ruta_imagen (str): Ruta a la imagen a analizar.
        modelo (keras Model): Modelo Keras a usar para analizar la imagen.

    Returns:
        ix (int): Índice en la lista de clasificación correspondiente al output del modelo.
                    Ej: ix = 3 indica que el modelo predice que la imagen pertenece a la clase
                        3 de la lista de clasificación en la que esté basado el modelo.    
    """
    
    archivo = ruta_imagen
    imagen2 = load_img(archivo, target_size=(224,224))
    img2 = img_to_array(imagen2)
    img2 = img2.reshape(1,224,224,3)/255
    pred = modelo.predict(img2)
    ix = np.argmax(pred)

    return ix