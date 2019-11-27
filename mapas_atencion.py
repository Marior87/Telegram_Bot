import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def preprocess(archivo):
    """
    Carga y preprocesamiento de la imagen en 'archivo'
    """

    img = load_img(archivo, target_size=(224,224))
    img = img_to_array(img)/255
    return np.expand_dims(img, 0)

def get_activations_at(input_image, modelo, i):
    """
    Obtener las activaciones del modelo en la capa i. Se suele usar la última capa conv, pero aplica para cualquiera.
    """

    #index the layer 
    out_layer = modelo.layers[i]
    
    #change the output of the model 
    submodel = Model(inputs = modelo.inputs, outputs = out_layer.output)
    
    #return the activations
    return submodel.predict(input_image)

def postprocess_activations(activations):
    """
    Transforma las activaciones en un formato de imagen con el que se pueda trabajar.
    """

    #using the approach in https://arxiv.org/abs/1612.03928
    output = np.abs(activations)
    output = np.sum(output, axis = -1).squeeze()

    #resize and convert to image 
    output = cv2.resize(output, (224, 224))
    output /= output.max()
    output *= 255
    return 255 - output.astype('uint8')

def apply_heatmap2(weights, img):
    """
    Combina la imagen inicial con la imagen de las activaciones para generar el heatmap de atención.
    """
    ax = plt.imshow(weights,cmap='viridis',alpha=0.8)
    ax = plt.imshow(img,cmap='gray',alpha=0.2)
    ax = plt.axis('off')
    ax = plt.grid(False)
    plt.savefig('mapa_atencion.jpg',format='jpg',bbox_inches='tight', pad_inches = 0.0)
    #return ax

def generar_img_atencion(archivo, modelo):
    """
    Flujo para obtener el mapa de atención.
    """
    input_image = preprocess(archivo)
    activations = get_activations_at(input_image,modelo,-5)#ültima capa conv de MobileNetV2
    weights = postprocess_activations(activations)
    apply_heatmap2(weights,input_image.squeeze())