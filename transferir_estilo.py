import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image

def load_img(path_to_img):
    """
    Función que toma la ruta de una imagen cualquiera y la transforma para ser usada en el modelo de cambio de estilo.
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    """
    Función para generar una imagen desde un tensor de TF y luego guardarla en formato jpg.
    Moificada ligeramente desde la doc de TF.
    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    im = PIL.Image.fromarray(tensor)
    im.save('estilizado.jpg')


def transferir(hub_module,content_path, style_path ='estilo1.jpg'):
    """
    Función que carga el modelo de cambio de estilo* y guarda la imagen con el estilo modificado.

    Nota:   La imagen 'estilo1.jpg' fue tomada de: https://unsplash.com/photos/e5LdlAMpkEw, aun siendo
            una imagen de uso libre, no reclamo ningún tipo de propiedad sobre la misma.

    Args:
        hub_module: Modelo del TFHub usado para transferir estilo.
        content_path (str): Ruta de la imagen a modificar.
        style_path (str): Ruta de la imagen a la que extraerle el estilo.

    Returns:
        Imagen 'estilizado.jpg' guardada en el directorio principal.
    """

    #Esta es la fuente del modelo dentro de style_folder
    #hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')


    content_image = load_img(content_path)
    style_image = load_img(style_path)

    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    tensor_to_image(stylized_image)