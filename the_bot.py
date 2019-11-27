import telegram
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)

import logging

import warnings
#No suele ser buena práctica, lo hago por ser un proyecto de prueba.
warnings.filterwarnings("ignore")

import os
TOKEN = os.environ['TGM_BOT_TEST']

from predictor import predictor
from clases_imagenet import clases
from mapas_atencion import generar_img_atencion

from tensorflow.keras.models import load_model


#Verificamos que estamos conectados
bot = telegram.Bot(token=TOKEN)
print(bot.get_me())

#Generamos los objetos generales de trabajo:
updater = Updater(token=TOKEN, use_context=True)
dispatcher = updater.dispatcher

#Log de rutina, tomado de la documentación:
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

#Comenzamos la aplicación:
FOTO = 1

def cancel(update, context):
    """
    Función genérica para terminar la interacción con el bot.
    """

    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('¡Hasta luego!')

    return ConversationHandler.END

################ INIT CONVERSATION CLASSIFICATION ###################################

#Llamamos al modelo descargado de MobileNetV2
modelo = load_model('modelos/clasificador.hdf5')

def start_clas(update, context):
    """
    Función para manejar el inicio de la conversación a través del handler. Saluda y solicita la imagen a analizar"
    """

    user = update.message.from_user
    update.message.reply_text("¡Hola {}! Envíame una foto, intentaré decirte el objeto que aparezca (en inglés).".format(user.first_name))

    return FOTO

def imagen_clas(update, context):
    """
    Función que:
        - Continua la conversación desde start.
        - Solicita la imagen a clasificar.
        - Envía el resultado de la clasificación.
        - Envía el mapa de atención de la clasificación.
    """              
    #Obtenemos la imagen enviada por el usuario:
    photo_file = update.message.photo[-1].get_file()
    update.message.reply_text('Imagen recibida. Dame un momento para procesarla.')
    
    #Descargamos la imagen al disco con un nombre genérico (esto es mejorable):
    photo_file.download('user_photo.jpg')

    #Obtenemos el índice de clasificación:
    ix = predictor('user_photo.jpg',modelo)

    #Obtenemos la clase predicha al ubicar su índice en el diccionario de clases:
    pred = clases[ix]

    #Generamos el mapa de atención:
    generar_img_atencion('user_photo.jpg',modelo)

    #Enviamos el mapa de atención y finalizamos la conversación:
    update.message.reply_text("La predicción fue {}. Te muestro una imagen de lo que más me llamó la atención para clasificarla. ¡Hasta Luego!".format(pred))
    bot.send_photo(chat_id=update.message.chat_id, photo=open('mapa_atencion.jpg', 'rb'))

    return ConversationHandler.END

#Creamos el handler de la conversación para clasificación:
conv_handler_clas = ConversationHandler(
    entry_points = [CommandHandler('clas', start_clas)],

    states = {
        FOTO: [MessageHandler(Filters.photo, imagen_clas),
                CommandHandler('cancel', cancel)]
    },

    fallbacks = [CommandHandler('cancel', cancel)]
)

#Añadimos la conversación al dispatcher:
dispatcher.add_handler(conv_handler_clas)

################ END CONVERSATION CLASSIFICATION###################################


################ INIT CONVERSATION STYLE ###################################
import tensorflow_hub as hub
from transferir_estilo import transferir

#Descargamos el modelo que realiza la transferencia de estilo (es posible que se pueda descargar)
hub_module = hub.load('style_folder')

def start_estilo(update, context):
    """
    Función para manejar el inicio de la conversación de transferencia de estilo a través del handler. Saluda y solicita la imagen a modificar"
    """

    user = update.message.from_user
    update.message.reply_text("¡Hola {}! Envíame una foto, intentaré cambiarle el estilo".format(user.first_name))

    return FOTO

def imagen_estilo(update, context):
    """
    Función que:
        - Continua la conversación desde start.
        - Solicita la imagen a la que aplicar estilizado.
        - Envía la imagen con el estilo cambiado.
    """   

    #Obtenemos la imagen enviada por el usuario:              
    photo_file = update.message.photo[-1].get_file()
    update.message.reply_text('Imagen recibida. Dame un momento para procesarla.')

    #Descargamos la imagen al disco con un nombre genérico (esto es mejorable):
    photo_file.download('user_photo.jpg')

    #Hacemos la transferencia de estilo y guardamos la imagen con un nombre generico (esto es mejorable)
    transferir(hub_module,'user_photo.jpg')

    #Enviamos la imagen con el estilo transferido y nos despedimos:
    update.message.reply_text("Aquí tienes la imagen estilizada, increíble ¿no?. ¡Hasta pronto!")
    bot.send_photo(chat_id=update.message.chat_id, photo=open('estilizado.jpg', 'rb'))

    return ConversationHandler.END

#Creamos el handler de la conversación para transferencia de estilo:
conv_handler_style = ConversationHandler(
    entry_points = [CommandHandler('style', start_estilo)],

    states = {
        FOTO: [MessageHandler(Filters.photo, imagen_estilo),
                CommandHandler('cancel', cancel)]
    },

    fallbacks = [CommandHandler('cancel', cancel)]
)

#Añadimos la conversación al dispatcher:
dispatcher.add_handler(conv_handler_style)

################ END CONVERSATION STYLE ###################################

#Activamos al bot para escuchar:
updater.start_polling()