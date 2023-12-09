from flask import Flask, request, jsonify
import requests
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator
import ibm_db_dbi as dbi
import os
from prompt.prompt import Prompt
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




app = Flask(__name__)

# Selección del modelo
model_id = os.getenv("MPT")

# Selección del ID del proyecto
project_id = os.getenv("IBM_WATSON_STUDIO_PROJECT_ID")

# Selección de la api key
api_key = os.getenv("IBM_CLOUD_API_KEY")

# Selección url
url_cloud = os.getenv("IBM_CLOUD_URL")

mi_apikey = os.getenv("mi_apikey")

url = os.getenv("url")

model_id_3 = ModelTypes.MPT_7B_INSTRUCT2

   

# Autenticación        
access_token = IAMTokenManager(
  apikey = api_key,
  url = url_cloud).get_token()

import getpass

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "SJKz5Uoz1AlZhL6IKRSkPCRHVt5yvVICvH3GWNTfzoG7"
}

# Parámetros
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 700,
    "repetition_penalty": 1
}

parametros = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

MPT_7B_INSTRUCT2 = Model(
    model_id=model_id_3, 
    params=parametros, 
    credentials=credentials,
    project_id=project_id)

MPT_7B_INSTRUCT2_llm = WatsonxLLM(model=MPT_7B_INSTRUCT2)



def extraccion_entidades(json_data):

    json_data = request.json
    

    if not json_data or 'pregunta' not in json_data:
        return jsonify({"error": "Invalid JSON format"}), 400    

    # Generación de sentencia SQL
    pregunta_usuario = json_data['pregunta']#"¿Cuál es el cliente que ha realizado el mayor número de transacciones?"

    ejemplos = json_data['ejemplos']


    import re

    def formatear_pregunta(pregunta_usuario):
        # Eliminar espacios en blanco adicionales
        pregunta_formateada = ' '.join(pregunta_usuario.split())
    
        # Convertir a minúsculas
        pregunta_formateada = pregunta_formateada.lower()
    
    
        return pregunta_formateada

    pregunta_formateada = formatear_pregunta(pregunta_usuario)
    print(f"Pregunta formateada: {pregunta_formateada}")

    # Inicializa una cadena vacía para almacenar el texto combinado
    texto_combinado = ""

    # Recorre la lista de ejemplos y agrega cada ejemplo al texto combinado
    for i, ejemplo in enumerate(ejemplos, start=1):
        pregunta = ejemplo["pregunta_usuario"]
        respuesta = ejemplo["respuesta"]
        
        # Agrega el número de ejemplo, la pregunta y la respuesta al texto
        texto_combinado += f"\nejemplo {i}:\n    Pregunta usuario : {pregunta}\n    respuesta : {respuesta}\n"

    # Agrega el cierre de la cadena
    texto_combinado += "\"\"\""
    
    
    extraccion_template = """
    Actúa como un webmaster que debe extraer información estructurada de textos en español. Lee el siguiente texto y extrae del texto las entidades , fechas , categorias , condicion y valor que sean mencionados y estan presente en el texto.
    Utiliza los ejemplos proporcionados solo como guía para seguir la estructura de extracción,pero no incluyas la información de los ejemplos en la respuesta , si recibes la misma pregunta de los ejemplos debes usar la respuesta del ejemplo.
    solo devuelve una unica respuesta y no entregues información adicional.
    Extrae Entidad: Representa un objeto o sujeto principal en la pregunta. Puede ser un cliente, producto, vendedor, etc.
    Extrae la Fecha si esta textualmente en el texto: Indica un período de tiempo relevante en la pregunta, como un año (2023), un mes (junio), o una referencia temporal (el año pasado).
    Extrae Categorías: Describe una característica específica de las entidades en cuestión (como "Herramientas" y "Suspensión") .
    Extrae Condición: Se refiere a una restricción o requisito específico asociado a la pregunta. (Por ejemplo, "más comprados," "trabajado menos," "más experiencia," o "por cada vendedor") son condiciones que limitan o califican la respuesta deseada.
    Extrae Valor: Representa la cantidad o medida que se busca en la pregunta. Puede ser un número (5), una descripción cualitativa (como "total de ventas" ) , o un estado (como "verde").
    Ejemplos:{ejemplos}
    texto para extraccion : {texto}
    Respuesta:
    """

    prompt = PromptTemplate(
        input_variables=["ejemplos","texto"],
        template=extraccion_template,
    )

    chain = LLMChain(llm=MPT_7B_INSTRUCT2_llm, prompt=prompt)
    p=prompt.format(ejemplos=texto_combinado, texto=pregunta_formateada)
    print(p)

    extraccion_2=chain.run({'ejemplos': texto_combinado, 'texto': pregunta_formateada})

    
    
    response_data = {'Entidades ':extraccion_2 }



    return response_data








