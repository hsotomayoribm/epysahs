from flask import Flask, request, jsonify
import requests
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator
import pandas as pd
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

# Selección del modelo
model_id_2 = os.getenv("GRANITE")

# Selección del ID del proyecto
project_id = os.getenv("IBM_WATSON_STUDIO_PROJECT_ID")

# Selección de la api key
api_key = os.getenv("IBM_CLOUD_API_KEY")

# Selección url
url_cloud = os.getenv("IBM_CLOUD_URL")



# Autenticación        
access_token = IAMTokenManager(
    apikey = api_key,
    url = url_cloud).get_token()

import getpass

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "SJKz5Uoz1AlZhL6IKRSkPCRHVt5yvVICvH3GWNTfzoG7"
}


parametros = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 700,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

MPT_7B_INSTRUCT2 = Model(
    model_id=model_id, 
    params=parametros, 
    credentials=credentials,
    project_id=project_id)

MPT_7B_INSTRUCT2_llm = WatsonxLLM(model=MPT_7B_INSTRUCT2)

def sentencia_sql(json_data):

    json_data = request.json
    

    if not json_data or 'pregunta' not in json_data:
        return jsonify({"error": "Invalid JSON format"}), 400    

    # Generación de sentencia SQL
    pregunta_usuario = json_data['pregunta']#"¿Cuál es el cliente que ha realizado el mayor número de transacciones?"

    # recibe la informacion de la base de datos
    dataModel = json_data['dataModel']

    ejemplos = json_data['ejemplos']

    import re

    def formatear_pregunta(pregunta_usuario):
        # Eliminar espacios en blanco adicionales
        pregunta_formateada = ' '.join(pregunta_usuario.split())
    
        # Convertir a minúsculas
        pregunta_formateada = pregunta_formateada.lower()
    
        # Eliminar caracteres especiales y signos de puntuación
        pregunta_formateada = re.sub(r'[^\w\s]', '', pregunta_formateada)
    
        return pregunta_formateada

    pregunta_formateada = formatear_pregunta(pregunta_usuario)


    def extractDatabaseModelInfo(dataModel):
        database_info_text = ""

        for table_info in dataModel["tables"]:
            table_name = table_info["tableName"]
            columns_info = []

            for column_info in table_info["columns"]:
                column_name = column_info["columnName"]
                column_type = column_info["columnType"]
                column_info_text = f"{column_name} ({column_type})"
                columns_info.append(column_info_text)

            table_info_text = f"Tabla {table_name} con los siguientes campos: {', '.join(columns_info)}"
            database_info_text += table_info_text + "\n"

        return database_info_text

    # Ejemplo de uso:
    texto_descriptivo = extractDatabaseModelInfo(dataModel)

    # Inicializa una cadena vacía para almacenar el texto combinado
    texto_combinado = "ejemplos = \"\"\"\n"

    # Recorre la lista de ejemplos y agrega cada ejemplo al texto combinado
    for i, ejemplo in enumerate(ejemplos, start=1):
        pregunta = ejemplo["pregunta_usuario"]
        respuesta = ejemplo["respuesta"]
        
        # Agrega el número de ejemplo, la pregunta y la respuesta al texto
        texto_combinado += f"ejemplo {i}:\n    Pregunta usuario : {pregunta}\n    respuesta : {respuesta}\n"

    # Agrega el cierre de la cadena
    texto_combinado += "\"\"\""


    sql_template = """
    Respira profundamente y enfoquémonos.Tu rol es el de un experto en SQL y tu tarea es traducir texto a sql , debes analizar la pregunta del usuario entendiendo que te esta solicitando y cual es el proposito de la pregunta , 
    para construir la sentencia sql debes tomar en consideracion la descripción de la unica tabla y sus columnas ademas debes utilizar textualmete el nombre de las columnas porque en la base de datos estan de esa misma forma con los espacios. 
    solo devolver la sentencia sql , no repetir informacion y no inventar informacion ademas crea alias para darle mas entendimiento a la sentencia sql.
    Terminos de ciertas palabras que recibiras: SKU es la columna ITEMID , la columna CLI_FLAG_CARTERA tiene dos opciones Obejtivo o Abierta, si te preguntaran por 'los SKU mas vendido para el cliente con ' deberas usar CLI_FLAG_CARTERA=OBJETIVO; cuenado tengas que usar en las clausulas select y where el rut del cliente debes agregarlo con doble comillas asi textualmente como te enseño "RUT_Cliente" .
    Acuerdate de usar bien las clausulas where y group by cuando usas funciones.
    descripción de la unica tabla y sus columnas de la base de datos que debes usar para construir la sentencia sql : {texto}.
    Debes identificar cuales columnas serian las mas apropiadas para crear la sentencia sql , despues identificar si es necesario hacer un calculo con alguna de las columnas que seleccionaste y por ultimo agrupar o ordenar segun sea necesario.
    La columna FECHA_OV_DATE necesita ser transformada con la siguiente funcion DATE(TIMESTAMP_FORMAT(FECHA_OV_DATE, 'DD-MM-YYYY')) para obtener la fecha de esta forma YYYY-MM-DD , usa la funcion para transformar la columna CADA VEZ QUE LA USES y poder trabajar con esa columna.
    Utiliza los ejemplos solo de guia para tus respuestas y saber como esperamos que construyas la sentencia sql nunca debes usar los valores de los ejemplos en tu respuesta.
    {ejemplos}
        
    pregunta del usuario que debes responder :{pregunta_usuario}
    respira profunadamente, toma te un tiempo y enfocate en que la respuesta que necesitamos es solo la sentencia sql que responda la pregunta del usuario, sin detalles adicionales.
    respuesta: 
    """

    prompt = PromptTemplate(
        input_variables=["texto","pregunta_usuario","ejemplos"],
        template=sql_template,
    )

    chain_2 = LLMChain(llm=MPT_7B_INSTRUCT2_llm, prompt=prompt)


    query=chain_2.run({'texto':texto_descriptivo ,'pregunta_usuario':pregunta_usuario,'ejemplos':texto_combinado})

    import re

    def formatear_pregunta(pregunta_usuario):
        # Eliminar espacios en blanco adicionales
        pregunta_formateada = ' '.join(pregunta_usuario.split())
    
        # Reemplazar saltos de línea por espacios en blanco
        pregunta_formateada = pregunta_formateada.replace('\n', ' ')
    
        return pregunta_formateada

    sql_formateado = formatear_pregunta(query)


    def agregar_comillas_sql(sentencia_sql, columnas_a_modificar):
        for columna in columnas_a_modificar:
            # Utilizar expresiones regulares para agregar doble comillas a la columna
            sentencia_sql = re.sub(rf'\b({columna})\b', r'"\1"', sentencia_sql)
        return sentencia_sql

    columnas_a_modificar = ['Cant_OV', 'Precio_Unitario_Venta' , 'CONTRIBUCIÓN', 'Facturación_Comuna']

    sql_modificado = agregar_comillas_sql(sql_formateado, columnas_a_modificar)
    print(sql_modificado)

    import pandas as pd
    import os, ibm_db, ibm_db_dbi as dbi, pandas as pd


    DB2DWH_dsn = 'DATABASE={};HOSTNAME={};PORT={};PROTOCOL=TCPIP;UID={uid};PWD={pwd};SECURITY=SSL'.format(
        'bludb',
        '125f9f61-9715-46f9-9399-c8177b21803b.c1ogj3sd0tgtu0lqde00.databases.appdomain.cloud',
        30426,
        uid='zkd63801',
        pwd='Huemul.DB.2023+'
    )

    # Establecer la conexión a la base de datos
    try:
        conn = dbi.connect(DB2DWH_dsn)
        cursor = conn.cursor()


        # Realizar la consulta y almacenar los resultados en un DataFrame
        tabla_sales = pd.read_sql(sql_modificado, conn)

        # Imprimir el DataFrame
        #print(tabla_sales)

        # Convertir el DataFrame a formato JSON
        datos = tabla_sales.to_json(orient='index')

        import json
        from datetime import datetime

        # Parsear el JSON a un diccionario de Python
        data = json.loads(datos)

        # Inicializar una cadena de texto vacía para almacenar la información
        informacion_texto = ""

        # Recorrer el diccionario y agregar la información en formato de texto lineal
        for key, value in data.items():
            
            for subkey, subvalue in value.items():
                if subkey == "DATE_SELL":
                    fecha_formateada = datetime.utcfromtimestamp(subvalue / 1000).strftime('%Y-%m-%d')
                    informacion_texto += f" {subkey} : {fecha_formateada}, "
                else:
                    informacion_texto += f" {subkey} : {subvalue}, "
            informacion_texto = informacion_texto.rstrip(', ') + "\n"


        # Imprimir o usar la cadena de texto según tus necesidades
        #print(informacion_texto)
        response_data = {
                        'sentencia SQL': sql_modificado,
                        'resultado SQL': informacion_texto,
                        }

        return jsonify(response_data)

    except Exception as e:
        # Manejar cualquier excepción que pueda ocurrir
        print(f"Se produjo un error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Cerrar el cursor y la conexión, independientemente de si hay una excepción o no
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        if 'conn' in locals() and conn is not None:
            conn.close()




    