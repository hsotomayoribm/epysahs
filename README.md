# Instrucciones previas para el deploy
### Loguearse en Container Registry de IBM
```
docker login us.icr.io -u iamapikey -p bNSxSNvzhZE8ZSypRaghbhnVWg3JCdi2LOKyVusoeXKh
```

### Tagear imagen en repo de IBM Container Regsitry
```
docker tag app3:latest us.icr.io/huemulnamespace/codeengine-app3-dc:latest
```

### Subir imagen local a repositorio del Container Registry
```

docker push us.icr.io/huemulnamespace/codeengine-app3-dc:latest
```

### Crear aplicación en Code Engine a partir de Container Image.
- Apuntar a
```
private.us.ic

# Traduccion de lenguaje natural a SQL con una respuesta humanizada

Este repositorio implementa un proceso para traducir preguntas en lenguaje natural a consultas SQL y proporcionar respuestas humanizadas. Cada ruta en el proceso recibe datos en formato JSON para llevar a cabo sus funciones específicas.

## Rutas y Funciones

### Ruta 1: Extracción de Entidades

**Descripción:** Esta ruta se encarga de extraer las entidades relevantes de la pregunta del usuario.

**Información necesaria:**

{
  "pregunta": "Cuál es el total de ventas hasta la fecha"
}

**Resultado:**

{
  entidad:total de ventas, fecha: hasta la fecha"
}


### Ruta 2: Clasificación de la Pregunta

**Descripción:** En esta ruta, la pregunta del usuario se clasifica para determinar su validez.

La segunda ruta es la "Clasificación de la pregunta." Para utilizar esta función, se debe entregar la pregunta del usuario junto con las entidades extraídas del paso anterior.

**Información necesaria:**

{
  "pregunta": "Cuál es el total de ventas hasta la fecha",
  "entidades": "entidad:total de ventas, fecha: hasta la fecha"
}

**Resultado:**
*Opcion A
{
  "clasificación" : "Válida"
}

*Opcion B
{
  "clasificación" : "No Válida"
}

### Ruta 3: Creación de la Sentencia SQL

**Descripción:** Esta ruta genera la sentencia SQL para responder la pregunta del usuario. Se realiza una clasificación para determinar si la sentencia SQL es exactamente igual a los ejemplos proporcionados, si es similar, si es diferente pero produce el mismo resultado, o si simplemente no es una sentencia SQL válida.

La tercera ruta es la "Creación de la sentencia SQL para responder la pregunta del usuario." Esta ruta solo necesita la pregunta del usuario.

**Información necesaria:**

{
  "pregunta": "Cuál es el total de ventas hasta la fecha"
}
  
**Resultado:**

{
    "resultado SQL": "TOTAL_VENTAS : 3379.31\n",
    "clasificacion" :"exactamente igual"
    "sentencia SQL": "\nSELECT sum(product_price) as total_ventas FROM sales"
}

### Ruta 4: Respuesta Humanizada del Asistente

**Descripción:** La respuesta final del asistente depende de la clasificación de la pregunta. Se pueden seguir dos caminos distintos.

Estos caminos dependerán de la clasificación de la pregunta. Si la pregunta es válida, se necesita la pregunta del usuario, la clasificación y el resultado SQL para poder dar una respuesta humanizada. En el camino 2, cuando la pregunta del usuario no es válida, solo se necesita la clasificación y la pregunta del usuario.

Camino 1.


**Información necesaria:**

{
  "pregunta": "Cuál es el número de ventas realizadas por cada vendedor de la tienda",
  "clasificación": "Válida",
  "resultado SQL": "SELLER_ID : 101,  VENTAS : 7\n SELLER_ID : 102,  VENTAS : 4\n SELLER_ID : 103,  VENTAS : 5\n SELLER_ID : 104,  VENTAS : 4"
}

**Resultado:**

{
  "Respuesta humanizada": "El número de ventas del vendedor con ID 101 es de 7 ventas, el vendedor con ID 102 tiene 4 ventas, el vendedor con ID 103 tiene 5 ventas y el vendedor con ID 104 tiene 4 ventas."
}

Camino 2.


**Información necesaria:**

{
  "pregunta": "¿Cuántos años tienes?",
  "clasificación": "No Válida"
}

**Resultado:**

{
  "Respuesta humanizada": "Soy un chatbot, por lo cual no tengo edad. Puedo ayudarte en alguna otra cosa."
}

### Ruta 5: Categorizar y crear una descripcion de las columnas de la base de datos

**Descripción:** En esta ruta, se define la categoria a la cual pertenece la informacion de la columna mas los datos de dicha columna y se crea un descripcion de la columna .

La quinta ruta es la "Categorizar y descripcion" Para utilizar esta función, se debe entregar el nombre de la columna la cual es obligatoria mas los datos o registors o filas de la columna para que el modelo pueda crear una descripcion de dicha columna.

**Información necesaria:**

{
  "nombre_columna": "RUT Cliente ",
  "datos": "1202934-4 "
}

**Resultado:**

{
  "Categoria": "Cliente",
  "Descripcion": "El conjunto de datos indica el rut del cliente y sirve para identificar al cliente."
}
