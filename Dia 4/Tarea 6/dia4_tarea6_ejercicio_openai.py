"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: USO DE OPENAI CHATCOMPLETION API (GPT)
--------------------------------------------------

Descripción general:
Este script realiza una solicitud a la API de OpenAI para generar texto
utilizando un modelo de chat (gpt-3.5-turbo).

El objetivo del código es:
- Configurar una clave de autenticación para la API
- Enviar un mensaje estructurado al modelo
- Recibir una respuesta generada automáticamente
- Manejar posibles errores durante la ejecución

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. API Key:
   - Se utiliza para autenticar la solicitud hacia los servicios de OpenAI
   - Permite acceso a modelos de lenguaje alojados en la nube

2. openai.ChatCompletion.create:
   - Método que envía una conversación al modelo
   - Recibe una lista de mensajes con roles (system, user)
   - Retorna una respuesta generada por el modelo

3. Parámetros principales:
   - model: especifica el modelo de lenguaje utilizado
   - messages: historial de conversación estructurado
   - max_tokens: límite de tokens en la respuesta
   - temperature: controla la creatividad del modelo

4. Manejo de errores:
   - Se utiliza un bloque try/except para capturar excepciones
   - Permite evitar fallos inesperados en tiempo de ejecución

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Se importa la librería openai
2. Se define la API key
3. Se construye la solicitud al modelo
4. Se envía el prompt al modelo GPT-3.5 Turbo
5. Se recibe la respuesta generada
6. Se imprime el contenido del mensaje
7. Se capturan errores si la llamada falla

--------------------------------------------------
NOTA FUNCIONAL
--------------------------------------------------

- El modelo responde en formato conversacional
- El mensaje del sistema define el comportamiento del asistente
- El mensaje del usuario define la tarea solicitada

--------------------------------------------------
"""

import openai

# --------------------------------------------------
# CONFIGURACIÓN DE API KEY
# --------------------------------------------------
# Se asigna la clave de autenticación para acceder a la API de OpenAI
openai.api_key = "sk-proj-bla-bla"

try:
    # --------------------------------------------------
    # SOLICITUD DE GENERACIÓN DE TEXTO
    # --------------------------------------------------
    # Se envía una conversación estructurada al modelo GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short story about a robot learning to cook."}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # --------------------------------------------------
    # SALIDA DEL MODELO
    # --------------------------------------------------
    # Se extrae y muestra el contenido generado por el modelo
    print("Generated Text:\n", response["choices"][0]["message"]["content"].strip())

except Exception as e:
    # --------------------------------------------------
    # MANEJO DE ERRORES
    # --------------------------------------------------
    # Captura cualquier excepción ocurrida durante la ejecución de la API
    print(f"An error occured: {e}")
