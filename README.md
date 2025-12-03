# Chat_empatico
Un chat que sea empatico

Librerias necesarias
pip install tensorflow numpy nltk streamlit pandas


Activa el entorno virtual
.\venv\Scripts\activate.bat

Una vez ejecutado el entorno, escribe en la terminal, siguiendo la ruta del proyecto en VS
python

Das enter,
luego escribes 
>>> import nltk
>>> nltk.download('punkt_tab')

1. Requisitos Previos y Preparación

    Instalar Python: Asegúrate de que tengan una versión compatible de Python (generalmente Python 3.8+).

    Descargar el Proyecto: El usuario debe descargar o clonar la carpeta completa del proyecto, que debe contener los siguientes archivos esenciales:

        train_model.py

        chat_app.py (lo que haremos en la Semana 2)

        intents.json (Tu dataset)

        model.h5 (El modelo entrenado)

        data.pickle (Los datos pre-procesados)

2. Configuración del Entorno Virtual (Recomendado)

Es crucial que el usuario trabaje dentro de un entorno virtual para evitar conflictos de librerías.

    Crear el Entorno Virtual: Abrir la terminal dentro de la carpeta del proyecto y ejecutar:
    Bash

    python -m venv venv

    Activar el Entorno:

        Windows (PowerShell): .\venv\Scripts\Activate.ps1

        macOS/Linux: source venv/bin/activate

3. Instalación de Dependencias

Una vez que el entorno virtual está activo, deben instalar las bibliotecas exactas que usaste.

    Instalar Librerías: Ejecutar el comando para instalar todas las dependencias principales:
    Bash

    pip install tensorflow numpy nltk streamlit pandas

4. Descarga de Recursos de NLTK

Dado que tuviste que descargar manualmente los datos de NLTK, es probable que la persona también necesite hacerlo para que el script train_model.py (o la parte de pre-procesamiento) funcione correctamente.

    Ingresar al Intérprete de Python: Con el entorno virtual activo, escribir python en la terminal.
    Bash

(venv) PS C:\...\> python

Descargar Recursos: Dentro del intérprete (>>>), ejecutar los comandos de descarga:
Python

    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('punkt_tab') # Importante para español
    >>> exit()
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Importante para español
exit()


5. Ejecución del Proyecto

Opción A: Usar el Modelo Entrenado (Recomendado)

Si la persona quiere usar tu IA de inmediato, simplemente debe ejecutar el archivo principal de la interfaz (que crearemos en la Semana 2):

    Ejecutar la App de Streamlit:
    Bash

    streamlit run chat_app.py

Opción B: Re-entrenar el Modelo

Si la persona quiere modificar el archivo intents.json o simplemente re-entrenar el modelo por su cuenta, debe ejecutar:

    Ejecutar el Entrenamiento:
    Bash

python train_model.py

Este comando recreará los archivos model.h5 y data.pickle.




Por otro lado, si se busca seguir mejorando...

Una vez que tengas la aplicación funcionando, el siguiente gran paso para tu requisito de "alimentación y aprendizaje" es mejorar el dataset (intents.json).

    Si la IA responde mal a una frase, añade esa frase como un nuevo pattern a la intención correcta.

    Si notas que la IA es repetitiva, añade más responses.

Después de cada modificación a intents.json, debes re-ejecutar python train_model.py y luego reiniciar la aplicación Streamlit para que cargue el nuevo modelo mejorado.

Nota adicional:
    si en dado caso no te permite ejecutar el entorno virtual, digitar como administrador en el powershell el siguiente comando:
    Recuerda cerrar visual studio antes.
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser