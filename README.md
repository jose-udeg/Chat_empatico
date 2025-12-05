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
 import nltk
 nltk.download('punkt_tab')

    Crear el Entorno Virtual: Abrir la terminal dentro de la carpeta del proyecto y ejecutar:
    Bash

    python -m venv venv

    Activar el Entorno:

     Windows (PowerShell): .\venv\Scripts\Activate.ps1

    pip install tensorflow numpy nltk streamlit pandas

Descargar Recursos, ejecutar los comandos de descarga:
Python

import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Importante para español
exit()


Ejecución del Proyecto

    Ejecutar la App de Streamlit:
    Bash

    streamlit run chat_app.py

Re-entrenar el Modelo

python train_model.py

Este comando recreará los archivos model.h5 y data.pickle.

Nota adicional:
    si en dado caso no te permite ejecutar el entorno virtual, digitar como administrador en el powershell el siguiente comando:
    Recuerda cerrar visual studio antes.
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


