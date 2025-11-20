from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Cargar el modelo
modelo = load_model('modelo_nubes.h5')
CLASES = ['Cumulus', 'Stratus', 'Cirrus', 'Nimbostratus']

def clasificar_nube_real(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(128, 128))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = modelo.predict(x)[0]
    indice = np.argmax(pred)
    clase = CLASES[indice]
    probabilidad = round(pred[indice] * 100, 2)
    return clase, probabilidad

def obtener_ficha_tecnica(tipo_nube):
    fichas = {
        'Cumulus': {'altitud': '1,000–2,000 m', 'descripcion': 'Nubes blancas y algodonosas que indican buen clima.'},
        'Stratus': {'altitud': '0–2,000 m', 'descripcion': 'Nubes grises que cubren el cielo como una manta.'},
        'Cirrus': {'altitud': '6,000–12,000 m', 'descripcion': 'Nubes delgadas y altas que anuncian cambios de clima.'},
        'Nimbostratus': {'altitud': '2,000–4,000 m', 'descripcion': 'Nubes oscuras que traen lluvia continua.'}
    }
    return fichas.get(tipo_nube, {'altitud': 'Desconocida', 'descripcion': 'Sin información disponible.'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    archivo = request.files['imagen']
    if archivo:
        ruta_guardado = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(ruta_guardado)

        tipo_nube, probabilidad = clasificar_nube_real(ruta_guardado)
        ficha = obtener_ficha_tecnica(tipo_nube)

        return render_template('resultado.html',
                               imagen_subida=ruta_guardado,
                               tipo_nube=tipo_nube,
                               probabilidad=probabilidad,
                               ficha=ficha)

if __name__ == '__main__':
    app.run(debug=True)
