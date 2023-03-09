import streamlit as st
from PIL import Image   
from utils import *

# Configura el favicon de la aplicación
favicon = open("logos/favicon.ico", "rb").read()
st.set_page_config(page_title="Registrado en PoH?", page_icon=favicon)

# agrego logo
logo = Image.open('logos/democratic-poh-logo-text-hi-res-p-500.png')
st.image(logo, use_column_width=True)

#Titulo en color verde
st.markdown("<h1 style='color: green;'>Registrado en Proof of Humanity?</h1>", unsafe_allow_html=True)

#definicion de las funciones principales
def process_image(path):
    try:
        _models = FaceNetModels()
        img = Image.open(path)
        image_embedding = _models.embedding(_models.mtcnn(img))
        distancia = _models.Distancia(image_embedding)
        return distancia[0][0],distancia[1][0]
    except:
        return None

def upload_image():
    uploaded_file = st.file_uploader("Subir la imagen de un Humano y verificar si esta registrado en https://proofofhumanity.org/", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida ', width=200)
        result = process_image(uploaded_file)
        if result:
            label, distance = result
            url = f"https://proofofhumanity.org/profile/{label}"
            st.markdown(f'<a href="{url}" target="_blank">https://proofofhumanity.org/profile/{label}</a>', unsafe_allow_html=True)
            st.write("Distancia Euclidiana: ", round(distance,4))
        else:
            st.write("Algo falló con la imagen proporcionada, intenta con otra !!")

 # información adicional
    with st.expander('Información adicional'):
         st.write('Esta aplicación utiliza el modelo de redes neuronales conocido como ResNet '+
                 'para reconocer características de rostros en imágenes. Con esta tecnología se construyó un diccionario que la app utiliza ' + 
                 'para comparar con las características del rostro ingresado por el usuario. '+     
                 'El usuario puede subir una imagen desde su dispositivo o utilizar la cámara para tomar una foto, y la aplicación '+ 
                 'devolverá el perfil de PoH más similar al de la base de datos junto con la distancia euclidiana entre los dos rostros.' +
                 ' Si la imagen corresponde a un humano registrado el rostro será reconocido en correspondencia con una distancia euclidiana muy baja, próxima a cero.'
                 ' Por el momento se reconocen 16 mil registrados en PoH, la base de datos se ira actualizando cada mil registrados nuevos.') 

#lanza app
upload_image()
