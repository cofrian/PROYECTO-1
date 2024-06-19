import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

#Configuración página de Streamlit
st.set_page_config(page_title="App de predicción de Diabetes",
                     page_icon='5935562.png',
                     layout='centered',
                     initial_sidebar_state='auto')

#Título de la página y descripción
st.title('App de predicción de Diabetes')
st.markdown('Esta aplicación es una herramienta que permite predecir si un paciente tiene la enfermedad diabetes o no. Para ello, se han utilizado datos de pacientes con y sin diabetes para entrenar un modelo de Machine Learning. A continuación, se solicitarán algunos datos del paciente y se mostrará la predicción del modelo.')
st.markdown("""---""")

#logo barra lateral 
logo = "./app_diabetes/28998.jpg"
st.sidebar.image(logo, width=150  )

#datos del paciente
st.sidebar.header('Datos del paciente')

#cargar CSV o manual
upload_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
if upload_file is not None:
   data = pd.read_csv(upload_file)
else:
   def user_input_features():
      #Controles deslizantes
      edad = st.sidebar.slider('Edad', 0, 80, 47)
      genero = st.sidebar.selectbox('Genero', ('Hombre', 'Mujer'))
      hipertension = st.sidebar.selectbox('Hipertensión', ('Sí', 'No')) 
      nivel_glucosa_sangre = st.sidebar.slider('Nivel medio de glucosa', 80, 300, 139)
      HbA1c= st.sidebar.slider('Nivel medio de HbA1c', 3, 9,56)
      imc=st.sidebar.slider('Indice de masa corporal', 10, 91,28)

      
      #Diccionario con los datos del paciente
      data = {'edad': edad,
            'genero': genero,
            'hipertension': hipertension,
            'HbA1c': HbA1c,
            'nivel_glucosa_sangre': nivel_glucosa_sangre,
            'imc': imc}
      
      #Crear DataFrame
      features = pd.DataFrame(data, index=[0])
      return features
   df = user_input_features()

#Convertir variables categóricas en numéricas
df['genero'] = df['genero'].replace({'Hombre': 1, 'Mujer': 0})
df['hipertension'] = df['hipertension'].replace({'Si': 1, 'No': 0})


#Seleccionar datos csv primera fila
df = df[:1]

if upload_file is not None:
   st.write(df)
else: 
   st.write('Esperando a que se cargue el archivo CSV')
   st.write(df)

#Cargar modelo
load_modelo  = pickle.load(open('model_diabetes.pkl', 'rb'))

#aplicar modelo
prediccion = load_modelo.predict(df)
prediccion_probab = load_modelo.predict_proba(df)

col1, col2 = st.columns(2) 

with col1:
   st.subheader('Predicción')
   st.write(prediccion)   

with col2: 
   st.subheader('Probabilidad')
   st.write(prediccion_probab)


if prediccion == 0:
   st.write('El paciente no tiene diabetes')
else:
   st.write('El paciente tiene diabetes')
st.markdown("""---""")
