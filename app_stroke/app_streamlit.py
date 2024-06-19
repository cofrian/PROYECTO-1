import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#Configuración página de Streamlit
st.set_page_config(page_title="App de predicción de Apoplejía",
                     page_icon='https://cdn-icons-png.flaticon.com/512/5935/5935638.png',
                     layout='centered',
                     initial_sidebar_state='auto')

#Título de la página y descripción
st.title('App de predicción de Apoplejía')
st.markdown('Esta aplicación es una herramienta que permite predecir si un paciente tiene la enfermedad apoplejia o no. Para ello, se han utilizado datos de pacientes con y sin apoplejía para entrenar un modelo de Machine Learning. A continuación, se solicitarán algunos datos del paciente y se mostrará la predicción del modelo.')
st.markdown("""---""")

#logo barra lateral 
logo = "logo.jpg"
st.sidebar.image(logo, width=150)

#datos del paciente
st.sidebar.header('Datos del paciente')

#cargar CSV o manual
upload_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
if upload_file is not None:
   data = pd.read_csv(upload_file)
else:
   def user_input_features():
      #Controles deslizantes
      edad = st.sidebar.slider('Edad', 10, 82, 48)
      genero = st.sidebar.selectbox('Genero', ('Hombre', 'Mujer'))
      hipertension = st.sidebar.selectbox('Hipertensión', ('Sí', 'No'))
      enfermedad_cardiaca = st.sidebar.slider('Enfermedad cardíaca', ('Sí', 'No')) 
      casado_alguna_vez = st.sidebar.slider('Casado alguna vez', ('Sí', 'No'))
      tipo_trabajo = st.sidebar.selectbox('Tipo de trabajo', ('Sector infantil', 'Sector del gobierno', 'Nunca trabajó', 'Sector privado', 'Autónomo'))
      tipo_residencia = st.sidebar.selectbox('Tipo de residencia', ('Rural', 'Urbana'))
      nivel_medio_glucosa = st.sidebar.slider('Nivel medio de glucosa', 55, 272, 109)
      imc = st.sidebar.selectbox('Índice de masa corporal', 11.5, 60, 28.1)
      tipo_fumador = st.sidebar.slider('Tipo de fumador', ('Ex-fumador', 'Nunca fumador', 'Fumador actual'))
      
      
      #Diccionario con los datos del paciente
      data = {'edad': edad,
            'genero': genero,
            'hipertension': hipertension,
            'enfermedad_cardiaca': enfermedad_cardiaca,
            'casado_alguna_vez': casado_alguna_vez,
            'tipo_trabajo': tipo_trabajo,
            'tipo_residencia': tipo_residencia,
            'nivel_medio_glucosa': nivel_medio_glucosa,
            'imc': imc,
            'tipo_fumador': tipo_fumador}
      
      #Crear DataFrame
      features = pd.DataFrame(data, index=[0])
      return features
   df = user_input_features()

#Convertir variables categóricas en numéricas
df['genero'] = df['genero'].replace({'Hombre': 1, 'Mujer': 0})
df['hipertension'] = df['hipertension'].replace({'Si': 1, 'No': 0})
df['enfermedad_cardiaca'] = df['enfermedad_cardiaca'].replace({'Si': 1, 'No': 0})  
df['casado_alguna_vez'] = df['casado_alguna_vez'].replace({'Si': 1, 'No': 0})
df['tipo_trabajo'] = df['tipo_trabajo'].replace({'Sector infantil': 0, 'Sector del gobierno': 1, 'Nunca trabajo': 2, 'Sector privado': 3, 'Autónomo': 4})
df['tipo_residencia'] = df['tipo_residencia'].replace({'Rural': 2 , 'Urbana': 1})
df['tipo_fumador'] = df['tipo_fumador'].replace({'Ex-fumador':1 , 'Nunca fumador': 0 , 'Fumador actual ': 2})


#Seleccionar datos csv primera fila
df = df[:1]

if upload_file is not None:
   st.write(df)
else: 
   st.write('Esperando a que se cargue el archivo CSV')
   st.write(df)

#Cargar modelo
load_modelo  = pickle.load(open('model_heart.pkl', 'rb'))

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
   st.write('El paciente no tiene apoplejía')
else:
   st.write('El paciente tiene apoplejía')
st.markdown("""---""")
