import streamlit as st
import pandas as pd
import numpy as np
import pickle


#Configuración página de Streamlit
st.set_page_config(page_title="App de predicción de enfermedades cardiacas",
                     page_icon='https://cdn-icons-png.flaticon.com/512/5935/5935638.png',
                     layout='centered',
                     initial_sidebar_state='auto')

#Título de la página y descripción
st.title('App de predicción de enfermedades cardiacas')
st.markdown('Esta aplicación es una herramienta que permite predecir si un paciente tiene una enfermedad cardíaca o no. Para ello, se han utilizado datos de pacientes con y sin enfermedades cardíacas para entrenar un modelo de Machine Learning. A continuación, se solicitarán algunos datos del paciente y se mostrará la predicción del modelo.')
st.markdown("""---""")



#datos del paciente
st.sidebar.header('Datos del paciente')

#cargar CSV o manual
upload_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
if upload_file is not None:
   data = pd.read_csv(upload_file)
else:
   def user_input_features():
      #Controles deslizantes
      edad = st.sidebar.slider('Edad', 29, 77, 55)
      sexo = st.sidebar.selectbox('Sexo', ('Hombre', 'Mujer'))
      tipo_dolor_pecho = st.sidebar.selectbox('Tipo de dolor en el pecho', ('Tipica Angina', 'Angina Atipica', 'Dolor no Anginal', 'Asintomatico'))
      presion_arterial_reposo = st.sidebar.slider('Presión arterial en reposo', 94, 200, 131)
      colesterol = st.sidebar.slider('Colesterol', 126, 564, 246)
      azucar = st.sidebar.selectbox('Azúcar en sangre en ayunas', ('Sin Azucar en Sangre', 'Azucar en Sangre'))
      electrocardiograma_reposo = st.sidebar.selectbox('Resultados del electrocardiograma en reposo', ('Normal', 'Ondas ST-T anormales', 'Hipertrofia Ventricular Izquierda'))
      frecuencia_cardiaca_maxima = st.sidebar.slider('Frecuencia cardíaca máxima', 71, 202, 149)
      angina_inducida_por_ejercicio = st.sidebar.selectbox('Angina inducida por ejercicio', ('Si', 'No'))
      depresion_st = st.sidebar.slider('Depresión del segmento ST inducida por el ejercicio', 0.0, 6.2, 1.0)
      pendiente_st = st.sidebar.selectbox('Pendiente del segmento ST', ('Ascendente', 'Plana', 'Descendente'))
      vasos_principales_coloreados = st.sidebar.selectbox('Número de vasos principales coloreados por la fluoroscopia', (0, 1, 2, 3, 4))
      puntos_frios_en_explo = st.sidebar.selectbox('Puntos fríos en la exploración de estrés de talio', ('Normal', 'Defecto Fijo', 'Defecto Reversible'))
      
      #Diccionario con los datos del paciente
      data = {'edad': edad,
            'sexo': sexo,
            'tipo_dolor_pecho': tipo_dolor_pecho,
            'presion_arterial_reposo': presion_arterial_reposo,
            'colesterol': colesterol,
            'azucar': azucar,
            'electrocardiograma_reposo': electrocardiograma_reposo,
            'frecuencia_cardiaca_maxima': frecuencia_cardiaca_maxima,
            'angina_inducida_por_ejercicio': angina_inducida_por_ejercicio,
            'depresion_st': depresion_st,
            'pendiente_st': pendiente_st,
            'vasos_principales_coloreados': vasos_principales_coloreados,
            'puntos_frios_en_explo': puntos_frios_en_explo}
      
      #Crear DataFrame
      features = pd.DataFrame(data, index=[0])
      return features
   df = user_input_features()

#Convertir variables categóricas en numéricas
df['sexo'] = df['sexo'].replace({'Mujer': 1, 'Hombre': 0})
df['tipo_dolor_pecho'] = df['tipo_dolor_pecho'].replace({'Tipica Angina':0 , 'Angina Atipica':1 , 'Dolor no Anginal':2 , 'Asintomatico':3 })
df['azucar'] = df['azucar'].replace({'Azucar en Sangre': 1, 'Sin Azucar en Sangre': 0})  
df['electrocardiograma_reposo'] = df['electrocardiograma_reposo'].replace({'Normal': 0,'Ondas ST-T anormales':1 , 'Hipertrofia Ventricular Izquierda':2 })
df['angina_inducida_por_ejercicio'] = df['angina_inducida_por_ejercicio'].replace({'Si': 1, 'No': 0})
df['pendiente_st'] = df['pendiente_st'].replace({'Ascendente': 0 , 'Plana': 1, 'Descendente': 2})
df['puntos_frios_en_explo'] = df['puntos_frios_en_explo'].replace({'Normal':1 , 'Defecto Fijo': 2 , 'Defecto Reversible': 3})


#Seleccionar datos csv primera fila
df = df[:1]

if upload_file is not None:
   st.write(df)
else: 
   st.write('Esperando a que se cargue el archivo CSV')
   st.write(df)

#Cargar modelo
load_modelo  = pickle.load(open('https://github.com/cofrian/PROYECTO-1/blob/427247708ae9b68e5031a69aece19b779a2384cb/app_heart/model_heart.pkl', 'rb'))

#aplicar modelo
prediccion = load_modelo.predict(df)
prediccion_proba = load_modelo.predict_proba(df)

col1, col2 = st.columns(2) 

with col1:
   st.subheader('Predicción')
   st.write(prediccion)   

with col2: 
   st.subheader('Probabilidad')
   st.write(prediccion_proba)


if prediccion == 0:
   st.write('El paciente no tiene enfermedad cardíaca')
else:
   st.write('El paciente tiene enfermedad cardíaca')
st.markdown("""---""")
