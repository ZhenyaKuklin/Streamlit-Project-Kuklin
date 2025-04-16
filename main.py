import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


#Назваине
st.title('Заполни пропуски')
st.write('Загрузи свой датафрейм и заполни его')



#Описание

## Шаг 1. Загрузка CSV файла
uploaded_file = st.sidebar.file_uploader('Загрузки CSV файл', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(15))
else:
    st.stop()

## Шаг 2. Проверка наличия пропусков в файле

missed_values = df.isna().sum()
missed_values = missed_values[missed_values > 1]

if len(missed_values) > 0:
    fig, ax = plt.subplots()
    sns.barplot(x = missed_values.index, y = missed_values.values)
    ax.set_title('Пропускив в столбцах')
    st.pyplot(fig)
else:
    st.write('Пропусков нет')
    st.stop()

## Шаг 3. Заполнить эти пропуски


if len(missed_values) != 0:
    button = st.button('Заполнить пропуски')
    if button:
        df_filled = df[missed_values.index].copy()
        for col in df_filled.columns:
            if df_filled[col].dtype =='object':
                df_filled[col] = df_filled[col].fillna(col).mode()[0]
            else:
                df_filled[col] = df_filled[col].fillna(col).median()[0]


        st.write(df_filled.head(5))

## Шаг 4ю Выгрузить заполненный от пропусков CSV файл


        dowload_button = st.download_button(label = 'Скачать CSV',
                   data = df_filled.to_csv(), 
                   file_name='filled_fate.csv')
