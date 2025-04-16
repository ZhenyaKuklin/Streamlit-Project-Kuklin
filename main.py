# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings

# warnings.filterwarnings('ignore')


# #Назваине
# st.title('Заполни пропуски')
# st.write('Загрузи свой датафрейм и заполни его')



# #Описание

# ## Шаг 1. Загрузка CSV файла

# uploaded_file = st.sidebar.file_uploader('Загрузки CSV файл', type='csv')
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df.head(15))
# else:
#     st.stop()

# ## Шаг 2. Проверка наличия пропусков в файле

# missed_values = df.isna().sum()
# missed_values = missed_values[missed_values > 1]

# if len(missed_values) > 0:
#     fig, ax = plt.subplots()
#     sns.barplot(x = missed_values.index, y = missed_values.values)
#     ax.set_title('Пропускив в столбцах')
#     st.pyplot(fig)
# else:
#     st.write('Пропусков нет')
#     st.stop()

# ## Шаг 3. Заполнить эти пропуски


# if len(missed_values) != 0:
#     button = st.button('Заполнить пропуски')
#     if button:
#         df_filled = df[missed_values.index].copy()
#         for col in df_filled.columns:
#             if df_filled[col].dtype =='object':
#                 df_filled[col] = df_filled[col].fillna(col).mode()[0]
#             else:
#                 df_filled[col] = df_filled[col].fillna(col).median()[0]


#         st.write(df_filled.head(5))

# ## Шаг 4ю Выгрузить заполненный от пропусков CSV файл


#         dowload_button = st.download_button(label = 'Скачать CSV',
#                    data = df_filled.to_csv(), 
#                    file_name='filled_fate.csv')
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import date,datetime
import math
import numpy as np
import matplotlib as mpl
import io
from PIL import Image

st.sidebar.title('Чаевые в ресторане')

st.sidebar.write('Загрузи свой датафрейм и заполни его')



#Описание

## Шаг 1. Загрузка CSV файла


@st.cache_data(ttl=3600, show_spinner='Freching data from API...', max_entries=1000)
def load_running(file):
    df = pd.read_csv(file)
    return df

def main():
    uploaded_file = st.sidebar.file_uploader('Загрузки CSV файл', type='csv')
    if uploaded_file is not None:
        st.header('Таблица сведений о кафе!')
        df = load_running(uploaded_file)
        st.write(df.head(120))
        return df
    else:
        st.stop()

df = main()

## Шаг 2. Проверка наличия пропусков в файле

missed_values = df.isna().sum()
missed_values = missed_values[missed_values > 1]

## Шаг 3. Заполнить эти пропуски
if len(missed_values) > 0:
    fig, ax = plt.subplots()
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
    else:
        st.write('Пропусков нет')

## Шаг 4. Добавим графики в приложение
st.write("""
## 1 Шаг. Определим динамику чаевых c 1 по 31 января 2023 года.
""")
fig = plt.figure(figsize=(10,4), dpi=150)
ax = fig.add_subplot()
start= '2023-01-01'
end = '2023-01-31'
date_range = pd.date_range(start=start, end=end)
random_dates = np.random.choice(date_range, size=len(df.index.tolist())) 

df['time_order'] = random_dates
sns.lineplot(df.groupby('time_order')['tip'].sum(), color = 'green', marker = 'o', markerfacecolor = 'y')
fig.set_facecolor('antiquewhite')
ax.set_title('Dynamic tips', fontweight = 'bold',size = 20)
ax.set_xlabel('Date', fontweight = 'bold', size = 20)
ax.set_ylabel('$',fontweight = 'bold', size = 20)
ax.tick_params(axis = 'y', labelcolor = 'black', size = 10)
ax.tick_params(axis = 'x', labelcolor = 'black',size = 10)
ax.legend(['tips'])
plt.grid(True)
st.line_chart(df.groupby('time_order')['tip'].sum(),color = '#ffaa00')
picture = fig.savefig('My savefig')

buf = io.BytesIO()
plt.savefig(buf,format = 'png')
buf.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf, 
                   file_name='График 1',
                   mime='image/png',
                   key='download_button_1')

# file = fig.savefig()
# dowload_button = st.download_button(label = 'Скачать график',
#                    data = file, 
#                    file_name='filled_fate.csv')

st.write("""
## 2 Шаг. Изобразим график распределения 
""")

fig1= plt.figure(figsize=(15,7),dpi = 150)
ax1 = fig1.add_subplot()
s = sns.histplot(data = df['total_bill'],bins=10, kde=True, color='green')
fig1.set_facecolor('grey')

plt.title('Total_bill', fontweight = 'bold',size = 20)
plt.xlabel('', fontweight = 'bold', size = 20)
plt.ylabel('$',fontweight = 'bold', size = 20)
plt.tick_params(axis = 'y', labelcolor = 'black', size = 15)
plt.tick_params(axis = 'x', labelcolor = 'black',size = 15)
plt.legend(['tips'],fontsize = 15)

st.pyplot(fig1)

buf2 = io.BytesIO()
plt.savefig(buf2,format = 'png')
buf2.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf2, 
                   file_name='График 2',
                   mime='image/png',
                   key='download_button_2')

st.write("""
## 3 Шаг. Отобразим график, показывающий связь между счетом and чаевыми.
""")

fig2= plt.figure(figsize=(10,7),dpi = 150)
ax2 = fig2.add_subplot()
fig2.set_facecolor('darksalmon')

sns.scatterplot(df.groupby('total_bill')['tip'].agg('sum'), color = 'orange')

plt.title('Scatterplot between total and tip', fontweight = 'bold',size = 20)
plt.xlabel('Total_bill', fontweight = 'bold', size = 20)
plt.ylabel('Tip',fontweight = 'bold', size = 20)


plt.tick_params(axis = 'y', labelcolor = 'black', size = 15)

plt.tick_params(axis = 'x', labelcolor = 'black',size = 15)

plt.legend(['tips'],fontsize = 15)
sns.set_style('white')

st.scatter_chart(df.groupby('total_bill')['tip'].agg('sum'), color = '#ff0000')

buf3 = io.BytesIO()
plt.savefig(buf3,format = 'png')
buf3.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf3, 
                   file_name='График 3',
                   mime='image/png',
                   key='download_button_3')



st.write("""
## 4 Шаг. Изобразим график, связывающий счет, чаевые, и размер.
""")

fig3= plt.figure(figsize=(10,7),dpi = 150)
ax3 = fig3.add_subplot()
fig3.set_facecolor('bisque')

sns.scatterplot(data=df, x='total_bill', y='tip', size='size', sizes=(20, 500),hue='size', palette='viridis',alpha=0.6,legend=False)    


plt.title('Scatter Plot of Total Bill vs Tip with Size of Group',fontweight = 'bold',size = 20)
plt.xlabel('Total Bill',fontweight = 'bold', size = 20)
plt.ylabel('Tip',fontweight = 'bold', size = 20)


plt.tick_params(axis = 'y', labelcolor = 'black', size = 15)

plt.tick_params(axis = 'x', labelcolor = 'black',size = 15)

st.scatter_chart(data=df, x='total_bill', y='tip')

buf4 = io.BytesIO()
plt.savefig(buf4,format = 'png')
buf4.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf4, 
                   file_name='График 4',
                   mime='image/png',
                   key='download_button_4')

st.write("""
## 5 Шаг. Определим связь между днем недели и размером счета.
""")

fig4= plt.figure(figsize=(20,8),dpi = 300)
ax4 = fig4.add_subplot()
fig4.set_facecolor('lightgrey')

df1 = df.copy()

df1['time_order'] = df1['time_order'].apply(lambda x: x.strftime('%A'))

df2 = df1.groupby('time_order')[['total_bill']].sum().reset_index()


sns.barplot(x = df2['time_order'],y = df2['total_bill'] ,color = 'orange',edgecolor = 'black')



plt.title('Sum total for week', fontweight = 'bold',size = 20)
plt.xlabel('Days of week', fontweight = 'bold', size = 20)
plt.ylabel('Total Bill',fontweight = 'bold', size = 20)


plt.tick_params(axis = 'y', labelcolor = 'black', size = 20)

plt.tick_params(axis = 'x', labelcolor = 'black',size = 20)

plt.legend(['tips'],fontsize = 15)

st.bar_chart(df2, x = 'time_order',y = 'total_bill')

buf5 = io.BytesIO()
plt.savefig(buf5,format = 'png')
buf5.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf5, 
                   file_name='График 5',
                   mime='image/png',
                   key='download_button_5')



st.write("""
## 6 Шаг. Определим scatter plot с днем недели по оси **Y**, чаевыми по оси **X**, и цветом по полу.
""")

fig5= plt.figure(figsize=(20,7),dpi = 150)

fig5.set_facecolor('khaki')

df3 = df.copy()

df3['time_order']=  df['time_order'].apply(lambda x: x.strftime('%A'))

# tips4 = tips3.groupby('time_order')[['tip','sex']].
# tips4
sns.scatterplot(df3, x = 'tip', y = 'time_order', hue = 'sex', style = 'sex',palette='deep', s=40)

plt.title('Tips from male and Female', fontweight = 'bold',size = 20)
plt.xlabel('Tips', fontweight = 'bold', size = 20)
plt.ylabel('Days of week',fontweight = 'bold', size = 20)


plt.tick_params(axis = 'y', labelcolor = 'black', size = 15)

plt.tick_params(axis = 'x', labelcolor = 'black',size = 15)


plt.legend(fontsize = 20)

st.scatter_chart(df3, x = 'tip', y = 'time_order', color = 'sex', size=40)


buf6 = io.BytesIO()
plt.savefig(buf6,format = 'png')
buf6.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf6, 
                   file_name='График 6',
                   mime='image/png',
                   key='download_button_6')

st.write("""
## 7 Шаг. Нарисуем box plot c суммой всех счетов за каждый день, разбивая по time (Dinner/Lunch).
""")

df4 = df.sort_values(by = 'time_order').copy()

fig6= plt.figure(figsize=(30,10),dpi = 300)
g = sns.boxplot(x = 'time_order', y = 'total_bill', hue = 'time', data = df4)

plt.title('Box Plot with sum with Lunch and Dinner', fontweight = 'bold',size = 20)
plt.xlabel('Time_order', fontweight = 'bold', size = 20)
plt.ylabel('Total_bill',fontweight = 'bold', size = 20)

plt.tick_params(axis = 'y', labelcolor = 'black', size = 20)

plt.tick_params(axis = 'x', labelcolor = 'black',size = 20)


plt.xticks(rotation = 45)


plt.legend(fontsize = 20)

st.pyplot(fig6)

buf7 = io.BytesIO()
plt.savefig(buf7,format = 'png')
buf7.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf7, 
                   file_name='График 7',
                   mime='image/png',
                   key='download_button_7')


st.write("""
## 8 Шаг. Определим 2 гистограммы чаевых на обед и ланч. Расположите их рядом по горизонтали.
""")

fig7 = plt.figure(figsize=(17, 6))

# Гистограмма для обеда
plt.subplot(1, 2, 1) 
sns.histplot(df4[df4['time'] == 'Lunch']['tip'], bins=10, kde=False, color='lightgreen',edgecolor = 'black')
plt.title('Tips for Lunch')
plt.xlabel('Tips')
plt.ylabel('Frequency')

# Гистограмма для ужина
plt.subplot(1, 2, 2)
sns.histplot(df4[df4['time'] == 'Dinner']['tip'], bins=10, kde=False, color='yellow',edgecolor = 'black')
plt.title('Tips for Dinner')
plt.xlabel('Tips')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

st.pyplot(fig7)



buf8 = io.BytesIO()
plt.savefig(buf8,format = 'png')
buf8.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf8, 
                   file_name='График 8',
                   mime='image/png',
                   key='download_button_8')

st.write("""
## 9 Шаг. Изобразим 2 scatterplots (для мужчин и женщин), показав связь размера счета и чаевых, дополнительно разбив по курящим/некурящим. Расположите их по горизонтали.
""")
fig8 = plt.figure(figsize=(15, 6))

# Scatterplot для женщин
plt.subplot(1, 2, 1)  
sns.scatterplot(data=df4[df4['sex'] == 'Female'], x='total_bill', y='tip', hue='smoker', style='smoker', palette='Set1', s=100)
plt.title('Female Tips vs Total Bill', fontweight = 'bold', size = 20)
plt.xlabel('Total Bill',fontweight = 'bold', size = 20)
plt.ylabel('Tips',fontweight = 'bold', size = 20)
plt.legend(title='Smoker', fontsize = 10)


# Scatterplot для мужчин
plt.subplot(1, 2, 2)  
sns.scatterplot(data=df4[df4['sex'] == 'Male'], x='total_bill', y='tip', hue='smoker', style='smoker', palette='Set1', s=100)
plt.title('Male Tips vs Total Bill', fontweight = 'bold', size = 20)
plt.xlabel('Total Bill',fontweight = 'bold', size = 20)
plt.ylabel('Tips',fontweight = 'bold', size = 20)
plt.legend(title='Smoker', fontsize = 10)
plt.tight_layout() 

st.pyplot(fig8)

buf9 = io.BytesIO()
plt.savefig(buf9,format = 'png')
buf9.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf9, 
                   file_name='График 9',
                   mime='image/png',
                   key='download_button_9')

st.write("""
## 10 Шаг. Построим тепловую карту зависимостей численных переменных.
""")

correlation = df[['total_bill', 'tip', 'size']].corr()


fig11 = plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".3f", square=True, cbar_kws={"shrink": .8})

plt.title('Тепловая карта зависимостей численных переменных')

st.pyplot(fig11)

buf10 = io.BytesIO()
plt.savefig(buf10,format = 'png')
buf10.seek(0)

dowload_button = st.download_button(label = 'Скачать график',
                   data = buf10, 
                   file_name='График 10',
                   mime='image/png',
                   key='download_button_10')