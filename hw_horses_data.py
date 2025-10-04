from inspect import stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#Задание 1(Загрузка данных)
horses_df = pd.read_csv('https://github.com/LinaAlter/horses-data/raw/main/horse_data.csv', header=None, na_values=['?'])


#print(horses_df.head(5))
#Выбрать нужные колонки и переназвать для удобства
selected_horses_df = horses_df.iloc[:, [0, 1, 3, 4, 5, 6, 10, 22]].copy()
selected_horses_df.columns = ['surgery', 'age', 'rectal temperature', 'pulse', 'respiratory rate',
                   'temperature of extremities', 'pain', 'outcome']
#Задание 2(Первичное изучение данных)
print(selected_horses_df.head(15))
print(selected_horses_df.info())

#Исправить типы данных по числовым колонкам.
int_columns = ['rectal temperature', 'pulse', 'respiratory rate']
selected_horses_df[int_columns] = selected_horses_df[int_columns].apply(pd.to_numeric)
category_columns = ['surgery', 'age', 'temperature of extremities', 'pain', 'outcome']
selected_horses_df[category_columns] = selected_horses_df[category_columns].astype('object')
selected_horses_df.info()

#Вывод базовых статистик по типам
print(selected_horses_df.describe().round(2))
print(selected_horses_df.describe(include=('object')))

#Вместо выбросов по категориальным данным:
plt.figure(figsize=(15, 10))

for i, col in enumerate(category_columns, 1):
    plt.subplot(2, 3, i)

    # Столбчатая диаграмма распределения
    sns.countplot(data=selected_horses_df, x=col)
    plt.title(f'Распределение: {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Выбросы по числовым
for col in int_columns:
    Q1 = selected_horses_df[col].quantile(0.25)
    Q3 = selected_horses_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = selected_horses_df[col][(selected_horses_df[col] < lower_bound) | (selected_horses_df[col] > upper_bound)]

    print(f"\n МЕТОД IQR для {col}:")
    print(f"Q1 (25-й перцентиль): {Q1:.2f}")
    print(f"Q3 (75-й перцентиль): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Границы выбросов: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Количество выбросов (IQR): {len(outliers_iqr)}")

print(f"\n{'='*50}")

#Задание 3(Работа с пропусками)
#Смотрим пропуски по колонкам
missing_values = selected_horses_df.isnull().sum()
print(missing_values)
#Визуализируем
msno.matrix(selected_horses_df)
msno.heatmap(selected_horses_df)

#по визуалу видно, что по одной строке пропуски по outcome и surgery.
missing_surgery = selected_horses_df[selected_horses_df['surgery'].isna()].index.tolist()
print("Строки с пропусками температуры:", missing_surgery)
missing_outcome = selected_horses_df[selected_horses_df['outcome'].isna()].index.tolist()
print("Строки с пропусками температуры:", missing_outcome)

horses_filled_df = selected_horses_df.copy()
#пропущенные данные в строке являются целевыми, следовательно без них вся строка не информативна, можно удалить.
horses_filled_df = horses_filled_df.dropna(subset=['outcome'])
#проверка
missing_values = horses_filled_df.isnull().sum()
print(missing_values)

#По визуализированным данным видно, что 'pain' и 'temperature of extremities' связаны сильно (0.6), возможно измеряются вместе(при первичном осмотре?).
#'rectal temperature', 'pulse', 'respiratory rate' связаны умеренно (0.3-0.4) - инструментальный осмотр по показаниям? Чем хуже показатели по боли 
#и температуре конечнойстей, тем чаще пропуски в 'rectal temperature', 'pulse', 'respiratory rate'. 

#Из файла .names: данные 'resperatory rate' имеют сомнительную пользу, можно заполнить медианой.

resp_rate_median = horses_filled_df['respiratory rate'].median()
horses_filled_df['respiratory rate'].fillna(resp_rate_median, inplace=True)
#missing_values = horses_filled_df.isnull().sum()
#print(missing_values)

#проводим анализ зависимостей пропусков от кат. данных
def analyze_missing_dependency_cat (df, target_col):
    temp_df = df.copy()
    temp_df['is_missing'] = temp_df[target_col].isna()

    categorical_cols = df.select_dtypes(include = ['object']).columns.tolist()
    print(f"Анализ связи пропусков в '{target_col}' c категориальными признаками\n")

    for col in categorical_cols:
      ct = pd.crosstab(temp_df['is_missing'], temp_df[col], normalize = 'index')
      #визуализация
      ct.plot.bar(stacked = True)
      plt.title(f'Распределение "{col}" в зависимости от пропусков в "{target_col}", fontsize = 15')
      plt.ylabel("Доля, %")
      plt.xlabel(f'Пропуск в"{target_col}"')
      plt.legend(title=col, bbox_to_anchor = (1.05, 1), loc='upper left')
      plt.xticks(rotation = 0)
      plt.show()

analyze_missing_dependency_cat(horses_filled_df, 'rectal temperature')
analyze_missing_dependency_cat(horses_filled_df, 'pulse')

#смотрим корреляцию между столбцами
horses_filled_df.corr(method='spearman')

#по полученным данным видно, что корреляция по всем столбцам не превышает 0,4. 'outcome' показывает сильную связь со столбцами с пропусками, сам пропусков больше не имеет.
#Заполняем пропуски по среднему, через группировку по 'outcome'



def fill_mean_by_outcome(col):
    means_by_outcome = horses_filled_df.groupby('outcome')[col].mean().round(0)
    
    
    horses_filled_df[col] = horses_filled_df.apply(
            lambda row: means_by_outcome[row['outcome']] if pd.isna(row[col]) else row[col],
            axis=1
        )

fill_mean_by_outcome('rectal temperature')
fill_mean_by_outcome('pulse')
fill_mean_by_outcome('pain')
fill_mean_by_outcome('temperature of extremities')

missing_values = horses_filled_df.isnull().sum()
print(missing_values)
print(horses_filled_df.head(15))
print(horses_filled_df.describe())















