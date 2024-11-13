# HR-аналитика: машинное обучение на службе персонала

HR-аналитики компании «Работа с заботой» помогают бизнесу оптимизировать управление персоналом: бизнес предоставляет данные, а аналитики предлагают, как избежать финансовых потерь и оттока сотрудников. В этом HR-аналитикам пригодится машинное обучение, с помощью которого получится быстрее и точнее отвечать на вопросы бизнеса.


Компания предоставила данные с характеристиками сотрудников компании. Среди них — уровень удовлетворённости сотрудника работой в компании. Эту информацию получили из форм обратной связи: сотрудники заполняют тест-опросник, и по его результатам рассчитывается доля их удовлетворённости от 0 до 1, где 0 — совершенно неудовлетворён, 1 — полностью удовлетворён

**Задачи:**
1. Предсказание удовлетворенности: Создать модель, которая будет прогнозировать уровень удовлетворенности сотрудника на основе данных о нем. 
2. Прогнозирование оттока: Разработать модель, которая сможет предсказать то, что сотрудник уволится из компании.

## Описание данных
* Информация о сотруднике.
* `job_satisfaction_rate` — уровень удовлетворённости сотрудника работой в компании, **целевой признак** для первой задачи.
* `quit` — увольнение сотрудника из компании, **целевой признак** для второй задачи.


## План работы

* **Задача 1: Предсказание удовлетворенности**
1. Загрузка и предобработка данных:<br/>
    1.1. Загрузка данных.<br/>
    1.2. Предобработка данных.<br/>
    1.3. Исследовательский анализ данных.<br/>
    1.4. Корреляционный анализ.<br/>
   <br/>
2. Построение пайплайна и обучение модели: <br/>
    2.1. Разбиение на обучающую и тестовую выборки.<br/>
    2.2. Подготовка данных и построение пайплайна.<br/>
    2.3. Обучение модели. <br/>
    2.4. Тестирование и вывод.<br/>
    <br/>
    
* **Задача 2: Прогнозирование оттока**
1. Загрузка и предобработка данных:<br/>
    1.1. Загрузка данных.<br/>
    1.2. Предобработка данных.<br/>
    1.3. Исследовательский и корреляционный анализ данных.<br/>
    1.4. Добавление нового входного признака. <br/>
    
   <br/>
2. Построение пайплайна и обучение модели: <br/>
     2.1. Разбиение на обучающую и тестовую выборки.<br/>
     2.2. Подготовка данных и построение пайплайна.<br/>
     2.3. Обучение модели. <br/>
     2.4. Тестирование и вывод.<br/>
    <br/>

* Общий вывод и рекомендации заказчику.

# Задача 1: Предсказание удовлетворенности

## 1 Загрузка и предобработка данных


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import phik
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.25
```

### 1.1. Загрузка данных


```python
train_job_satisfaction = pd.read_csv(r'~/train_job_satisfaction_rate.csv')
test_features = pd.read_csv(r'~/test_features.csv')
test_target_job_satisfaction = pd.read_csv(r'~/test_target_job_satisfaction_rate.csv')
```


```python
train_job_satisfaction.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>155278</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>1</td>
      <td>24000</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>653870</td>
      <td>hr</td>
      <td>junior</td>
      <td>high</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>38400</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184592</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>12000</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>171431</td>
      <td>technology</td>
      <td>junior</td>
      <td>low</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>18000</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>693419</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>22800</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>405448</td>
      <td>hr</td>
      <td>middle</td>
      <td>low</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>30000</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>6</th>
      <td>857135</td>
      <td>sales</td>
      <td>sinior</td>
      <td>medium</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>56400</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>7</th>
      <td>400657</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>high</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>52800</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>8</th>
      <td>198846</td>
      <td>hr</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>13200</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>149797</td>
      <td>technology</td>
      <td>middle</td>
      <td>high</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>54000</td>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_job_satisfaction.info()
display(train_job_satisfaction.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4000 entries, 0 to 3999
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   id                     4000 non-null   int64  
     1   dept                   3994 non-null   object 
     2   level                  3996 non-null   object 
     3   workload               4000 non-null   object 
     4   employment_years       4000 non-null   int64  
     5   last_year_promo        4000 non-null   object 
     6   last_year_violations   4000 non-null   object 
     7   supervisor_evaluation  4000 non-null   int64  
     8   salary                 4000 non-null   int64  
     9   job_satisfaction_rate  4000 non-null   float64
    dtypes: float64(1), int64(4), object(5)
    memory usage: 312.6+ KB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>employment_years</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4000.000000</td>
      <td>4000.000000</td>
      <td>4000.000000</td>
      <td>4000.000000</td>
      <td>4000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>544957.621000</td>
      <td>3.718500</td>
      <td>3.476500</td>
      <td>33926.700000</td>
      <td>0.533995</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257883.104622</td>
      <td>2.542513</td>
      <td>1.008812</td>
      <td>14900.703838</td>
      <td>0.225327</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100954.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>12000.000000</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>322836.750000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>22800.000000</td>
      <td>0.360000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>534082.500000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>30000.000000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>771446.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>43200.000000</td>
      <td>0.710000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999521.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>98400.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


Данные выглядят чистыми, без пропусков. 


```python
test_features.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>485046</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>28800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>686555</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>467458</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>418655</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>789145</td>
      <td>hr</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>40800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>429973</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>medium</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>42000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>850699</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>26400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>500791</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>49200</td>
    </tr>
    <tr>
      <th>8</th>
      <td>767867</td>
      <td>marketing</td>
      <td>middle</td>
      <td>high</td>
      <td>3</td>
      <td>no</td>
      <td>yes</td>
      <td>4</td>
      <td>62400</td>
    </tr>
    <tr>
      <th>9</th>
      <td>937235</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>26400</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_features.info()
display(test_features.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 9 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   id                     2000 non-null   int64 
     1   dept                   1998 non-null   object
     2   level                  1999 non-null   object
     3   workload               2000 non-null   object
     4   employment_years       2000 non-null   int64 
     5   last_year_promo        2000 non-null   object
     6   last_year_violations   2000 non-null   object
     7   supervisor_evaluation  2000 non-null   int64 
     8   salary                 2000 non-null   int64 
    dtypes: int64(4), object(5)
    memory usage: 140.8+ KB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>employment_years</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>552765.213500</td>
      <td>3.666500</td>
      <td>3.526500</td>
      <td>34066.800000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>253851.326129</td>
      <td>2.537222</td>
      <td>0.996892</td>
      <td>15398.436729</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100298.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>12000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>339052.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>22800.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>550793.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>765763.750000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>43200.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999029.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>96000.000000</td>
    </tr>
  </tbody>
</table>
</div>


Есть несколько пропусков в `dept` и `level`


```python
test_target_job_satisfaction.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130604</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>825977</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>418490</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>555320</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>826430</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>817219</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>269033</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>7</th>
      <td>962356</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>649052</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>532834</td>
      <td>0.59</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_target_job_satisfaction.info()
display(test_target_job_satisfaction.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 2 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   id                     2000 non-null   int64  
     1   job_satisfaction_rate  2000 non-null   float64
    dtypes: float64(1), int64(1)
    memory usage: 31.4 KB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>552765.213500</td>
      <td>0.54878</td>
    </tr>
    <tr>
      <th>std</th>
      <td>253851.326129</td>
      <td>0.22011</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100298.000000</td>
      <td>0.03000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>339052.000000</td>
      <td>0.38000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>550793.000000</td>
      <td>0.58000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>765763.750000</td>
      <td>0.72000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999029.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>


Здесь тоже все хорошо.

**Вывод:** данные хорошие, только есть несколько пропусков в датасете test_features.

### 1.2. Предобработка данных

Столбцы уже приведены к змеиному регистру, типы данных соответствует нужному, проверим на дубликаты.


```python
print('Количество дубликатов в train_job_satisfaction: ', train_job_satisfaction.duplicated().sum())
print('Количество дубликатов в test_features: ', test_features.duplicated().sum())
print('Количество дубликатов в test_target_job_satisfaction: ', test_target_job_satisfaction.duplicated().sum())
```

    Количество дубликатов в train_job_satisfaction:  0
    Количество дубликатов в test_features:  0
    Количество дубликатов в test_target_job_satisfaction:  0
    


```python
train_job_satisfaction[train_job_satisfaction.duplicated(
    subset=['dept', 'level', 'workload', 'employment_years', 'last_year_promo', 
            'last_year_violations', 'supervisor_evaluation','salary', 'job_satisfaction_rate'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>437</th>
      <td>302957</td>
      <td>purchasing</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>15600</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>502</th>
      <td>752399</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>28800</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>520</th>
      <td>802286</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>21600</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>676</th>
      <td>167303</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>24000</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>784</th>
      <td>191841</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>21600</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3969</th>
      <td>737303</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>33600</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>3984</th>
      <td>281204</td>
      <td>technology</td>
      <td>junior</td>
      <td>low</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>15600</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3989</th>
      <td>261436</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>22800</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>457950</td>
      <td>technology</td>
      <td>junior</td>
      <td>high</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>46800</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>957499</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>21600</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>
<p>245 rows × 10 columns</p>
</div>



Без id присутствуют 245 дубликатов. Не будем трогать, так как датасет небольшой.


```python
train_job_satisfaction.isna().sum()
```




    id                       0
    dept                     6
    level                    4
    workload                 0
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    job_satisfaction_rate    0
    dtype: int64



Обработаем пропуски в пайплайне.

**Вывод**: В данных обнаружены дубликаты. Их не обрабатывали, пока возьмем на заметку.

### 1.3. Исследовательский анализ данных

Построим функции для построения графиков.


```python
def cat_plot(df, column):
    display(df[column].unique())
    plt.figure(figsize=(9,7))
    sns.countplot(y=column, data=df)
    plt.title(f'Рапределение по {column}', fontsize=16)
    plt.xlabel('Количество', fontsize=14)
    plt.show()
```


```python
def hist_plot(df, column):
    display(df[column].unique())
    plt.figure(figsize=(9,7))
    sns.histplot(y=column, data=df)
    plt.title(f'Рапределение по {column}', fontsize=16)
    plt.xlabel('Количество', fontsize=14)
    plt.show()    
```

Будем сразу строить признаки из двух датасетов и сранивать.


```python
cat_plot(train_job_satisfaction, 'dept')
display(cat_plot(test_features, 'dept'))
```


    array(['sales', 'hr', 'technology', 'purchasing', 'marketing', nan],
          dtype=object)



    
![png](output_30_1.png)
    



    array(['marketing', 'hr', 'sales', 'purchasing', 'technology', nan, ' '],
          dtype=object)



    
![png](output_30_3.png)
    



    None


Видим, что `sales`является преобладающим. Ага, у нас есть данные в виде " ". Нужно это обработать.


```python
test_features['dept'] = test_features['dept'].replace(' ', np.nan)
```


```python
cat_plot(train_job_satisfaction,'level')
display(cat_plot(test_features,'level'))
```


    array(['junior', 'middle', 'sinior', nan], dtype=object)



    
![png](output_33_1.png)
    



    array(['junior', 'middle', 'sinior', nan], dtype=object)



    
![png](output_33_3.png)
    



    None


Ого, скорее всего, компания приличного уровня, так как меньшинство составляет именно sinior с огромным отрывом.


```python
cat_plot(train_job_satisfaction,'workload')
display(cat_plot(test_features,'workload'))
```


    array(['medium', 'high', 'low'], dtype=object)



    
![png](output_35_1.png)
    



    array(['medium', 'low', 'high', ' '], dtype=object)



    
![png](output_35_3.png)
    



    None



```python
test_features['workload'] = test_features['workload'].replace(' ', np.nan)
```


```python
test_features['workload'].unique()
```




    array(['medium', 'low', 'high', nan], dtype=object)




```python
test_features['dept'].unique()
```




    array(['marketing', 'hr', 'sales', 'purchasing', 'technology', nan],
          dtype=object)



Преобладающим ответом является medium. Возможно, в зоне риска располагаются как раз сотрудники с ответом high.


```python
hist_plot(train_job_satisfaction, 'employment_years')
display(hist_plot(test_features, 'employment_years'))
```


    array([ 2,  1,  4,  7,  9,  6,  3, 10,  8,  5], dtype=int64)



    
![png](output_40_1.png)
    



    array([ 2,  1,  5,  6,  3,  9,  7,  4,  8, 10], dtype=int64)



    
![png](output_40_3.png)
    



    None



```python
train_job_satisfaction.boxplot(column=['employment_years'])
plt.ylabel("Стаж работы (лет)") 
plt.show()
```


    
![png](output_41_0.png)
    



```python
display(test_features.boxplot(column=['employment_years']))
plt.ylabel("Стаж работы (лет)") 
plt.show()
```


    <AxesSubplot:>



    
![png](output_42_1.png)
    


Большинство сотрудников работает в этой компании не так давно. 


```python
cat_plot(train_job_satisfaction,'last_year_promo')
display(cat_plot(test_features,'last_year_promo'))
```


    array(['no', 'yes'], dtype=object)



    
![png](output_44_1.png)
    



    array(['no', 'yes'], dtype=object)



    
![png](output_44_3.png)
    



    None


Возможно, стоит предложить некоторым сотрудникам повышение, кто подходит на эту роль, разумеется. Нужно будет учесть такой дисбаланс при построении модели. 


```python
cat_plot(train_job_satisfaction,'last_year_violations')
display(cat_plot(test_features,'last_year_violations'))
```


    array(['no', 'yes'], dtype=object)



    
![png](output_46_1.png)
    



    array(['no', 'yes'], dtype=object)



    
![png](output_46_3.png)
    



    None


О, несмотря на дисбаланс, график радует ответом!


```python
cat_plot(train_job_satisfaction,'supervisor_evaluation')
display(cat_plot(test_features,'supervisor_evaluation'))
```


    array([1, 5, 2, 3, 4], dtype=int64)



    
![png](output_48_1.png)
    



    array([5, 4, 3, 1, 2], dtype=int64)



    
![png](output_48_3.png)
    



    None



```python
train_job_satisfaction.boxplot(column=['supervisor_evaluation'])
plt.ylabel("Оценка") 
plt.show()
```


    
![png](output_49_0.png)
    



```python
test_features.boxplot(column=['supervisor_evaluation'])
plt.ylabel("Оценка") 
plt.show()
```


    
![png](output_50_0.png)
    


В среднем оценки неплохие. Возможно, у тех, кто получил 1 или 2 нарушали договор? сейчас проверим.


```python
train_job_satisfaction[
    (train_job_satisfaction['supervisor_evaluation']<=2) & 
    (train_job_satisfaction['last_year_violations']=='yes')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>385514</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>8</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>32400</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>64</th>
      <td>188350</td>
      <td>marketing</td>
      <td>middle</td>
      <td>high</td>
      <td>6</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>60000</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>87</th>
      <td>702721</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>6</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>31200</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>157</th>
      <td>335445</td>
      <td>technology</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>21600</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>193</th>
      <td>493733</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>26400</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3954</th>
      <td>706200</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>high</td>
      <td>10</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>51600</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>3964</th>
      <td>351063</td>
      <td>hr</td>
      <td>sinior</td>
      <td>high</td>
      <td>6</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>64800</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>3972</th>
      <td>134106</td>
      <td>hr</td>
      <td>middle</td>
      <td>high</td>
      <td>10</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>57600</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>713279</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>12000</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>338347</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>44400</td>
      <td>0.18</td>
    </tr>
  </tbody>
</table>
<p>147 rows × 10 columns</p>
</div>



Хм, видимо, эти признаки не так сильно коррелируют, как казалось изначально. Проверим это позже, когда будем строить матрицу. 


```python
hist_plot(train_job_satisfaction,'salary')
```


    array([24000, 38400, 12000, 18000, 22800, 30000, 56400, 52800, 13200,
           54000, 19200, 40800, 34800, 27600, 26400, 33600, 50400, 15600,
           14400, 25200, 72000, 31200, 32400, 48000, 43200, 46800, 58800,
           84000, 44400, 39600, 37200, 21600, 28800, 62400, 60000, 42000,
           49200, 55200, 57600, 68400, 45600, 51600, 64800, 80400, 20400,
           61200, 76800, 69600, 16800, 36000, 63600, 81600, 66000, 74400,
           67200, 70800, 73200, 75600, 79200, 94800, 78000, 88800, 92400,
           85200, 91200, 98400, 96000, 97200], dtype=int64)



    
![png](output_54_1.png)
    



```python
test_features.boxplot(column=['salary'])
plt.ylabel("Заработная плата") 
plt.show()
```


    
![png](output_55_0.png)
    



```python
train_job_satisfaction.boxplot(column=['salary'])
plt.ylabel("Заработная плата") 
plt.show()
```


    
![png](output_56_0.png)
    


Большинство получает около 30000. Возможно, стоит повысить зарплату?...


```python
hist_plot(train_job_satisfaction,'job_satisfaction_rate')
display(hist_plot(test_target_job_satisfaction,'job_satisfaction_rate'))
```


    array([0.58, 0.76, 0.11, 0.37, 0.2 , 0.78, 0.56, 0.44, 0.14, 0.47, 0.74,
           0.42, 0.32, 0.57, 0.16, 0.69, 0.33, 0.64, 0.39, 0.8 , 0.79, 0.17,
           0.65, 0.18, 0.19, 0.49, 0.63, 0.22, 0.23, 0.5 , 0.35, 0.3 , 0.77,
           0.88, 0.59, 0.21, 0.36, 0.85, 0.7 , 0.48, 0.6 , 0.1 , 0.27, 0.71,
           0.86, 0.54, 0.73, 0.46, 0.31, 0.72, 0.51, 0.61, 0.81, 0.99, 0.15,
           0.91, 0.68, 0.4 , 0.89, 0.67, 0.75, 0.98, 0.26, 0.45, 0.92, 0.82,
           0.66, 0.55, 0.38, 0.53, 0.84, 0.52, 0.24, 0.62, 0.41, 0.28, 0.09,
           0.97, 0.83, 0.25, 0.43, 0.04, 0.13, 0.29, 0.95, 0.93, 0.87, 0.08,
           0.94, 0.07, 0.34, 0.9 , 0.12, 0.06, 0.96, 0.05, 1.  , 0.03])



    
![png](output_58_1.png)
    



    array([0.74, 0.75, 0.6 , 0.72, 0.08, 0.76, 0.64, 0.38, 0.14, 0.59, 0.91,
           0.78, 0.7 , 0.79, 0.34, 0.81, 0.23, 0.4 , 0.58, 0.77, 0.68, 0.24,
           0.42, 0.69, 0.47, 0.35, 0.71, 0.83, 0.61, 0.65, 0.37, 0.45, 0.63,
           0.82, 0.16, 0.89, 0.28, 0.32, 0.88, 0.36, 0.33, 0.31, 0.27, 0.73,
           0.53, 0.26, 0.57, 0.2 , 1.  , 0.56, 0.67, 0.19, 0.52, 0.43, 0.12,
           0.11, 0.21, 0.13, 0.49, 0.22, 0.86, 0.46, 0.41, 0.48, 0.29, 0.87,
           0.66, 0.8 , 0.55, 0.5 , 0.51, 0.62, 0.85, 0.84, 0.15, 0.39, 0.25,
           0.9 , 0.07, 0.1 , 0.92, 0.3 , 0.44, 0.18, 0.93, 0.54, 0.96, 0.09,
           0.99, 0.17, 0.95, 0.06, 0.94, 0.03, 0.98, 0.97, 0.04, 0.05])



    
![png](output_58_3.png)
    



    None



```python
print(train_job_satisfaction['job_satisfaction_rate'].mean())
test_target_job_satisfaction['job_satisfaction_rate'].mean()
```

    0.5339950000000013
    




    0.5487799999999999




```python
train_job_satisfaction.boxplot(column=['job_satisfaction_rate'])
plt.ylabel("Оценка удовлетворенности") 
plt.show()
```


    
![png](output_60_0.png)
    



```python
test_target_job_satisfaction.boxplot(column=['job_satisfaction_rate'])
plt.ylabel("Оценка удовлетворенности") 
plt.show()
```


    
![png](output_61_0.png)
    


Оценка сотрудников не в критическом состоянии. Это радует!
Но как видими, оценка в целевом датасете выше. 

### 1.4. Корреляционный анализ

Посмотрим на матирцу корреляции.


```python
matrix_corr = train_job_satisfaction.drop('id', axis=1).phik_matrix()
plt.figure(figsize=(10, 10))
sns.heatmap(matrix_corr, annot=True, cmap='magma')
plt.title('Матрица корреляции')
plt.show()
```

    interval columns not set, guessing: ['employment_years', 'supervisor_evaluation', 'salary', 'job_satisfaction_rate']
    


    
![png](output_65_1.png)
    



```python
full_test = test_target_job_satisfaction.merge(test_features, on='id', how='left')
matrix_corr = full_test.drop('id', axis=1).phik_matrix()
plt.figure(figsize=(10, 10))
sns.heatmap(matrix_corr, annot=True, cmap='magma')
plt.title('Матрица корреляции')
plt.show()
```

    interval columns not set, guessing: ['job_satisfaction_rate', 'employment_years', 'supervisor_evaluation', 'salary']
    


    
![png](output_66_1.png)
    


**Вывод:** в данных в некоторых признаках присутствует дисбаланс. В целом, распределение хорошее, есть некоторые моменты, на которые мы уже указали при анализе. Целевой признак коррелирует с только `supervisor_evaluation`, `last_year_violations`. Необычно, однако. Также датасеты train_job_satisfaction и test_features+test_target_job_satisfaction несильно отличаются, так что модель должна адекватно обучиться в будущем. Также мы обнаружили пробелы в данных, заполнили их значением nan.

## 2. Построение пайплайна и обучение модели


```python
full_test.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job_satisfaction_rate</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130604</td>
      <td>0.74</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>34800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>825977</td>
      <td>0.75</td>
      <td>marketing</td>
      <td>middle</td>
      <td>high</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>58800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>418490</td>
      <td>0.60</td>
      <td>purchasing</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>555320</td>
      <td>0.72</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>34800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>826430</td>
      <td>0.08</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>817219</td>
      <td>0.76</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>31200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>269033</td>
      <td>0.64</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>27600</td>
    </tr>
    <tr>
      <th>7</th>
      <td>962356</td>
      <td>0.38</td>
      <td>technology</td>
      <td>middle</td>
      <td>high</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>56400</td>
    </tr>
    <tr>
      <th>8</th>
      <td>649052</td>
      <td>0.14</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>45600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>532834</td>
      <td>0.59</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>38400</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_test.isna().sum()
```




    id                       0
    job_satisfaction_rate    0
    dept                     3
    level                    1
    workload                 1
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    dtype: int64




```python
full_test.dropna(inplace=True)
full_test.isna().sum()
```




    id                       0
    job_satisfaction_rate    0
    dept                     0
    level                    0
    workload                 0
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    dtype: int64




```python
full_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1995 entries, 0 to 1999
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   id                     1995 non-null   int64  
     1   job_satisfaction_rate  1995 non-null   float64
     2   dept                   1995 non-null   object 
     3   level                  1995 non-null   object 
     4   workload               1995 non-null   object 
     5   employment_years       1995 non-null   int64  
     6   last_year_promo        1995 non-null   object 
     7   last_year_violations   1995 non-null   object 
     8   supervisor_evaluation  1995 non-null   int64  
     9   salary                 1995 non-null   int64  
    dtypes: float64(1), int64(4), object(5)
    memory usage: 171.4+ KB
    

### 2.1. Разбиение на обучающую и тестовую выборки


```python
X_train = train_job_satisfaction.drop(['id', 'job_satisfaction_rate'], axis=1)
y_train = train_job_satisfaction['job_satisfaction_rate']
X_test = full_test.drop(['id', 'job_satisfaction_rate'], axis=1)
y_test = full_test['job_satisfaction_rate']
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>1</td>
      <td>24000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hr</td>
      <td>junior</td>
      <td>high</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>38400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>12000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>technology</td>
      <td>junior</td>
      <td>low</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>18000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>22800</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>34800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>marketing</td>
      <td>middle</td>
      <td>high</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>58800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>purchasing</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>34800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>30000</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2. Подготовка данных и построение пайплайна


```python
ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
ord_columns = ['level', 'workload']
num_columns = ['employment_years', 'supervisor_evaluation', 'salary']
```


```python
ohe_pipe = Pipeline(
[
    ('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')), 
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])
```


```python
ord_pipe = Pipeline(
    [
        ('simpleImputer_before_ord',
        SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        
        ('ord',
        OrdinalEncoder(categories=[
            ['junior', 'middle', 'sinior'],
            ['low', 'medium', 'high']],
                       handle_unknown='use_encoded_value',
                       unknown_value=np.nan)
        ),
        
        ('simpleImputer_after_ord',
        SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        )
    ]
)
```


```python
data_preprocessor = ColumnTransformer(
[
    ('ohe', ohe_pipe, ohe_columns),
    ('ord', ord_pipe, ord_columns),
    ('num', StandardScaler(), num_columns)
],
    remainder='passthrough'
)
```


```python
pipe_final = Pipeline([
('preprocessor', data_preprocessor),
('models', LinearRegression())
])
```


```python
param_grid = [

    {
        'models': [LinearRegression()],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']   
    },

    {
        'models': [DecisionTreeRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(5, 20),
        'models__max_features': range(5, 20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    }
]
```


```python
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
```


```python
smape_scorer = make_scorer(score_func=smape, greater_is_better=False)
```

### 2.3.Обучение модели


```python
grid_search = GridSearchCV(
    pipe_final,
    param_grid,
    n_jobs=-1,
    cv=5,
    scoring=smape_scorer
)
grid_search.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;ohe&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;ohe&#x27;,
                                                                                          OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                                        handle_unknown=&#x27;ignore&#x27;,
                                                                                                        sparse_output=False))]),
                                                                         [&#x27;dept&#x27;,
                                                                          &#x27;last_year_promo&#x27;,
                                                                          &#x27;last_year_violations&#x27;]),
                                                                        (&#x27;ord&#x27;,
                                                                         Pipeline(...
             param_grid=[{&#x27;models&#x27;: [LinearRegression()],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler(),
                                                &#x27;passthrough&#x27;]},
                         {&#x27;models&#x27;: [DecisionTreeRegressor(random_state=42)],
                          &#x27;models__max_depth&#x27;: range(5, 20),
                          &#x27;models__max_features&#x27;: range(5, 20),
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler(),
                                                &#x27;passthrough&#x27;]}],
             scoring=make_scorer(smape, greater_is_better=False, response_method=&#x27;predict&#x27;))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;ohe&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;ohe&#x27;,
                                                                                          OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                                        handle_unknown=&#x27;ignore&#x27;,
                                                                                                        sparse_output=False))]),
                                                                         [&#x27;dept&#x27;,
                                                                          &#x27;last_year_promo&#x27;,
                                                                          &#x27;last_year_violations&#x27;]),
                                                                        (&#x27;ord&#x27;,
                                                                         Pipeline(...
             param_grid=[{&#x27;models&#x27;: [LinearRegression()],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler(),
                                                &#x27;passthrough&#x27;]},
                         {&#x27;models&#x27;: [DecisionTreeRegressor(random_state=42)],
                          &#x27;models__max_depth&#x27;: range(5, 20),
                          &#x27;models__max_features&#x27;: range(5, 20),
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler(),
                                                &#x27;passthrough&#x27;]}],
             scoring=make_scorer(smape, greater_is_better=False, response_method=&#x27;predict&#x27;))</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;ohe&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;ohe&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  [&#x27;dept&#x27;, &#x27;last_year_promo&#x27;,
                                                   &#x27;last_year_violations&#x27;]),
                                                 (&#x27;ord&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_befor...
                                                                                               &#x27;middle&#x27;,
                                                                                               &#x27;sinior&#x27;],
                                                                                              [&#x27;low&#x27;,
                                                                                               &#x27;medium&#x27;,
                                                                                               &#x27;high&#x27;]],
                                                                                  handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                  unknown_value=nan)),
                                                                  (&#x27;simpleImputer_after_ord&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;))]),
                                                  [&#x27;level&#x27;, &#x27;workload&#x27;]),
                                                 (&#x27;num&#x27;, MinMaxScaler(),
                                                  [&#x27;employment_years&#x27;,
                                                   &#x27;supervisor_evaluation&#x27;,
                                                   &#x27;salary&#x27;])])),
                (&#x27;models&#x27;,
                 DecisionTreeRegressor(max_depth=13, max_features=11,
                                       random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;preprocessor: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;ohe&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ohe&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse_output=False))]),
                                 [&#x27;dept&#x27;, &#x27;last_year_promo&#x27;,
                                  &#x27;last_year_violations&#x27;]),
                                (&#x27;ord&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_before_ord&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ord&#x27;,
                                                  OrdinalEncoder(categories=[[&#x27;junior&#x27;,
                                                                              &#x27;middle&#x27;,
                                                                              &#x27;sinior&#x27;],
                                                                             [&#x27;low&#x27;,
                                                                              &#x27;medium&#x27;,
                                                                              &#x27;high&#x27;]],
                                                                 handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=nan)),
                                                 (&#x27;simpleImputer_after_ord&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;))]),
                                 [&#x27;level&#x27;, &#x27;workload&#x27;]),
                                (&#x27;num&#x27;, MinMaxScaler(),
                                 [&#x27;employment_years&#x27;, &#x27;supervisor_evaluation&#x27;,
                                  &#x27;salary&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">ohe</label><div class="sk-toggleable__content fitted"><pre>[&#x27;dept&#x27;, &#x27;last_year_promo&#x27;, &#x27;last_year_violations&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">ord</label><div class="sk-toggleable__content fitted"><pre>[&#x27;level&#x27;, &#x27;workload&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OrdinalEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OrdinalEncoder(categories=[[&#x27;junior&#x27;, &#x27;middle&#x27;, &#x27;sinior&#x27;],
                           [&#x27;low&#x27;, &#x27;medium&#x27;, &#x27;high&#x27;]],
               handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=nan)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">num</label><div class="sk-toggleable__content fitted"><pre>[&#x27;employment_years&#x27;, &#x27;supervisor_evaluation&#x27;, &#x27;salary&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;MinMaxScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html">?<span>Documentation for MinMaxScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>MinMaxScaler()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;DecisionTreeRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeRegressor.html">?<span>Documentation for DecisionTreeRegressor</span></a></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeRegressor(max_depth=13, max_features=11, random_state=42)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
print ('Метрика SMAPE лучшей модели на тренировочной выборке:', round(grid_search.best_score_*(-1), 2))
```

    Метрика SMAPE лучшей модели на тренировочной выборке: 15.01
    

Мы получили идентичные параметры.

### 2.3. Тестирование и вывод


```python
print(f'Метрика SMAPE лучшей модели на тестовой выборке: {round(smape(y_test, grid_search.best_estimator_.predict(X_test)),2)}')
```

    Метрика SMAPE лучшей модели на тестовой выборке: 14.3
    

Можно сделать следующие выводы: 
* Лучшей моделью оказалась DecisionTreeRegressor со следующими параметрами max_depth=13, max_features=11.
* Метрика SMAPE лучшей модели на тренировочной выборке составила 15.01.
* Метрика SMAPE лучшей модели на тестовой выборке составила 14.3.

Таким образом, мы обучили модель предсказывать уровень удовлетворённости сотрудника с точностью 14.3/15.00.

# Задача 2: Прогнозирование оттока

## 1 Загрузка и предобработка данных

### 1.1. Загрузка данных


```python
train_quit = pd.read_csv(r'~/train_quit.csv')
test_features = pd.read_csv(r'~/test_features.csv')
test_target_quit = pd.read_csv(r'~/test_target_quit.csv')
```


```python
train_quit.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>quit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>723290</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>54000</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>814010</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>27600</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>155091</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>1</td>
      <td>37200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>257132</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>24000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>910140</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>25200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>699916</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>18000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>417070</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>8</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>44400</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>165489</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>19200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>996399</td>
      <td>marketing</td>
      <td>middle</td>
      <td>low</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>25200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>613206</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>45600</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_quit.info()
display(train_quit.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4000 entries, 0 to 3999
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   id                     4000 non-null   int64 
     1   dept                   4000 non-null   object
     2   level                  4000 non-null   object
     3   workload               4000 non-null   object
     4   employment_years       4000 non-null   int64 
     5   last_year_promo        4000 non-null   object
     6   last_year_violations   4000 non-null   object
     7   supervisor_evaluation  4000 non-null   int64 
     8   salary                 4000 non-null   int64 
     9   quit                   4000 non-null   object
    dtypes: int64(4), object(6)
    memory usage: 312.6+ KB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>employment_years</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4000.000000</td>
      <td>4000.000000</td>
      <td>4000.000000</td>
      <td>4000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>552099.283750</td>
      <td>3.701500</td>
      <td>3.474750</td>
      <td>33805.800000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>260158.031387</td>
      <td>2.541852</td>
      <td>1.004049</td>
      <td>15152.415163</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100222.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>12000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>327785.750000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>22800.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>546673.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>781497.750000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>43200.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999915.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>96000.000000</td>
    </tr>
  </tbody>
</table>
</div>


Данные хорошие, без пропусков. 


```python
test_features.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>485046</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>28800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>686555</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>467458</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>418655</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>789145</td>
      <td>hr</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>40800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>429973</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>medium</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>42000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>850699</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>26400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>500791</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>49200</td>
    </tr>
    <tr>
      <th>8</th>
      <td>767867</td>
      <td>marketing</td>
      <td>middle</td>
      <td>high</td>
      <td>3</td>
      <td>no</td>
      <td>yes</td>
      <td>4</td>
      <td>62400</td>
    </tr>
    <tr>
      <th>9</th>
      <td>937235</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>26400</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_features.info()
display(test_features.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 9 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   id                     2000 non-null   int64 
     1   dept                   1998 non-null   object
     2   level                  1999 non-null   object
     3   workload               2000 non-null   object
     4   employment_years       2000 non-null   int64 
     5   last_year_promo        2000 non-null   object
     6   last_year_violations   2000 non-null   object
     7   supervisor_evaluation  2000 non-null   int64 
     8   salary                 2000 non-null   int64 
    dtypes: int64(4), object(5)
    memory usage: 140.8+ KB
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>employment_years</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>552765.213500</td>
      <td>3.666500</td>
      <td>3.526500</td>
      <td>34066.800000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>253851.326129</td>
      <td>2.537222</td>
      <td>0.996892</td>
      <td>15398.436729</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100298.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>12000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>339052.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>22800.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>550793.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>765763.750000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>43200.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999029.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>96000.000000</td>
    </tr>
  </tbody>
</table>
</div>


Присутстсвуют пропуски в данных.


```python
test_target_quit.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>quit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>999029</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>372846</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>726767</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>490105</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>416898</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>223063</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>810370</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>998900</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>578329</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>648850</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_target_quit.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      2000 non-null   int64 
     1   quit    2000 non-null   object
    dtypes: int64(1), object(1)
    memory usage: 31.4+ KB
    

Данные без пропусков.

**Вывод:** данные чистые, столбцы приведены к змеиному регистру, присутствуют пропуски в датасете test_features, тип данных соответствует нужному.

### 1.2. Предобработка данных

Данные уже приведены к змеиному регистру, проверим на пропуски еще раз и на дубли.


```python
print('Количество пропусков в train_quit:\n', train_quit.isna().sum())
print()
print('Количество пропусков в test_features:\n', test_features.isna().sum())
print('Количество пропусков в test_target_quit:\n ', test_target_quit.isna().sum())
```

    Количество пропусков в train_quit:
     id                       0
    dept                     0
    level                    0
    workload                 0
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    quit                     0
    dtype: int64
    
    Количество пропусков в test_features:
     id                       0
    dept                     2
    level                    1
    workload                 0
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    dtype: int64
    Количество пропусков в test_target_quit:
      id      0
    quit    0
    dtype: int64
    


```python
print('Количество дубликатов в train_quit: ', train_quit.duplicated().sum())
print('Количество дубликатов в test_features: ', test_features.duplicated().sum())
print('Количество дубликатов в test_target_quit: ', test_target_quit.duplicated().sum())
```

    Количество дубликатов в train_quit:  0
    Количество дубликатов в test_features:  0
    Количество дубликатов в test_target_quit:  0
    


```python
train_quit[train_quit.duplicated(
    subset=['dept', 'level', 'workload', 'employment_years', 'last_year_promo', 
            'last_year_violations', 'supervisor_evaluation','salary', 'quit'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>quit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>873412</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>31200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>152</th>
      <td>749683</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>8</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>175</th>
      <td>689526</td>
      <td>marketing</td>
      <td>middle</td>
      <td>low</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>30000</td>
      <td>no</td>
    </tr>
    <tr>
      <th>205</th>
      <td>786443</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>low</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
      <td>no</td>
    </tr>
    <tr>
      <th>254</th>
      <td>362060</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>12000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3990</th>
      <td>632886</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>12000</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3993</th>
      <td>387733</td>
      <td>marketing</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>44400</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>588809</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>26400</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>672059</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>52800</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>853842</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>27600</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>1413 rows × 10 columns</p>
</div>



Удручающая ситуация, ничего трогать не будем...


```python
test_features[test_features.duplicated(
    subset=['dept', 'level', 'workload', 'employment_years', 'last_year_promo', 
            'last_year_violations', 'supervisor_evaluation','salary'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>523542</td>
      <td>marketing</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>16800</td>
    </tr>
    <tr>
      <th>56</th>
      <td>582128</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>62</th>
      <td>482624</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>4</td>
      <td>yes</td>
      <td>no</td>
      <td>2</td>
      <td>44400</td>
    </tr>
    <tr>
      <th>111</th>
      <td>770429</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>113</th>
      <td>761490</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>15600</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>760964</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>21600</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>380255</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>8</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>38400</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>393147</td>
      <td>marketing</td>
      <td>junior</td>
      <td>low</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>20400</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>305653</td>
      <td>technology</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>14400</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>160233</td>
      <td>technology</td>
      <td>middle</td>
      <td>low</td>
      <td>8</td>
      <td>no</td>
      <td>no</td>
      <td>1</td>
      <td>32400</td>
    </tr>
  </tbody>
</table>
<p>557 rows × 9 columns</p>
</div>




```python
test_features['dept'].value_counts()
```




    sales         763
    technology    455
    marketing     279
    purchasing    273
    hr            227
                    1
    Name: dept, dtype: int64




```python
test_features['workload'].value_counts()
```




    medium    1043
    low        593
    high       363
                 1
    Name: workload, dtype: int64




```python
test_features['dept'] = test_features['dept'].replace(' ', np.nan)
test_features['workload'] = test_features['workload'].replace(' ', np.nan)
```

**Вывод:** мы обнаружили пропуски в данных и дубли, которые исправим в пайплайне. 

### 1.3. Исследовательский и корреляционный анализ данных


```python
cat_plot(train_quit, 'dept')
```


    array(['sales', 'purchasing', 'marketing', 'technology', 'hr'],
          dtype=object)



    
![png](output_119_1.png)
    


Большинство сотрудников из отдела продаж.


```python
cat_plot(train_quit, 'level')
```


    array(['middle', 'junior', 'sinior'], dtype=object)



    
![png](output_121_1.png)
    


Преобладает уровень junior. Хороший результат. 


```python
cat_plot(train_quit, 'workload')
```


    array(['high', 'medium', 'low'], dtype=object)



    
![png](output_123_1.png)
    


Нужно будем рассмотреть отдельно высокую загруженность и среднуюю.


```python
cat_plot(train_quit, 'last_year_promo')
```


    array(['no', 'yes'], dtype=object)



    
![png](output_125_1.png)
    



```python
cat_plot(train_quit, 'last_year_violations')
```


    array(['no', 'yes'], dtype=object)



    
![png](output_126_1.png)
    



```python
hist_plot(train_quit, 'employment_years')  
```


    array([ 2,  5,  1,  8,  4,  7,  3,  9,  6, 10], dtype=int64)



    
![png](output_127_1.png)
    



```python
cat_plot(train_quit, 'supervisor_evaluation')
```


    array([4, 1, 3, 5, 2], dtype=int64)



    
![png](output_128_1.png)
    


Хм, в теории, неплохие оценки.


```python
hist_plot(train_quit, 'salary')
```


    array([54000, 27600, 37200, 24000, 25200, 18000, 44400, 19200, 45600,
           57600, 33600, 16800, 22800, 26400, 82800, 32400, 39600, 30000,
           46800, 12000, 15600, 58800, 60000, 66000, 21600, 38400, 62400,
           40800, 56400, 34800, 28800, 52800, 20400, 36000, 61200, 48000,
           43200, 73200, 31200, 78000, 64800, 72000, 94800, 96000, 63600,
           79200, 55200, 42000, 49200, 50400, 14400, 13200, 51600, 67200,
           88800, 68400, 69600, 70800, 84000, 81600, 87600, 75600, 91200,
           76800, 74400, 80400, 85200, 86400, 92400], dtype=int64)



    
![png](output_130_1.png)
    


Зарплата оставляет желать лучшего.


```python
cat_plot(train_quit, 'quit')
```


    array(['no', 'yes'], dtype=object)



    
![png](output_132_1.png)
    


Одна треть - впечатляет.

**Рассмотрим отдельно сотрудников, которые покинули компанию.**

Посмотрим на матрицу корреляции.


```python
matrix_corr = train_quit.drop('id', axis=1).phik_matrix()
plt.figure(figsize=(10, 10))
sns.heatmap(matrix_corr, annot=True, cmap='magma')
plt.title('Матрица корреляции')
plt.show()
```

    interval columns not set, guessing: ['employment_years', 'supervisor_evaluation', 'salary']
    


    
![png](output_136_1.png)
    


Интересный результат, попробуем рассмотреть поподробнее.


```python
train_quit_yes = train_quit[train_quit['quit'] == 'yes']
```


```python
train_quit_yes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>quit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>257132</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>24000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>699916</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>3</td>
      <td>18000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>613206</td>
      <td>technology</td>
      <td>middle</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>45600</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>24</th>
      <td>468145</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>30000</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>25</th>
      <td>982346</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>30000</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(train_quit_yes['dept'].value_counts())
cat_plot(train_quit_yes, 'dept')
```

    sales         407
    technology    276
    purchasing    166
    marketing     163
    hr            116
    Name: dept, dtype: int64
    


    array(['sales', 'technology', 'marketing', 'purchasing', 'hr'],
          dtype=object)



    
![png](output_140_2.png)
    


Большинство уволенных сотрудников работали в sales. 


```python
cat_plot(train_quit_yes, 'workload')
```


    array(['medium', 'low', 'high'], dtype=object)



    
![png](output_142_1.png)
    


Это странно. Было бы понятно, если бы эти сотрудники имели высокую загруженность, тут же совершенно другая картина.


```python
plt.figure(figsize=(16, 10))
plt.title("График распределения зарплат уволенных и оставшихся сотрудников", 
          fontsize=16)
plt.xlim(train_quit['salary'].min(), train_quit['salary'].max())
ax = train_quit['salary'][train_quit['quit'] == 'yes'].plot.kde(color='red', label="Уволенные")
ax = train_quit['salary'][train_quit['quit'] == 'no'].plot.kde(color='green', label="Оставшиеся")
ax.set_ylabel("Плотность", fontsize=14)
ax.set_xlabel("Количество", fontsize=14)
plt.legend()
plt.show()
```


    
![png](output_144_0.png)
    


Тут уже все логично. Их могла мотивировать уволиться большая заработная плата в другой компании (например).


```python
plt.figure(figsize=(16, 10))
plt.title("График распределения supervisor_evaluation уволенных и оставшихся сотрудников", 
          fontsize=16)
plt.xlim(train_quit['supervisor_evaluation'].min(), train_quit['supervisor_evaluation'].max())
ax = train_quit['supervisor_evaluation'][train_quit['quit'] == 'yes'].plot.kde(color='red', label="Уволенные")
ax = train_quit['supervisor_evaluation'][train_quit['quit'] == 'no'].plot.kde(color='green', label="Оставшиеся")
ax.set_ylabel("Плотность", fontsize=14)
ax.set_xlabel("Количество", fontsize=14)
plt.legend()
plt.show()
```


    
![png](output_146_0.png)
    



```python
grouped_data = train_quit.groupby(['workload', 'quit']).size().unstack()
ax = grouped_data.plot(kind='bar', color=['red', 'green'],  figsize=(16,10),  width=0.8,
                      title="Распределение workload для уволенных и оставшихся сотрудников") 
ax.title.set_size(20)
ax.set_ylabel("Количество сотрудников", fontsize=14)
plt.legend(['Уволенные', 'Оставшиеся'])
plt.xticks(rotation=0) 
plt.show()
```


    
![png](output_147_0.png)
    


Как видно, покидали компанию большинство сотрудников, которые имели среднюю нагрузку.


```python
grouped_data = train_quit.groupby(['level', 'quit']).size().unstack()
ax = grouped_data.plot(kind='bar', color=['red', 'green'],  figsize=(16,10),  width=0.8,
                      title="Распределение level для уволенных и оставшихся сотрудников") 
ax.title.set_size(20)
ax.set_ylabel("Количество сотрудников", fontsize=14)
plt.legend(['Уволенные', 'Оставшиеся'])
plt.xticks(rotation=0) 
plt.show()
```


    
![png](output_149_0.png)
    


Уволившиеся сотрудники имели разный опыт работы, но в основном до 4 лет. Возможно, тут они планировали просто набраться опыта, достигнуть уровнень мидл и перейти в другую компанию.

**Вывод**: большинство сотрудников, покинувших компанию, имели следующие характеристики:
* работали в отделе продаж;
* имели небольшую зарплату;
* покидали компанию после 1-4 лет работы;
* имели среднуюю или низкую загруженность;
* в основном, это middle и junior.

Последний пункт янво напрягает, но, возможно, им было скучно? и/или хотелось чего-то большего? 

### 1.4. Добавление нового входного признака

Добавим признак job_satisfaction_rate, предсказанный лучшей моделью ранее, к входным признакам датасета train_quit.


```python
train_quit['job_satisfaction_rate'] = grid_search.best_estimator_.predict(train_quit)
```


```python
test_features['job_satisfaction_rate'] = grid_search.best_estimator_.predict(test_features)
```


```python
test_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>485046</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>28800</td>
      <td>0.872000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>686555</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>30000</td>
      <td>0.668621</td>
    </tr>
    <tr>
      <th>2</th>
      <td>467458</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
      <td>0.657143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>418655</td>
      <td>sales</td>
      <td>middle</td>
      <td>low</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
      <td>0.655000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>789145</td>
      <td>hr</td>
      <td>middle</td>
      <td>medium</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>40800</td>
      <td>0.824127</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_quit.sort_values(by='id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>quit</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2600</th>
      <td>100222</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>20400</td>
      <td>yes</td>
      <td>0.340000</td>
    </tr>
    <tr>
      <th>717</th>
      <td>100459</td>
      <td>purchasing</td>
      <td>junior</td>
      <td>medium</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>21600</td>
      <td>yes</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2455</th>
      <td>100469</td>
      <td>marketing</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>28800</td>
      <td>no</td>
      <td>0.659545</td>
    </tr>
    <tr>
      <th>1592</th>
      <td>100601</td>
      <td>technology</td>
      <td>middle</td>
      <td>high</td>
      <td>4</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>68400</td>
      <td>no</td>
      <td>0.667586</td>
    </tr>
    <tr>
      <th>2657</th>
      <td>100858</td>
      <td>sales</td>
      <td>junior</td>
      <td>medium</td>
      <td>2</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>25200</td>
      <td>yes</td>
      <td>0.823636</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2194</th>
      <td>998517</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>19200</td>
      <td>no</td>
      <td>0.470000</td>
    </tr>
    <tr>
      <th>3701</th>
      <td>999003</td>
      <td>hr</td>
      <td>middle</td>
      <td>low</td>
      <td>3</td>
      <td>no</td>
      <td>no</td>
      <td>2</td>
      <td>24000</td>
      <td>yes</td>
      <td>0.170000</td>
    </tr>
    <tr>
      <th>3364</th>
      <td>999158</td>
      <td>purchasing</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>yes</td>
      <td>1</td>
      <td>21600</td>
      <td>yes</td>
      <td>0.060000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>999835</td>
      <td>sales</td>
      <td>junior</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>18000</td>
      <td>no</td>
      <td>0.865000</td>
    </tr>
    <tr>
      <th>2120</th>
      <td>999915</td>
      <td>hr</td>
      <td>middle</td>
      <td>low</td>
      <td>5</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>20400</td>
      <td>no</td>
      <td>0.300000</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 11 columns</p>
</div>




```python
plt.figure(figsize=(16, 10))
plt.title("График распределения job_satisfaction_rate уволенных и оставшихся сотрудников", 
          fontsize=16)
plt.xlim(train_quit['job_satisfaction_rate'].min(), train_quit['job_satisfaction_rate'].max())
ax = train_quit['job_satisfaction_rate'][train_quit['quit'] == 'yes'].plot.kde(color='red', label="Уволенные")
ax = train_quit['job_satisfaction_rate'][train_quit['quit'] == 'no'].plot.kde(color='green', label="Оставшиеся")
ax.set_xlabel("Количество", fontsize=14)
ax.set_ylabel("Плотность", fontsize=14)
plt.legend()
plt.show()
```


    
![png](output_158_0.png)
    


**Вывод:** мы добавили входной признак. Как мы видим, уровень удовлетворённости сотрудника работой в компании влияет на принятие его решения.

## 2 Построение пайплайна и обучение модели


```python
full_test = test_target_quit.merge(test_features, on='id', how='left')
full_test.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>quit</th>
      <th>dept</th>
      <th>level</th>
      <th>workload</th>
      <th>employment_years</th>
      <th>last_year_promo</th>
      <th>last_year_violations</th>
      <th>supervisor_evaluation</th>
      <th>salary</th>
      <th>job_satisfaction_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>999029</td>
      <td>yes</td>
      <td>technology</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>31200</td>
      <td>0.335000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>372846</td>
      <td>no</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>10</td>
      <td>no</td>
      <td>yes</td>
      <td>2</td>
      <td>32400</td>
      <td>0.230000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>726767</td>
      <td>no</td>
      <td>marketing</td>
      <td>middle</td>
      <td>low</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>20400</td>
      <td>0.670000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>490105</td>
      <td>no</td>
      <td>purchasing</td>
      <td>middle</td>
      <td>low</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>19200</td>
      <td>0.695000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>416898</td>
      <td>yes</td>
      <td>purchasing</td>
      <td>junior</td>
      <td>low</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>12000</td>
      <td>0.510000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>223063</td>
      <td>no</td>
      <td>sales</td>
      <td>middle</td>
      <td>medium</td>
      <td>6</td>
      <td>no</td>
      <td>no</td>
      <td>4</td>
      <td>38400</td>
      <td>0.824127</td>
    </tr>
    <tr>
      <th>6</th>
      <td>810370</td>
      <td>no</td>
      <td>hr</td>
      <td>junior</td>
      <td>medium</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>26400</td>
      <td>0.688750</td>
    </tr>
    <tr>
      <th>7</th>
      <td>998900</td>
      <td>no</td>
      <td>marketing</td>
      <td>middle</td>
      <td>medium</td>
      <td>7</td>
      <td>no</td>
      <td>no</td>
      <td>3</td>
      <td>45600</td>
      <td>0.438400</td>
    </tr>
    <tr>
      <th>8</th>
      <td>578329</td>
      <td>no</td>
      <td>sales</td>
      <td>sinior</td>
      <td>medium</td>
      <td>10</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>46800</td>
      <td>0.752000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>648850</td>
      <td>no</td>
      <td>sales</td>
      <td>middle</td>
      <td>high</td>
      <td>9</td>
      <td>no</td>
      <td>no</td>
      <td>5</td>
      <td>57600</td>
      <td>0.846842</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_test.isna().sum()
```




    id                       0
    quit                     0
    dept                     3
    level                    1
    workload                 1
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    job_satisfaction_rate    0
    dtype: int64




```python
full_test.dropna(inplace=True)
full_test.isna().sum()
```




    id                       0
    quit                     0
    dept                     0
    level                    0
    workload                 0
    employment_years         0
    last_year_promo          0
    last_year_violations     0
    supervisor_evaluation    0
    salary                   0
    job_satisfaction_rate    0
    dtype: int64



**Вывод:** мы подготовили тестовый датасет.

### 2.1. Разбиение на обучающую и тестовую выборки


```python
X_train = train_quit.drop(['id', 'quit'], axis=1)
y_train = train_quit['quit']

X_test = full_test.drop(['id', 'quit'], axis=1)
y_test = full_test['quit']
```

### 2.2. Подготовка данных и построение пайплайна


```python
ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
ord_columns = ['level', 'workload']
num_columns = ['employment_years', 'supervisor_evaluation', 'salary', 'job_satisfaction_rate']
```


```python
ohe_pipe = Pipeline(
[
    ('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')), 
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])
```


```python
ord_pipe = Pipeline(
    [
        ('simpleImputer_before_ord',
        SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        
        ('ord',
        OrdinalEncoder(categories=[
            ['junior', 'middle', 'sinior'],
            ['low', 'medium', 'high']],
                       handle_unknown='use_encoded_value',
                       unknown_value=np.nan)
        ),
        
        ('simpleImputer_after_ord',
        SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        )
    ]
)
```


```python
data_preprocessor = ColumnTransformer(
[
    ('ohe', ohe_pipe, ohe_columns),
    ('ord', ord_pipe, ord_columns),
    ('num', StandardScaler(), num_columns)
],
    remainder='passthrough'
)
```


```python
pipe_final = Pipeline([
('preprocessor', data_preprocessor),
('models', DecisionTreeClassifier())
])
```


```python
param_grid = [
    {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 20),
        'models__max_features': range(2,20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    
    {
        'models': [KNeighborsClassifier()],
        'models__n_neighbors': range(2,20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']   
    },

    {
        'models': [RandomForestClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 20),
        'models__max_features': range(2,20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
                   
    },
    {
        'models': [SVC(random_state=RANDOM_STATE, kernel='poly')],
        'models__degree': range(2, 5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
    }
]
```

### 2.3. Обучение модели


```python
randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
randomized_search.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,
                   estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                              ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                                transformers=[(&#x27;ohe&#x27;,
                                                                               Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                                SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                               (&#x27;ohe&#x27;,
                                                                                                OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                                              handle_unknown=&#x27;ignore&#x27;,
                                                                                                              sparse_output=False))]),
                                                                               [&#x27;dept&#x27;,
                                                                                &#x27;last_year_promo&#x27;,
                                                                                &#x27;last_year_violations&#x27;]),
                                                                              (&#x27;ord&#x27;,
                                                                               Pip...
                                        {&#x27;models&#x27;: [RandomForestClassifier(random_state=42)],
                                         &#x27;models__max_depth&#x27;: range(2, 20),
                                         &#x27;models__max_features&#x27;: range(2, 20),
                                         &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                               MinMaxScaler(),
                                                               &#x27;passthrough&#x27;]},
                                        {&#x27;models&#x27;: [SVC(kernel=&#x27;poly&#x27;,
                                                        random_state=42)],
                                         &#x27;models__degree&#x27;: range(2, 5),
                                         &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                               MinMaxScaler(),
                                                               &#x27;passthrough&#x27;]}],
                   random_state=42, scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomizedSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html">?<span>Documentation for RandomizedSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomizedSearchCV(cv=5,
                   estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                              ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                                transformers=[(&#x27;ohe&#x27;,
                                                                               Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                                SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                               (&#x27;ohe&#x27;,
                                                                                                OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                                              handle_unknown=&#x27;ignore&#x27;,
                                                                                                              sparse_output=False))]),
                                                                               [&#x27;dept&#x27;,
                                                                                &#x27;last_year_promo&#x27;,
                                                                                &#x27;last_year_violations&#x27;]),
                                                                              (&#x27;ord&#x27;,
                                                                               Pip...
                                        {&#x27;models&#x27;: [RandomForestClassifier(random_state=42)],
                                         &#x27;models__max_depth&#x27;: range(2, 20),
                                         &#x27;models__max_features&#x27;: range(2, 20),
                                         &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                               MinMaxScaler(),
                                                               &#x27;passthrough&#x27;]},
                                        {&#x27;models&#x27;: [SVC(kernel=&#x27;poly&#x27;,
                                                        random_state=42)],
                                         &#x27;models__degree&#x27;: range(2, 5),
                                         &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                               MinMaxScaler(),
                                                               &#x27;passthrough&#x27;]}],
                   random_state=42, scoring=&#x27;roc_auc&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;ohe&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;ohe&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  [&#x27;dept&#x27;, &#x27;last_year_promo&#x27;,
                                                   &#x27;last_year_violations&#x27;]),
                                                 (&#x27;ord&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_befor...
                                                                                              [&#x27;low&#x27;,
                                                                                               &#x27;medium&#x27;,
                                                                                               &#x27;high&#x27;]],
                                                                                  handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                  unknown_value=nan)),
                                                                  (&#x27;simpleImputer_after_ord&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;))]),
                                                  [&#x27;level&#x27;, &#x27;workload&#x27;]),
                                                 (&#x27;num&#x27;, MinMaxScaler(),
                                                  [&#x27;employment_years&#x27;,
                                                   &#x27;supervisor_evaluation&#x27;,
                                                   &#x27;salary&#x27;,
                                                   &#x27;job_satisfaction_rate&#x27;])])),
                (&#x27;models&#x27;,
                 RandomForestClassifier(max_depth=10, max_features=2,
                                        random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;preprocessor: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;ohe&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ohe&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse_output=False))]),
                                 [&#x27;dept&#x27;, &#x27;last_year_promo&#x27;,
                                  &#x27;last_year_violations&#x27;]),
                                (&#x27;ord&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_before_ord&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ord&#x27;,
                                                  OrdinalEncoder(categories=[[&#x27;junior&#x27;,
                                                                              &#x27;middle&#x27;,
                                                                              &#x27;sinior&#x27;],
                                                                             [&#x27;low&#x27;,
                                                                              &#x27;medium&#x27;,
                                                                              &#x27;high&#x27;]],
                                                                 handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=nan)),
                                                 (&#x27;simpleImputer_after_ord&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;))]),
                                 [&#x27;level&#x27;, &#x27;workload&#x27;]),
                                (&#x27;num&#x27;, MinMaxScaler(),
                                 [&#x27;employment_years&#x27;, &#x27;supervisor_evaluation&#x27;,
                                  &#x27;salary&#x27;, &#x27;job_satisfaction_rate&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">ohe</label><div class="sk-toggleable__content fitted"><pre>[&#x27;dept&#x27;, &#x27;last_year_promo&#x27;, &#x27;last_year_violations&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">ord</label><div class="sk-toggleable__content fitted"><pre>[&#x27;level&#x27;, &#x27;workload&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OrdinalEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OrdinalEncoder(categories=[[&#x27;junior&#x27;, &#x27;middle&#x27;, &#x27;sinior&#x27;],
                           [&#x27;low&#x27;, &#x27;medium&#x27;, &#x27;high&#x27;]],
               handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=nan)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">num</label><div class="sk-toggleable__content fitted"><pre>[&#x27;employment_years&#x27;, &#x27;supervisor_evaluation&#x27;, &#x27;salary&#x27;, &#x27;job_satisfaction_rate&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;MinMaxScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html">?<span>Documentation for MinMaxScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>MinMaxScaler()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">remainder</label><div class="sk-toggleable__content fitted"><pre>[]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">passthrough</label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(max_depth=10, max_features=2, random_state=42)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели на тренировочной выборке:', round(randomized_search.best_score_, 2))
```

    Лучшая модель и её параметры:
    
     Pipeline(steps=[('preprocessor',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('ohe',
                                                      Pipeline(steps=[('simpleImputer_ohe',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       OneHotEncoder(drop='first',
                                                                                     handle_unknown='ignore',
                                                                                     sparse_output=False))]),
                                                      ['dept', 'last_year_promo',
                                                       'last_year_violations']),
                                                     ('ord',
                                                      Pipeline(steps=[('simpleImputer_befor...
                                                                                                  ['low',
                                                                                                   'medium',
                                                                                                   'high']],
                                                                                      handle_unknown='use_encoded_value',
                                                                                      unknown_value=nan)),
                                                                      ('simpleImputer_after_ord',
                                                                       SimpleImputer(strategy='most_frequent'))]),
                                                      ['level', 'workload']),
                                                     ('num', MinMaxScaler(),
                                                      ['employment_years',
                                                       'supervisor_evaluation',
                                                       'salary',
                                                       'job_satisfaction_rate'])])),
                    ('models',
                     RandomForestClassifier(max_depth=10, max_features=2,
                                            random_state=42))])
    Метрика лучшей модели на тренировочной выборке: 0.94
    

### 2.4. Тестирование и вывод


```python
probabilities = randomized_search.predict_proba(X_test)[:,1]
print('Площадь ROC-кривой:', roc_auc_score(y_test, probabilities))
```

    Площадь ROC-кривой: 0.9294086020170214
    

**Вывод:** мы обучили модель, которая предсказала на тренировочной выборке с точносью 0.94,а площадь ROC-кривой составила 0.93. Лучшей моделью оказалась RandomForestClassifier(max_depth=10, max_features=2, random_state=42). На тестовой 0.93. Хороший результат! 

## Общий вывод и рекомендации заказчику

В этом исследовании были достигнуты две поставленные задачи:

**1. Предсказание уровня удовлетворённости сотрудника.**

Для ее достижения мы сделали следующие: 
* загрузили и изучили данные.
* провели предобрабодку.
* провели исследовательский анализ данных.
* построили модель для предсказания уровня удовлетворённости сотрудника. Лучшей моделью оказалась DecisionTreeRegressor со следующими параметрами max_depth=13, max_features=11.
* Метрика SMAPE лучшей модели на тренировочной выборке составила 15.01.
* Метрика SMAPE лучшей модели на тестовой выборке составила 14.3.

Таким образом, мы обучили модель предсказывать уровень удовлетворённости сотрудника с точностью 14.3/15.00.

**2. Предсказание увольнения сотрудника из компании.**

Для ее достижения мы сделали следующие: 
* загрузили и изучили данные.
* провели предобрабодку.
* провели исследовательский анализ данных.
* сотавили портрет сотрудника, который может покинуть компанию. Большинство из них работали в отделе продаж, имели небольшую зарплату, покидали компанию после 1-4 лет работы, имели среднюю или низкую загруженность.
* Добавили признак job_satisfaction_rate, предсказанный лучшей моделью, построенной в первой задаче, к входным признакам датасета train_quit.
* Обнаружили, что уровень удовлетворённости сотрудника работой в компании влияет на принятие его решения.
* Построили модель, которая предсказала на тренировочной выборке с точносью 0.94,а площадь ROC-кривой составила 0.93. Лучшей моделью оказалась RandomForestClassifier с параметрами max_depth=10, max_features=2, random_state=42.
* На тестовой 0.93

Таким образом, видно прямую взаимосвязь между тем, покинет ли сотрудник компанию и его удовлетворённостью работой в этой компании. Судя по анализу, можно повысить зарплату сотрудникам, предложить повышение (заслуженным сотрудникам, разумеется). Возможно, еще стоит поискать какие-нибудь интересные проеты, так как многие уволившиеся сотрудники имели низкую занятность. Можно каким-либо способом оптимизировать работу: предложить удаленку или провести опрос среди сотрудников. Например, как можно улучшить рабочую обстановку. Так ка мы посмотрели среднуюю оценку удовлетворённости (она состовляет 0.54), думаю, ее стоит повысить, чтобы не рисковать ресурсами.


```python

```
