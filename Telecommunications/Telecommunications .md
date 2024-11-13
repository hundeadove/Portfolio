# Телекоммуникации

Оператор связи «ТелеДом» хочет бороться с оттоком клиентов. Для этого его сотрудники начнут предлагать промокоды и специальные условия всем, кто планирует отказаться от услуг связи. Чтобы заранее находить таких пользователей, «ТелеДому» нужна модель, которая будет предсказывать, разорвёт ли абонент договор.

## Описание данных
* Персональная информация.
* Тарифы и услуги: Стационарный телефон, Интернет (DSL/Fiber optic), Интернет-безопасность, Техническая поддержка, Облачное хранилище, Стриминговое ТВ и каталог фильмов.
* Условия договора: Срок действия, способ оплаты, получение электронного чека.

Во всех файлах столбец customerID содержит код клиента. Информация о договорах актуальна на 1 февраля 2020 года.

**Цель:** Создание модели, которая позволит "ТелеДому" выявлять клиентов с высоким риском оттока и предлагать им специальные условия, чтобы сохранить их лояльность.

Следует проверить несколько разных моделей. Совместно с заказчиком определили список моделей для исследования:

* Логистическая регрессия
* Случайный лес
* CatBoost

## План работы

1. Загрузка и предобработка данных:<br/>
    1.1. Загрузка данных.<br/>
    1.2. Предобработка данных.<br/>
    1.3. Объединение данных.<br/>
    1.4. Исследовательский анализ и предобработка данных объединённого датафрейма.<br/>

   <br/>
2. Обучение моделей машинного обучения: <br/>
    2.1. Разбиение на обучающую и тестовую выборки.<br/>
    2.2. Подготовка данных для RandomForestClassifier.<br/>
    2.3. Подготовка данных для LogisticRegression и CatBoostClassifier.<br/>
    2.4. Обучение RandomForestClassifier.<br/>
    2.5. Обучение LogisticRegression.<br/>
    2.6. Обучение CatBoostClassifier.<br/>
    <br/>
3. Тестирование и анализ лучшей модели: <br/>
    3.1. Тестирование и выбор лучшей модели.<br/>
    3.2. Проверка лучшей модели на адекватность.<br/>
    3.3. Вычисление важности признаков для лучшей модели.<br/>
    3.4. Матрица ошибок.<br/>
    <br/>
4. Общий вывод и рекомендации заказчику.


## 1. Загрузка и предобработка данных


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import phik

from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV 
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

RANDOM_STATE=141024
TEST_SIZE=0.25
```

### 1.1. Загрузка данных


```python
contract_new = pd.read_csv(r'~/contract_new.csv')
personal_new = pd.read_csv(r'~/personal_new.csv')
internet_new = pd.read_csv(r'~/internet_new.csv')
phone_new = pd.read_csv(r'~/phone_new.csv')
```


```python
def reading(data):
    display(data.head(15))
    data.info()
    display(data.describe())
```


```python
reading(contract_new)
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
      <th>customerID</th>
      <th>BeginDate</th>
      <th>EndDate</th>
      <th>Type</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>2020-01-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>31.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>2017-04-01</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>2071.84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>2019-10-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>226.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>2016-05-01</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1960.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>2019-09-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>353.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>2019-03-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>1150.96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1452-KIOVK</td>
      <td>2018-04-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>89.10</td>
      <td>2058.21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6713-OKOMC</td>
      <td>2019-04-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>29.75</td>
      <td>300.48</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>2017-07-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.80</td>
      <td>3573.68</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6388-TABGU</td>
      <td>2014-12-01</td>
      <td>2017-05-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>56.15</td>
      <td>1628.35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9763-GRSKD</td>
      <td>2019-01-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>49.95</td>
      <td>649.35</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7469-LKBCI</td>
      <td>2018-10-01</td>
      <td>No</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>18.95</td>
      <td>312.3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8091-TTVAX</td>
      <td>2015-04-01</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>100.35</td>
      <td>6111.31</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0280-XJGEX</td>
      <td>2015-09-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>103.70</td>
      <td>5496.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5129-JLPIS</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.50</td>
      <td>2637.5</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7043 non-null   object 
     1   BeginDate         7043 non-null   object 
     2   EndDate           7043 non-null   object 
     3   Type              7043 non-null   object 
     4   PaperlessBilling  7043 non-null   object 
     5   PaymentMethod     7043 non-null   object 
     6   MonthlyCharges    7043 non-null   float64
     7   TotalCharges      7043 non-null   object 
    dtypes: float64(1), object(7)
    memory usage: 440.3+ KB
    


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
      <th>MonthlyCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.761692</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.090047</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.250000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.350000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>89.850000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>118.750000</td>
    </tr>
  </tbody>
</table>
</div>


* Пропуски отсутствуют.
* Есть несоответствие типов данных: BeginDat и EndDate нужно будет привести к datetime; TotalCharge нужно будет привести float.


```python
reading(personal_new)
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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1452-KIOVK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6713-OKOMC</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6388-TABGU</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9763-GRSKD</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7469-LKBCI</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8091-TTVAX</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0280-XJGEX</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5129-JLPIS</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   customerID     7043 non-null   object
     1   gender         7043 non-null   object
     2   SeniorCitizen  7043 non-null   int64 
     3   Partner        7043 non-null   object
     4   Dependents     7043 non-null   object
    dtypes: int64(1), object(4)
    memory usage: 275.2+ KB
    


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
      <th>SeniorCitizen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162147</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.368612</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


* Пропусков нет.


```python
reading(internet_new)
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
      <th>customerID</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1452-KIOVK</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6713-OKOMC</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6388-TABGU</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9763-GRSKD</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8091-TTVAX</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0280-XJGEX</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5129-JLPIS</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3655-SNQYZ</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5517 entries, 0 to 5516
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   customerID        5517 non-null   object
     1   InternetService   5517 non-null   object
     2   OnlineSecurity    5517 non-null   object
     3   OnlineBackup      5517 non-null   object
     4   DeviceProtection  5517 non-null   object
     5   TechSupport       5517 non-null   object
     6   StreamingTV       5517 non-null   object
     7   StreamingMovies   5517 non-null   object
    dtypes: object(8)
    memory usage: 344.9+ KB
    


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
      <th>customerID</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
      <td>5517</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5517</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>7590-VHVEG</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>3096</td>
      <td>3498</td>
      <td>3088</td>
      <td>3095</td>
      <td>3473</td>
      <td>2810</td>
      <td>2785</td>
    </tr>
  </tbody>
</table>
</div>


* Пропусков нет.
* Возможно, нужно будет привести столбцы к bool.


```python
reading(phone_new)
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
      <th>customerID</th>
      <th>MultipleLines</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5575-GNVDE</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3668-QPYBK</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9237-HQITU</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9305-CDSKC</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1452-KIOVK</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7892-POOKP</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6388-TABGU</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9763-GRSKD</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7469-LKBCI</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8091-TTVAX</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0280-XJGEX</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5129-JLPIS</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3655-SNQYZ</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8191-XWSZG</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9959-WOFKT</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6361 entries, 0 to 6360
    Data columns (total 2 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   customerID     6361 non-null   object
     1   MultipleLines  6361 non-null   object
    dtypes: object(2)
    memory usage: 99.5+ KB
    


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
      <th>customerID</th>
      <th>MultipleLines</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6361</td>
      <td>6361</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>6361</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>5575-GNVDE</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>3390</td>
    </tr>
  </tbody>
</table>
</div>


* Пропусков нет.
* Возможно, потребуется bool.

**Вывод:** 
* На первый взгляд данные чистые.
* Требуются изменения следующих типов данных в файле contract_new: BeginDat и EndDate нужно будет привести к datetime; TotalCharge нужно будет привести float.
* Целевой признак `EndDate`, который показывает, продолжает ли пользователь использовать услуги компании. 

### 1.2. Предобработка данных

План работы:
1. Привести к змеиному регистру.
2. Изменить тип данных.
3. Проверить на пропуски и дубли.


```python
#приводим к нужному регистру
contract_new.columns = contract_new.columns.str.lower()
internet_new.columns = internet_new.columns.str.lower()
personal_new.columns = personal_new.columns.str.lower()
phone_new.columns = phone_new.columns.str.lower()
```


```python
#меняем тип данных
contract_new['totalcharges'] = pd.to_numeric(contract_new['totalcharges'], errors='coerce')
```


```python
contract_new['begindate'] = pd.to_datetime(contract_new['begindate'])
```

Значение `No` в столбце `EndDate`означает, что абонент все еще использует услуги.
Создадим новый признак, где `No` будет равен 1, а где присутствует дата окончания, будет 0.


```python
contract_new['target'] = 0
contract_new.loc[contract_new['enddate'] == 'No', 'target'] = 1
```

Для подсчета общего количества дней пользования, изменим `No` в EndDate на 1 февраля 2020 года. 


```python
contract_new['enddate'] = contract_new['enddate'].replace(['No'], ['2020-02-01'])
contract_new['enddate'] = pd.to_datetime(contract_new['enddate'])
```


```python
contract_new['totaldays'] = (contract_new['enddate'] - contract_new['begindate']).dt.days
```


```python
contract_new.head(15)
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
      <th>customerid</th>
      <th>begindate</th>
      <th>enddate</th>
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>target</th>
      <th>totaldays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>2020-01-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>31.04</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>2017-04-01</td>
      <td>2020-02-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>2071.84</td>
      <td>1</td>
      <td>1036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>2019-10-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>226.17</td>
      <td>1</td>
      <td>123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>2016-05-01</td>
      <td>2020-02-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1960.60</td>
      <td>1</td>
      <td>1371</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>2019-09-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>353.50</td>
      <td>1</td>
      <td>153</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>2019-03-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>1150.96</td>
      <td>1</td>
      <td>337</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1452-KIOVK</td>
      <td>2018-04-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>89.10</td>
      <td>2058.21</td>
      <td>1</td>
      <td>671</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6713-OKOMC</td>
      <td>2019-04-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>29.75</td>
      <td>300.48</td>
      <td>1</td>
      <td>306</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>2017-07-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.80</td>
      <td>3573.68</td>
      <td>1</td>
      <td>945</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6388-TABGU</td>
      <td>2014-12-01</td>
      <td>2017-05-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>56.15</td>
      <td>1628.35</td>
      <td>0</td>
      <td>882</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9763-GRSKD</td>
      <td>2019-01-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>49.95</td>
      <td>649.35</td>
      <td>1</td>
      <td>396</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7469-LKBCI</td>
      <td>2018-10-01</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>18.95</td>
      <td>312.30</td>
      <td>1</td>
      <td>488</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8091-TTVAX</td>
      <td>2015-04-01</td>
      <td>2020-02-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>100.35</td>
      <td>6111.31</td>
      <td>1</td>
      <td>1767</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0280-XJGEX</td>
      <td>2015-09-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>103.70</td>
      <td>5496.10</td>
      <td>1</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5129-JLPIS</td>
      <td>2018-01-01</td>
      <td>2020-02-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.50</td>
      <td>2637.50</td>
      <td>1</td>
      <td>761</td>
    </tr>
  </tbody>
</table>
</div>



Теперь можно удалить признак `enddate`, так как в нем больше нет необходимости.


```python
contract_new.drop('enddate', axis=1, inplace=True)
```


```python
def prep(data, name='DataFrame'):
    print(f'Количество дубликатов в {name}:', data.duplicated(subset=data.columns).sum())
    print('\n')
    print(f'Количество пропусков в {name}:\n', data.isna().sum())
```


```python
prep(contract_new, name='contract_new')
```

    Количество дубликатов в contract_new: 0
    
    
    Количество пропусков в contract_new:
     customerid           0
    begindate            0
    type                 0
    paperlessbilling     0
    paymentmethod        0
    monthlycharges       0
    totalcharges        11
    target               0
    totaldays            0
    dtype: int64
    

После изменения типа данных появились пропуски.


```python
contract_new[contract_new['totalcharges'].isna()]
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
      <th>customerid</th>
      <th>begindate</th>
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>target</th>
      <th>totaldays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>488</th>
      <td>4472-LVYGI</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>52.55</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>753</th>
      <td>3115-CZMZD</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.25</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>936</th>
      <td>5709-LVOEQ</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>80.85</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1082</th>
      <td>4367-NUYAO</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.75</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>1371-DWPAZ</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>56.05</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>7644-OMVMY</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>19.85</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3826</th>
      <td>3213-VVOLG</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>25.35</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4380</th>
      <td>2520-SGTTA</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>20.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5218</th>
      <td>2923-ARZLG</td>
      <td>2020-02-01</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>19.70</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6670</th>
      <td>4075-WKNIU</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>73.35</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6754</th>
      <td>2775-SEFEE</td>
      <td>2020-02-01</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>61.90</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



 totaldays равен `Nan`. Скорее всего, это новые пользователи. Заполним тогда 0.


```python
contract_new['totalcharges'] = contract_new['totalcharges'].replace(np.nan, 0)
```


```python
prep(internet_new, name='internet_new')
```

    Количество дубликатов в internet_new: 0
    
    
    Количество пропусков в internet_new:
     customerid          0
    internetservice     0
    onlinesecurity      0
    onlinebackup        0
    deviceprotection    0
    techsupport         0
    streamingtv         0
    streamingmovies     0
    dtype: int64
    


```python
prep(personal_new, name='personal_new')
```

    Количество дубликатов в personal_new: 0
    
    
    Количество пропусков в personal_new:
     customerid       0
    gender           0
    seniorcitizen    0
    partner          0
    dependents       0
    dtype: int64
    


```python
prep(phone_new, name='phone_new')
```

    Количество дубликатов в phone_new: 0
    
    
    Количество пропусков в phone_new:
     customerid       0
    multiplelines    0
    dtype: int64
    

Таким образом, мы изменили тип данных, где это нужно, и добавили новые признаки: количество дней и целевой признак, который показывает, продолжает ли абонент еще пользоваться услугами.

### 1.3. Объединение данных

Объединим датасеты по `customerid`


```python
df = (
    contract_new
.merge(personal_new, how='left', on='customerid')
.merge(internet_new, how='left', on='customerid')
.merge(phone_new, how='left', on='customerid')
     )
```


```python
#для проверки уникальности customerid
df['customerid'].is_unique
```




    True




```python
reading(df)
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
      <th>customerid</th>
      <th>begindate</th>
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>target</th>
      <th>totaldays</th>
      <th>gender</th>
      <th>...</th>
      <th>partner</th>
      <th>dependents</th>
      <th>internetservice</th>
      <th>onlinesecurity</th>
      <th>onlinebackup</th>
      <th>deviceprotection</th>
      <th>techsupport</th>
      <th>streamingtv</th>
      <th>streamingmovies</th>
      <th>multiplelines</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>2020-01-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>31.04</td>
      <td>1</td>
      <td>31</td>
      <td>Female</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>2017-04-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>2071.84</td>
      <td>1</td>
      <td>1036</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>2019-10-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>226.17</td>
      <td>1</td>
      <td>123</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>2016-05-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1960.60</td>
      <td>1</td>
      <td>1371</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>2019-09-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>353.50</td>
      <td>1</td>
      <td>153</td>
      <td>Female</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>2019-03-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>1150.96</td>
      <td>1</td>
      <td>337</td>
      <td>Female</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1452-KIOVK</td>
      <td>2018-04-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>89.10</td>
      <td>2058.21</td>
      <td>1</td>
      <td>671</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6713-OKOMC</td>
      <td>2019-04-01</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>29.75</td>
      <td>300.48</td>
      <td>1</td>
      <td>306</td>
      <td>Female</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>2017-07-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.80</td>
      <td>3573.68</td>
      <td>1</td>
      <td>945</td>
      <td>Female</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6388-TABGU</td>
      <td>2014-12-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>56.15</td>
      <td>1628.35</td>
      <td>0</td>
      <td>882</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9763-GRSKD</td>
      <td>2019-01-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>49.95</td>
      <td>649.35</td>
      <td>1</td>
      <td>396</td>
      <td>Male</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7469-LKBCI</td>
      <td>2018-10-01</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>18.95</td>
      <td>312.30</td>
      <td>1</td>
      <td>488</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8091-TTVAX</td>
      <td>2015-04-01</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>100.35</td>
      <td>6111.31</td>
      <td>1</td>
      <td>1767</td>
      <td>Male</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0280-XJGEX</td>
      <td>2015-09-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>103.70</td>
      <td>5496.10</td>
      <td>1</td>
      <td>1614</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5129-JLPIS</td>
      <td>2018-01-01</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.50</td>
      <td>2637.50</td>
      <td>1</td>
      <td>761</td>
      <td>Male</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 21 columns</p>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   customerid        7043 non-null   object        
     1   begindate         7043 non-null   datetime64[ns]
     2   type              7043 non-null   object        
     3   paperlessbilling  7043 non-null   object        
     4   paymentmethod     7043 non-null   object        
     5   monthlycharges    7043 non-null   float64       
     6   totalcharges      7043 non-null   float64       
     7   target            7043 non-null   int64         
     8   totaldays         7043 non-null   int64         
     9   gender            7043 non-null   object        
     10  seniorcitizen     7043 non-null   int64         
     11  partner           7043 non-null   object        
     12  dependents        7043 non-null   object        
     13  internetservice   5517 non-null   object        
     14  onlinesecurity    5517 non-null   object        
     15  onlinebackup      5517 non-null   object        
     16  deviceprotection  5517 non-null   object        
     17  techsupport       5517 non-null   object        
     18  streamingtv       5517 non-null   object        
     19  streamingmovies   5517 non-null   object        
     20  multiplelines     6361 non-null   object        
    dtypes: datetime64[ns](1), float64(2), int64(3), object(15)
    memory usage: 1.2+ MB
    


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
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>target</th>
      <th>totaldays</th>
      <th>seniorcitizen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.761692</td>
      <td>2115.312885</td>
      <td>0.843675</td>
      <td>898.555729</td>
      <td>0.162147</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.090047</td>
      <td>2112.742814</td>
      <td>0.363189</td>
      <td>683.130510</td>
      <td>0.368612</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.500000</td>
      <td>436.750000</td>
      <td>1.000000</td>
      <td>276.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.350000</td>
      <td>1343.350000</td>
      <td>1.000000</td>
      <td>761.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>89.850000</td>
      <td>3236.690000</td>
      <td>1.000000</td>
      <td>1461.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>118.750000</td>
      <td>9221.380000</td>
      <td>1.000000</td>
      <td>2314.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
prep(df)
```

    Количество дубликатов в DataFrame: 0
    
    
    Количество пропусков в DataFrame:
     customerid             0
    begindate              0
    type                   0
    paperlessbilling       0
    paymentmethod          0
    monthlycharges         0
    totalcharges           0
    target                 0
    totaldays              0
    gender                 0
    seniorcitizen          0
    partner                0
    dependents             0
    internetservice     1526
    onlinesecurity      1526
    onlinebackup        1526
    deviceprotection    1526
    techsupport         1526
    streamingtv         1526
    streamingmovies     1526
    multiplelines        682
    dtype: int64
    

Появились пропуски в столбцах `internetservice`, `onlinesecurity`,`onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`, `multiplelines`. Возможно, эти столбцы не были заполнены по техническим причинам (например, данные не сохранились). Заполнить `No_service`


```python
columns = ['internetservice','onlinesecurity','onlinebackup','deviceprotection', 
        'techsupport','streamingtv', 'streamingmovies','multiplelines']

for column in columns:
    df[column] = df[column].fillna('No_service')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   customerid        7043 non-null   object        
     1   begindate         7043 non-null   datetime64[ns]
     2   type              7043 non-null   object        
     3   paperlessbilling  7043 non-null   object        
     4   paymentmethod     7043 non-null   object        
     5   monthlycharges    7043 non-null   float64       
     6   totalcharges      7043 non-null   float64       
     7   target            7043 non-null   int64         
     8   totaldays         7043 non-null   int64         
     9   gender            7043 non-null   object        
     10  seniorcitizen     7043 non-null   int64         
     11  partner           7043 non-null   object        
     12  dependents        7043 non-null   object        
     13  internetservice   7043 non-null   object        
     14  onlinesecurity    7043 non-null   object        
     15  onlinebackup      7043 non-null   object        
     16  deviceprotection  7043 non-null   object        
     17  techsupport       7043 non-null   object        
     18  streamingtv       7043 non-null   object        
     19  streamingmovies   7043 non-null   object        
     20  multiplelines     7043 non-null   object        
    dtypes: datetime64[ns](1), float64(2), int64(3), object(15)
    memory usage: 1.2+ MB
    

Мы объединили датасеты, заполнив пропуски `No_service`.

### 1.4.  Исследовательский анализ и предобработка данных объединённого датафрейма

Построим графики для категориальных данных.


```python
def cat_plot(df, column):
    display(df[column].unique())
    plt.figure(figsize=(9,7))
    sns.countplot(y=column, data=df)
    plt.title(f'Рапределение по {column}', fontsize=16)
    plt.xlabel('Количество', fontsize=14)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   customerid        7043 non-null   object        
     1   begindate         7043 non-null   datetime64[ns]
     2   type              7043 non-null   object        
     3   paperlessbilling  7043 non-null   object        
     4   paymentmethod     7043 non-null   object        
     5   monthlycharges    7043 non-null   float64       
     6   totalcharges      7043 non-null   float64       
     7   target            7043 non-null   int64         
     8   totaldays         7043 non-null   int64         
     9   gender            7043 non-null   object        
     10  seniorcitizen     7043 non-null   int64         
     11  partner           7043 non-null   object        
     12  dependents        7043 non-null   object        
     13  internetservice   7043 non-null   object        
     14  onlinesecurity    7043 non-null   object        
     15  onlinebackup      7043 non-null   object        
     16  deviceprotection  7043 non-null   object        
     17  techsupport       7043 non-null   object        
     18  streamingtv       7043 non-null   object        
     19  streamingmovies   7043 non-null   object        
     20  multiplelines     7043 non-null   object        
    dtypes: datetime64[ns](1), float64(2), int64(3), object(15)
    memory usage: 1.2+ MB
    


```python
categorial_columns = ['type', 'paperlessbilling','paymentmethod', 'gender', 'partner', 'dependents', 'internetservice', 
                      'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 
                     'multiplelines', 'seniorcitizen', 'target']
num_columns = ['monthlycharges', 'totalcharges', 'totaldays']
```


```python
cols = 2
rows = int(np.ceil(len(categorial_columns) / cols))


plt.figure(figsize=(20, 35))

for idx, column in enumerate(categorial_columns):
    sns.set()
    ax = plt.subplot(rows, cols, idx + 1)
    
    df[column].value_counts().plot(
        kind='bar', 
        title=column, 
        ax=ax,
        rot=0,
        color='thistle'
    )
    
    ax.set_xlabel('Категория') 
    ax.set_ylabel('Количество')
        
plt.tight_layout()
plt.show()
```


    
![png](output_54_0.png)
    


* Тип оплаты: лидирует month-to-month.
* Электронный расчётный лист чаще с согласием.
* Тип платежа чаще всего электронный чек.
* Столбцы `gender` и `partner`  распределены равномерны. 
* У большинства нет детей.
* Популярный тип подключения Fiber optic.
* Пенсионеры редко пользуются услугами.
* Остальные услуги `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`, `multiplelines` не пользуются популярностью. Также некоторая часть неизвестна.
* Большая часть абонентов еще пользуются услугами.

Посмотрим теперь на количественные данные.


```python
for idx, column in enumerate(num_columns):
    
    sns.set()
    f, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].set_title(f'Гистограмма для {column}', fontsize=16)
    axes[0].set_ylabel('Количество', fontsize=14)
    sns.histplot(df, bins=20, kde=True, ax=axes[0], x=column)
    
    axes[1].set_title(f'График ящик с усами для {column}', fontsize=16)
    sns.boxplot(data=df, ax=axes[1], y=column)
    axes[1].set_ylabel(column, fontsize=14)
    
    plt.show()
```


    
![png](output_57_0.png)
    



    
![png](output_57_1.png)
    



    
![png](output_57_2.png)
    


* В MonthlyCharges заметен пик около 20, который соответствует минимальному значению. 
* Значения в столбце TotalCharges уменьшаются, в то время как в столбце MonthlyCharges наблюдается пик около 80, что указывает на то, что пользователи, платящие высокую сумму за тариф, не задерживаются надолго. 
* В столбце TotalDays можно выделить два пика: один — среди новых клиентов, а другой — среди "пользователей-старичков".
* Аномальных значений не наблюдается.

Посмотрим на распределение признаков в разрезе оттока: сначала на количественные, потом на категориальные данные.


```python
def outflow_num(column, bins):
    plt.figure(figsize=(10, 6))
    
    ax = sns.histplot(
        data=df,
        x=column,
        hue='target',
        stat='density',
        common_norm=False,
        bins=bins,
        hue_order=df['target'].unique() 
    )

    plt.title('Распределение значений с учетом оттока')
    plt.xlabel(column)
    plt.ylabel('Плотность')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(['Отток', 'Не отток'])

    plt.show()
```


```python
outflow_num('monthlycharges', 80)
```


    
![png](output_61_0.png)
    


Клиенты, которые использовали более дорогие тарифы, чаще отказывались от услуг.


```python
outflow_num('totalcharges', 120)
```


    
![png](output_63_0.png)
    


Пользователи, в сумме которые платили меньше, чаще остаются пользователями.


```python
outflow_num('totaldays', 80)
```


    
![png](output_65_0.png)
    


Здесь наблюдаем два пика пользователей: либо они только начали пользоваться, либо уже являются "старичками". Как раз в середине периода наибольшее количество клиентов перестают пользоваться услугами. Возможно, именно здесь нужно что-то предпринимать.


```python
def outflow_cat(column):
    
    average_target = df['target'].mean()  

    plt.figure(figsize=(12, 6))
    sns.barplot(x=column, y='target', data=df, ci=None)  
    plt.axhline(average_target, color='red', linestyle='--', label='Средний уровень оттока')
    plt.title('Уровень оттока по категориям')
    plt.xlabel(column)
    plt.ylabel('Уровень оттока')
    plt.legend()
    
    plt.show()
```


```python
for idx, column in enumerate(categorial_columns):
    outflow_cat(column)
```


    
![png](output_68_0.png)
    



    
![png](output_68_1.png)
    



    
![png](output_68_2.png)
    



    
![png](output_68_3.png)
    



    
![png](output_68_4.png)
    



    
![png](output_68_5.png)
    



    
![png](output_68_6.png)
    



    
![png](output_68_7.png)
    



    
![png](output_68_8.png)
    



    
![png](output_68_9.png)
    



    
![png](output_68_10.png)
    



    
![png](output_68_11.png)
    



    
![png](output_68_12.png)
    



    
![png](output_68_13.png)
    



    
![png](output_68_14.png)
    



    
![png](output_68_15.png)
    


В зоне риска находятся пользователи, которые:
* платят ежемесячно;
* не имеют партнера;
* которые не подключали какие-либо дополнительные услуги;
* не пенсионеры (это может быть связано с тем, что зачастую именно молодые люди пробуют и/или сравнивают разные пакеты/услуги, в то время как пожилые люди, во-первых, часто меньше разбираются в этом, во-вторых, им зачастую это не нужно - что подключили, то подключили, главное, чтобы хорошо работало).

На основе полученной информации можно составить портрет пользователя, который вскоре может покинуть компанию:
* это молодой человек (пол не важен); 
* возможно, один, не в браке; 
* без дополнительных услуг;
* пользуется услугами этой компании больше года;
* его может не устраивать общая плата за услуги: допустим, он нашел более выгодный пакает у конкурентов  и считает, что в этой компании он переплачивает.


То есть в зоне риска находятся клиенты, которые уже прошли пероид адаптации и уже неплохо разбираются в этом рынке.

Посмотрим на матрицу корреляции, предварительно удалив даты из df.


```python
df= df.drop(['begindate'], axis=1)
```


```python
matrix_corr = df.drop('customerid', axis=1).phik_matrix(interval_cols=['monthlycharges', 'totalcharges','totaldays'])
plt.figure(figsize=(15, 15)) 
sns.heatmap(matrix_corr, annot=True, fmt=".2f", cmap='magma', annot_kws={"size": 10})
plt.title('Матрица корреляции')
plt.show()
```


    
![png](output_73_0.png)
    


Удалим из-за высокой корреляции следующие столбцы: internetservice, onlinesecurity, deviceprotection, streamingtv, streamingmovies, techsupport. Удалим gender, так как он почти не коррелирует с target.


```python
df1 = df.copy()
```


```python
df1.columns
```




    Index(['customerid', 'type', 'paperlessbilling', 'paymentmethod',
           'monthlycharges', 'totalcharges', 'target', 'totaldays', 'gender',
           'seniorcitizen', 'partner', 'dependents', 'internetservice',
           'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
           'streamingtv', 'streamingmovies', 'multiplelines'],
          dtype='object')




```python
df1= df1.drop(['internetservice', 'onlinesecurity', 'deviceprotection', 'streamingtv', 
               'streamingmovies', 'techsupport', 'gender', 'onlinebackup', 'totaldays'], axis=1)
```


```python
matrix_corr = df1.drop('customerid', axis=1).phik_matrix(interval_cols=['monthlycharges','totalcharges'])
plt.figure(figsize=(15, 15)) 
sns.heatmap(matrix_corr, annot=True, fmt=".2f", cmap='magma', annot_kws={"size": 10})
plt.title('Матрица корреляции')
plt.show()
```


    
![png](output_78_0.png)
    


Теперь высокой корреляции не наблюдается. 

## 2 Обучение моделей машинного обучения

### 2.1. Разбиение на обучающую и тестовую выборки

Установим `customerid` в качестве индекса


```python
df1 = df1.set_index('customerid')
```


```python
df1.head(15)
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
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>monthlycharges</th>
      <th>totalcharges</th>
      <th>target</th>
      <th>seniorcitizen</th>
      <th>partner</th>
      <th>dependents</th>
      <th>multiplelines</th>
    </tr>
    <tr>
      <th>customerid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7590-VHVEG</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>31.04</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>No_service</td>
    </tr>
    <tr>
      <th>5575-GNVDE</th>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>2071.84</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3668-QPYBK</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>226.17</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7795-CFOCW</th>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1960.60</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No_service</td>
    </tr>
    <tr>
      <th>9237-HQITU</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>353.50</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9305-CDSKC</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>1150.96</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1452-KIOVK</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>89.10</td>
      <td>2058.21</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6713-OKOMC</th>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>29.75</td>
      <td>300.48</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No_service</td>
    </tr>
    <tr>
      <th>7892-POOKP</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.80</td>
      <td>3573.68</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6388-TABGU</th>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>56.15</td>
      <td>1628.35</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9763-GRSKD</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>49.95</td>
      <td>649.35</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7469-LKBCI</th>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>18.95</td>
      <td>312.30</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8091-TTVAX</th>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>100.35</td>
      <td>6111.31</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>0280-XJGEX</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>103.70</td>
      <td>5496.10</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5129-JLPIS</th>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.50</td>
      <td>2637.50</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Разделим на train и test выборки и преобразуем категориальные данные с помощь get_dummies, масштабируем числовые с помощью StandarScaler.


```python
X = df1.drop(columns=['target'])
y = df1['target']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=TEST_SIZE, 
                                                    random_state=RANDOM_STATE
                                                   )
```


```python
cat=['type', 'paperlessbilling', 'paymentmethod',
       'seniorcitizen', 'partner', 'dependents', 'multiplelines']
num=['totalcharges', 'monthlycharges']
```


```python
df1.columns
```




    Index(['type', 'paperlessbilling', 'paymentmethod', 'monthlycharges',
           'totalcharges', 'target', 'seniorcitizen', 'partner', 'dependents',
           'multiplelines'],
          dtype='object')



### 2.2. Подготовка данных для RandomForestClassifier


```python
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_ord = encoder.fit_transform(X_train[cat])
X_test_ord = encoder.transform(X_test[cat])
```


```python
X_train_ord = pd.DataFrame(X_train_ord, columns=cat)
X_test_ord = pd.DataFrame(X_test_ord, columns=cat)
X_test_ord
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
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>seniorcitizen</th>
      <th>partner</th>
      <th>dependents</th>
      <th>multiplelines</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
    </tr>
    <tr>
      <th>1756</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1757</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1758</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1760</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1761 rows × 7 columns</p>
</div>



Масштабируем количественные признаки с помощью StandardScaler.


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num])
X_test_scaled = scaler.transform(X_test[num])
```


```python
X_train_scaled = pd.DataFrame(X_train_scaled, columns=num)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=num)
```


```python
X_train_forest = pd.concat((X_train_ord, X_train_scaled), axis=1)
X_test_forest = pd.concat((X_test_ord, X_test_scaled), axis=1)
```


```python
X_train_forest.head(15)
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
      <th>type</th>
      <th>paperlessbilling</th>
      <th>paymentmethod</th>
      <th>seniorcitizen</th>
      <th>partner</th>
      <th>dependents</th>
      <th>multiplelines</th>
      <th>totalcharges</th>
      <th>monthlycharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-0.307391</td>
      <td>0.320908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.737517</td>
      <td>1.474037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.043361</td>
      <td>1.194491</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.526622</td>
      <td>-1.133400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-0.472235</td>
      <td>0.342539</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.664964</td>
      <td>0.104592</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.907262</td>
      <td>-0.414566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>-0.476156</td>
      <td>0.324236</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-0.731959</td>
      <td>-0.338024</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.031364</td>
      <td>0.124559</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.225101</td>
      <td>0.991487</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.920377</td>
      <td>0.863361</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-0.041633</td>
      <td>0.472329</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.081260</td>
      <td>-0.973659</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.025659</td>
      <td>0.638725</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3. Подготовка данных для LogisticRegression и CatBoostClassifier


```python
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
```


```python
X_train_ohe = encoder.fit_transform(X_train[cat])
X_test_ohe = encoder.transform(X_test[cat])

encoder_col_names = encoder.get_feature_names_out()
encoder_col_names
```




    array(['type_One year', 'type_Two year', 'paperlessbilling_Yes',
           'paymentmethod_Credit card (automatic)',
           'paymentmethod_Electronic check', 'paymentmethod_Mailed check',
           'seniorcitizen_1', 'partner_Yes', 'dependents_Yes',
           'multiplelines_No_service', 'multiplelines_Yes'], dtype=object)




```python
X_train_ohe = pd.DataFrame(X_train_ohe, columns=encoder_col_names)
X_test_ohe = pd.DataFrame(X_test_ohe, columns=encoder_col_names)
```


```python
X_train_log = pd.concat((X_train_ohe, X_train_scaled), axis=1)
X_test_log = pd.concat((X_test_ohe, X_test_scaled), axis=1)
```


```python
X_train_log.head(15)
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
      <th>type_One year</th>
      <th>type_Two year</th>
      <th>paperlessbilling_Yes</th>
      <th>paymentmethod_Credit card (automatic)</th>
      <th>paymentmethod_Electronic check</th>
      <th>paymentmethod_Mailed check</th>
      <th>seniorcitizen_1</th>
      <th>partner_Yes</th>
      <th>dependents_Yes</th>
      <th>multiplelines_No_service</th>
      <th>multiplelines_Yes</th>
      <th>totalcharges</th>
      <th>monthlycharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.307391</td>
      <td>0.320908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.737517</td>
      <td>1.474037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.043361</td>
      <td>1.194491</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.526622</td>
      <td>-1.133400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.472235</td>
      <td>0.342539</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.664964</td>
      <td>0.104592</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.907262</td>
      <td>-0.414566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.476156</td>
      <td>0.324236</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.731959</td>
      <td>-0.338024</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.031364</td>
      <td>0.124559</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.225101</td>
      <td>0.991487</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.920377</td>
      <td>0.863361</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.041633</td>
      <td>0.472329</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.081260</td>
      <td>-0.973659</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.025659</td>
      <td>0.638725</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.shape[0] == y_train.shape[0]
```




    True




```python
X_train_forest.shape[0] == y_train.shape[0]
```




    True




```python
X_train_log.shape[0] == y_train.shape[0]
```




    True



Мы установили customerid в качестве индекса, разбили нашу выборку на test и train, подготовив данные.

Обучим три модели на тренировачной выборке: RandomForestClassifier, LogisticRegression и CatBoostClassifier.

### 2.4.  Обучение RandomForestClassifier


```python
model = RandomForestClassifier(random_state=RANDOM_STATE)

param_grid = {
  'n_estimators': [50, 100, 200, 500],
  'max_depth': [5, 10, 20],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}
randomized_search = RandomizedSearchCV(
    model,
    param_grid, 
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
randomized_search.fit(X_train_forest, y_train)

print('Лучшая модель и её параметры:\n\n', randomized_search.best_params_)
print ('Метрика лучшей модели на кросс-валидации:', round(randomized_search.best_score_, 2))
```

    Лучшая модель и её параметры:
    
     {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 10}
    Метрика лучшей модели на кросс-валидации: 0.8
    

### 2.5.  Обучение LogisticRegression


```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

model = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='roc_auc')

grid_search.fit(X_train_log, y_train)

print('Лучшая модель и её параметры:\n\n', grid_search.best_params_)
print ('Метрика лучшей модели на кросс-валидации:', round(grid_search.best_score_, 2))
```

    Лучшая модель и её параметры:
    
     {'C': 1}
    Метрика лучшей модели на кросс-валидации: 0.75
    

### 2.6.  Обучение CatBoostClassifier


```python
param_grid = {
    'learning_rate': [1, 0.5],
    'iterations': [50, 150],
    'l2_leaf_reg': [2, 9]
}

model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)

cat_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='roc_auc')

cat_search.fit(X_train_log, y_train)

print('Лучшая модель и её параметры:\n\n', cat_search.best_params_)
print ('Метрика лучшей модели на кросс-валидации:', round(cat_search.best_score_, 2))
```

    Лучшая модель и её параметры:
    
     {'iterations': 50, 'l2_leaf_reg': 9, 'learning_rate': 0.5}
    Метрика лучшей модели на кросс-валидации: 0.81
    

**Вывод:** Лучшей моделью оказалась CatBoostClassifier с параметрами {'iterations': 50, 'l2_leaf_reg': 9, 'learning_rate': 0.5}. Протестируем ее на тестовой выборке.

## 3 Тестирование и анализ лучшей модели

### 3.1. Тестирование и выбор лучшей модели


```python
probabilities = cat_search.predict_proba(X_test_log)[:,1]
print('Площадь ROC-кривой:', round(roc_auc_score(y_test, probabilities),2))
```

    Площадь ROC-кривой: 0.84
    

Посчитаем также accuracy для более легкой интерпретируемости модели.


```python
predicted_classes = np.where(probabilities >= 0.5, 1, 0)
accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy:", accuracy)
```

    Accuracy: 0.8659852356615559
    

Значение Accuracy, или точность, 0.866  указывает на то, что модель имеет хорошую производительность, так как более 86% предсказаний верны.

### 3.2. Проверка лучшей модели на адекватность

Проверим лучшую модель на адекватность, сравнив качество её предсказаний с качеством модели DummyClassifier.


```python
dummy_clf = DummyClassifier(strategy='most_frequent')

dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)

print('Площадь ROC-кривой:', roc_auc_score(y_test, y_pred))
```

    Площадь ROC-кривой: 0.5
    

Видим, что модель CatBoostClassifier прошла проверку на адекватность: она хорошо различает классы независимо от их распределения.

### 3.3. Вычисление важности признаков для лучшей модели

Вычислим важность признаков для лучшей модели.


```python
importance = pd.DataFrame(cat_search.best_estimator_.feature_importances_, index = X_test_log.columns, columns=['importances'])
importance = importance.sort_values(by='importances', ascending=False)
display(importance)

importance.plot(kind='bar', figsize=(10, 5), title='Важность факторов')
plt.xlabel('Факторы') 
plt.ylabel('Важность')
plt.show()
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
      <th>importances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>totalcharges</th>
      <td>36.389912</td>
    </tr>
    <tr>
      <th>monthlycharges</th>
      <td>17.497515</td>
    </tr>
    <tr>
      <th>type_Two year</th>
      <td>13.627689</td>
    </tr>
    <tr>
      <th>type_One year</th>
      <td>7.797946</td>
    </tr>
    <tr>
      <th>partner_Yes</th>
      <td>5.916557</td>
    </tr>
    <tr>
      <th>multiplelines_Yes</th>
      <td>4.604226</td>
    </tr>
    <tr>
      <th>multiplelines_No_service</th>
      <td>3.547756</td>
    </tr>
    <tr>
      <th>paymentmethod_Credit card (automatic)</th>
      <td>2.960150</td>
    </tr>
    <tr>
      <th>dependents_Yes</th>
      <td>2.404827</td>
    </tr>
    <tr>
      <th>paymentmethod_Mailed check</th>
      <td>2.105347</td>
    </tr>
    <tr>
      <th>paperlessbilling_Yes</th>
      <td>1.349709</td>
    </tr>
    <tr>
      <th>seniorcitizen_1</th>
      <td>1.273092</td>
    </tr>
    <tr>
      <th>paymentmethod_Electronic check</th>
      <td>0.525277</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_128_1.png)
    


* Наиболее важный фактор - это общая сумма плат.
* Далее идут такие факторы, как ежемесячная плата и тип.

### 3.4. Матрица ошибок

Проанализируем матрицу ошибок.


```python
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
      xticklabels=np.unique(y_test),
      yticklabels=np.unique(y_test))
plt.xlabel("Предсказанные значения")
plt.ylabel("Истинные значения")
plt.title("Матрица ошибок")

plt.show()
```


    
![png](output_132_0.png)
    


Матрица показывает следующее:
* TP (Верно положительные): 1444 пользователя, которые еще пользуются услугами и были предсказаны как действительные пользователи.
* TN (Верно отрицательные): 81 пользователь, которые уже не пользуются услугами и были предсказаны как недействительные пользователи.
* FP (Ложно положительные): 200 пользователей, которые уже не пользуются услугами, но были предсказаны как действительные пользователи.
* FN (Ложно отрицательные): 36 пользователей, которые еще пользуются услугами, но модель предсказала, что они недействительные пользователи.

## 4 Общий вывод и рекомендации заказчику

В этой работе было проведено исследование, которое поможет заказчику предсказать отток клиентов.
* Были добавлены дополнительные признакми: TotalDays и target, который показывал, пользуется ли еще абонент услугами.
* Проанализировали все признаки, посмотрели на корреляцию, удалили те, которые имели недопустимые значения.
* Были построены три модели, лучшей оказалась CatBoostClassifier с параметрами {'iterations': 50, 'l2_leaf_reg': 9, 'learning_rate': 0.5}. 
* На тестовой выборке ее площадь ROC-кривой составила 0.84.
* Acсuracy на тестовой выборке составила 0.87. То есть точность нашей модели составляет 87%.
* Решающим фактором является общая сумма плат пользователя в этой компании.
* На следующем месте - ежемесячная плата. На это тоже стоит обратить внимание.

На основе полученной информации можно составить портрет пользователя, который вскоре может покинуть компанию:
* это молодой человек (пол не важен); 
* возможно, один, не в браке; 
* без дополнительных услуг;
* пользуется услугами этой компании больше года;
* его может не устраивать общая плата за услуги: допустим, он нашел более выгодный пакает у конкурентов  и считает, что в этой компании он переплачивает.


То есть в зоне риска находятся клиенты, которые уже прошли пероид адаптации и уже неплохо разбираются в этом рынке.
Можно попробовать предлагать специальные условия пользователям, которы уже пользуются услугами определенный период или предложить им тариф за меньшую плату.
