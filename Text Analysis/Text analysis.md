# Анализ текстов

Наш заказчик интернет-магазин «Викишоп» запускает новый сервис: теперь пользователи могут редактировать и дополнять описания товаров. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 

Следует обучить модель классифицировать комментарии на позитивные и негативные. Заказчик предоставил набор данных с разметкой о токсичности правок.

Заказчик требует, чтобы метрика качества *F1* была не меньше 0.75.. 


## Описание данных

Данные находятся в файле `toxic_comments.csv`:
* `text`  содержит текст комментария.
* `toxic` — целевой признак.

## План работы

1. Загрузка и подготовка данных:<br/>
    1.1. Загрузка данных.<br/>
    1.2. Исследование данных.<br/>
    1.3. Исследование на дисбаланс классов.<br/>
    1.4. Нормализация и лемматизация текстов.<br/>
    1.5. Разбиение на обучающую и тестовую выборки.<br/>


2. Обучение моделей: <br/>
    2.1. Логистическая регрессия и Дерево решений.<br/>
    2.2. Выбор лучшей модели.<br/>

3. Выводы.

### 1 Загрузка и подготовка данных


```python
import re
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import nltk
from nltk import pos_tag
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
```

### 1.1. Загрузка данных


```python
data = pd.read_csv(r'~/toxic_comments.csv')
data.head(15)
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
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Explanation\nWhy the edits made under my usern...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>D'aww! He matches this background colour I'm s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Hey man, I'm really not trying to edit war. It...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>"\nMore\nI can't make any real suggestions on ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>You, sir, are my hero. Any chance you remember...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>"\n\nCongratulations from me as well, use the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Your vandalism to the Matt Shirvington article...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Sorry if the word 'nonsense' was offensive to ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>alignment on this subject and which are contra...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>"\nFair use rationale for Image:Wonju.jpg\n\nT...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>bbq \n\nbe a man and lets discuss it-maybe ove...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Hey... what is it..\n@ | talk .\nWhat is it......</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Before you start throwing accusations and warn...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Oh, and the girl above started her arguments w...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2. Исследование данных


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159292 entries, 0 to 159291
    Data columns (total 3 columns):
     #   Column      Non-Null Count   Dtype 
    ---  ------      --------------   ----- 
     0   Unnamed: 0  159292 non-null  int64 
     1   text        159292 non-null  object
     2   toxic       159292 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 3.6+ MB
    

Столбец Unnamed: 0 повторяет нумерацию. Можем его удалить.


```python
data = data.drop(['Unnamed: 0'], axis=1)
```


```python
data.head(15)
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
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Explanation\nWhy the edits made under my usern...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D'aww! He matches this background colour I'm s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey man, I'm really not trying to edit war. It...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"\nMore\nI can't make any real suggestions on ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You, sir, are my hero. Any chance you remember...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>"\n\nCongratulations from me as well, use the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Your vandalism to the Matt Shirvington article...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sorry if the word 'nonsense' was offensive to ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>alignment on this subject and which are contra...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>"\nFair use rationale for Image:Wonju.jpg\n\nT...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bbq \n\nbe a man and lets discuss it-maybe ove...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hey... what is it..\n@ | talk .\nWhat is it......</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Before you start throwing accusations and warn...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Oh, and the girl above started her arguments w...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Проверим на пропуски и дубликаты


```python
print('Количество пропусков:\n', data.isna().sum())
```

    Количество пропусков:
     text     0
    toxic    0
    dtype: int64
    


```python
print('Количество дубликатов:', data.duplicated().sum())
```

    Количество дубликатов: 0
    

Данные хорошие, без пропусков и дублей. Был один лишний столбец.

### 1.3. Исследование на дисбаланс классов

Построим гистограмму для наглядности.


```python
toxic_counts = data['toxic'].value_counts()
plt.bar(toxic_counts.index, toxic_counts.values) 

plt.xlabel('Токсичность')
plt.ylabel('Количество')
plt.title('Столбчатая диаграмма Токсичности')

plt.show()
```


    
![png](output_16_0.png)
    



```python
toxic_len = data[data['toxic'] == 1].shape[0]
toxic_non_len = data[data['toxic'] != 1].shape[0]

print('toxic: {:d} rows, {:.2%}'.format(toxic_len, toxic_len / data.shape[0]))
print('non_toxic: {:d} rows, {:.2%}'.format(toxic_non_len, toxic_non_len / data.shape[0]))
```

    toxic: 16186 rows, 10.16%
    non_toxic: 143106 rows, 89.84%
    

Большинство отзывов положительные. Потребуется делать балансировку классов. Учтём это в параметрах моделей.

### 1.4. Нормализация и лемматизация текстов


```python
stop_words = set(stopwords.words('english'))
```


```python
wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'
```


```python
def preprocessing(text):
    
    text = re.sub(r'[^a-zA-Z ]', ' ', text) 
    tokens = [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
              for word, tag in pos_tag(word_tokenize(text))] 
    text = ' '.join(tokens) 
    return text
```


```python
data['lemm_text'] = data['text'].progress_apply(preprocessing)
```


      0%|          | 0/159292 [00:00<?, ?it/s]



```python
data.head()
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
      <th>text</th>
      <th>toxic</th>
      <th>lemm_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Explanation\nWhy the edits made under my usern...</td>
      <td>0</td>
      <td>explanation why the edits make under my userna...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D'aww! He matches this background colour I'm s...</td>
      <td>0</td>
      <td>d aww he match this background colour i m seem...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey man, I'm really not trying to edit war. It...</td>
      <td>0</td>
      <td>hey man i m really not try to edit war it s ju...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"\nMore\nI can't make any real suggestions on ...</td>
      <td>0</td>
      <td>more i can t make any real suggestion on impro...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You, sir, are my hero. Any chance you remember...</td>
      <td>0</td>
      <td>you sir be my hero any chance you remember wha...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = data.copy()
```


```python
df = df.drop(['text'], axis = 1)
```


```python
df.head()
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
      <th>toxic</th>
      <th>lemm_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>explanation why the edits make under my userna...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>d aww he match this background colour i m seem...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>hey man i m really not try to edit war it s ju...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>more i can t make any real suggestion on impro...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>you sir be my hero any chance you remember wha...</td>
    </tr>
  </tbody>
</table>
</div>



Тексты готовы для превращения в признаки.

### 1.5. Разбиение на тестовую и тренировочную выборки


```python
y = df['toxic']
X = df.drop(['toxic'], axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, stratify=y)

print('Train data shape:', X_train.shape, y_train.shape)
print('Test data shape:', X_test.shape, y_test.shape)
```

    Train data shape: (143362, 2) (143362,)
    Test data shape: (15930, 2) (15930,)
    

Данные подготовлены для обучения модели: они векторизированы, очищены от стоп-слов, лемматизированы. Также они разбиты на тренировачную выборку и тестовую.

## 2 Обучение моделей 

### 2.1. Логистическая регрессия и Дерево решений


```python
def training(model, params):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df = 1)),
        ('model', model)])
    grid = GridSearchCV(pipeline, param_grid = params, cv = 3, n_jobs = -1, scoring = 'f1', verbose = False)
    grid.fit(X_train['lemm_text'], y_train)
    print('Лучший результат:', grid.best_score_)
    print('Лучшие параметры:', grid.best_params_)
    return grid
```


```python
lr_model = training(LogisticRegression(), {"model__C":[7, 10]})
```

    Лучший результат: 0.7806368905162798
    Лучшие параметры: {'model__C': 10}
    


```python
rm_model = training(DecisionTreeClassifier(), {'model__max_depth':[4,10]})
```

    Лучший результат: 0.584888337001264
    Лучшие параметры: {'model__max_depth': 10}
    

На тренировачной выборке себя лучше показала LogisticRegression

### 2.2. Выбор лучшей модели


```python
y_pred = lr_model.predict(X_test['lemm_text'])
```


```python
print("Результат LogisticRegression", f1_score(y_test, y_pred))
```

    Результат LogisticRegression 0.7991872671859126
    

Метрика f1 больше 0.75. Модель справилась.

## Выводы

В проекте нужно было обучить модель классифицировать комментарии на позитивные и негативные. Модель LogisticRegression справилась лучше, чем DecisionTreeClassifier. Метрика f1 на тренировачной выборке 0.78 у первой модели, у второй 0.583. На тестовой выборке метрика f1 модели LogisticRegression составила 0.8.
