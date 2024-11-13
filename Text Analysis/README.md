# Анализ текстов
[md](https://github.com/hundeadove/Portfolio/blob/main/Text%20Analysis/Text%20analysis.md) 
[ipynd](https://github.com/hundeadove/Portfolio/blob/main/Text%20Analysis/Text%20analysis.ipynb)


## Описание проекта
Требуется провести анализ комментариев пользователей на английском языке и выделять токсичные, чтобы отправить на модерацию.

## Навыки и инструменты
* python 
* pandas
* numpy
* matplotlib
* sklearn.pipeline
* nltk.stem.WordNetLemmatizer
* sklearn.feature_extraction.text.TfidfVectorizer
* sklearn.linear_model.LogisticRegression
* sklearn.tree.DecisionTreeClassifier

## Вывод
Была проведена исследовательская работа по обработке текстов и обучению и выбору модели для определения токсичных комментариев по методу TF-IDF. Выбрана LogisticRegression. На тестовой выборке ее метрика f1  составила 0.8.
