# Анализ текстов
md ipynb

## Описание проекта
Требуется анализировать комментарии пользователей на английском языке и выделять токсичные, чтобы отправить на модерацию.

## Навыки и инструменты
* python 
* pandas
* numpy
matplotlib
sklearn.pipeline
nltk.stem.WordNetLemmatizer
sklearn.feature_extraction.text.TfidfVectorizer
sklearn.linear_model.LogisticRegression
sklearn.tree.DecisionTreeClassifier

## Вывод
Была проведена исследовательская работа по обработке текстов и обучению и выбору модели для определения токсичных комментариев по методу TF-IDF. Выбрана линейная регрессия. На тестовой выборке метрика f1 модели LogisticRegression составила 0.8.