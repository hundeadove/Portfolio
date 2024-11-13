# Анализ текстов
md ipynb

## Описание проекта
Требуется анализировать комментарии пользователей на английском языке и выделять токсичные, чтобы отправить на модерацию.

## Навыки и инструменты
python <\br>
pandas<\br>
numpy<\br>
matplotlib<\br>
sklearn.pipeline<\br>
nltk.stem.WordNetLemmatizer<\br>
sklearn.feature_extraction.text.TfidfVectorizer<\br>
sklearn.linear_model.LogisticRegression<\br>
sklearn.tree.DecisionTreeClassifier<\br>

## Вывод
Была проведена исследовательская работа по обработке текстов и обучению и выбору модели для определения токсичных комментариев по методу TF-IDF. Выбрана линейная регрессия. На тестовой выборке метрика f1 модели LogisticRegression составила 0.8.
