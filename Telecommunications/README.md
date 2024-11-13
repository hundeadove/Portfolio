# Телекоммуникация
[md](https://github.com/hundeadove/Portfolio/blob/main/Telecommunications/Telecommunications%20.md)
[ipynd](https://github.com/hundeadove/Portfolio/blob/main/Telecommunications/Telecommunications%20.ipynb)

## Описание проекта
Создание модели, которая позволит компании "ТелеДом" выявлять клиентов с высоким риском оттока и предлагать им специальные условия, чтобы сохранить их лояльность.

## Навыки и инструменты
* pandas 
* numpy
* matplotlib
* seaborn 
* phik
* sklearn
* catboost
* sklearn.preprocessing.StandardScaler
* sklearn.ensemble.RandomForestClassifier
* sklearn.linear_model.LogisticRegression
* sklearn.dummy.DummyClassifier
* sklearn.metrics.confusion_matrix
* 
## Общий вывод
Были построены три модели, лучшей оказалась CatBoostClassifier. На тестовой выборке ее площадь ROC-кривой составила 0.84. Acсuracy на тестовой выборке составила 0.87. В зоне риска находятся клиенты, которые уже прошли пероид адаптации и уже неплохо разбираются в этом рынке.
