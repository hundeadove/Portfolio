# HR-аналитика: машинное обучение на службе персонала
[md](https://github.com/hundeadove/Portfolio/blob/main/Text%20Analysis/Text%20analysis.md)
[ipynd](https://github.com/hundeadove/Portfolio/blob/main/Text%20Analysis/Text%20analysis.md)

## Описание проекта
Разработать две модели для прогнозирования 1) уровеня удовлетворенности сотрудника на основе данных о нем и 2) оттока сотрудников.

## Навыки и инструменты
* python
* pandas
* numpy
* matplotlib
* seaborn
* phik
* sklearn
* sklearn.pipeline.Pipeline
* sklearn.linear_model.LinearRegression
* sklearn.neighbors.KNeighborsClassifier
* sklearn.ensemble.RandomForestClassifier
* sklearn.svm.SVC
* sklearn.tree.DecisionTreeRegressor
* sklearn.tree.DecisionTreeClassifier

## Общий вывод
Первая модель DecisionTreeRegressor смогла спрогнозировать уровень удовлетворённости сотрудника с точностью 14.3/15.00 по метрике SMAPE на тестовой выборке. 
Модель RandomForestClassifier для второй задачи на тестовой выборке показала результут 0.93. 
При анализе была выявлена прямая взаимосвязь между «покинет ли сотрудник компанию» и «его удовлетворённостью работой в этой компании».
