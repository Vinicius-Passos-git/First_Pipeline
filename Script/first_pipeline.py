#%%
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

print('Bibliotecas importadas com sucesso!')

housing = pd.read_csv('housing.csv')

print ('Dados importados com sucesso!')

X_train, X_test, y_train, y_test = train_test_split(housing.drop(['median_house_value'], axis=1), 
                                                    housing['median_house_value'], 
                                                    test_size=0.2, 
                                                    random_state=42)

print('Train_test criado com sucesso!')
# criando o modelo usando pipeline
model = Pipeline(steps=[
    ('one-hot encoder', OneHotEncoder()),
    ('imputer', SimpleImputer(strategy='median')),
    ('forest_regressor', RandomForestRegressor(random_state=42))
])


model.fit(X_train, y_train)

print('Modelo treinado com sucesso!')

housing_predictions = model.predict(X_train)

some_data = X_train.iloc[:5]
model.predict(some_data)