import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Definir los modelos que se evaluarán
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            # Definir los hiperparámetros para GridSearch
            param_grid = {
                'Random Forest': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20]
                },
                'Decision Tree Regressor': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 10]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1]
                },
                'K-Neighbors Regressor': {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance']
                },
                'XGB Regressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1]
                },
                'AdaBoost Regressor': {
                    'n_estimators': [50, 100]
                },
            }

            model_report = {}

            # Entrenar y evaluar cada modelo con GridSearchCV
            for model_name, model in models.items():
                try:
                    # Si el modelo tiene parámetros de GridSearch, realizamos la búsqueda
                    if model_name in param_grid:
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=3)
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        predicted = best_model.predict(X_test)
                        r2_square = r2_score(y_test, predicted)
                        model_report[model_name] = r2_square
                        logging.info(f"{model_name} Best Params: {best_params} | R2 Score: {r2_square}")
                    else:
                        # Si no tiene parámetros para GridSearch, entrenamos el modelo normalmente
                        model.fit(X_train, y_train)
                        predicted = model.predict(X_test)
                        r2_square = r2_score(y_test, predicted)
                        model_report[model_name] = r2_square
                        logging.info(f"{model_name} R2 Score: {r2_square}")
                except Exception as e:
                    logging.error(f"Error with model {model_name}: {e}")
                    model_report[model_name] = None

            # Filtrar los modelos que tienen un valor de R2
            model_report = {name: score for name, score in model_report.items() if score is not None}

            # Seleccionar el mejor modelo
            if not model_report:
                raise CustomException('No valid models found')

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if model_report[best_model_name] < 0.6:
                raise CustomException(f'No best model found. Best R2 is below 0.6')

            logging.info(f'Best model found: {best_model_name} with R2 score: {model_report[best_model_name]}')

            # Guardar el mejor modelo
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Incluir el modelo seleccionado como parte de la salida
            return model_report, best_model_name, best_model

        except Exception as e:
            raise CustomException(e, sys)
