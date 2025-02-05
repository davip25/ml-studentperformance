import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

# Función para guardar objetos (modelos, etc.)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

# Función para evaluar modelos usando GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Recorremos todos los modelos proporcionados
        for model_name, model in models.items():
            param = params.get(model_name)
            if param is None:
                raise ValueError(f"No parameters provided for model: {model_name}")
            
            # Usamos GridSearchCV para encontrar los mejores parámetros
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Tomamos el mejor modelo encontrado por GridSearchCV
            best_model = gs.best_estimator_

            # Realizamos las predicciones con el mejor modelo
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculamos el R^2 para entrenamiento y prueba
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Guardamos el puntaje del modelo en el reporte
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# Función para cargar objetos guardados (modelos, etc.)
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
