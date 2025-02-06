from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.components.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

# Router for Home Page
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score')
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=round(results[0]))
        
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    # Gunicorn manejará la ejecución
    pass
