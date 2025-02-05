import sys
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path('artifacts') / 'train.csv'
    test_data_path: Path = Path('artifacts') / 'test.csv'
    raw_data_path: Path = Path('artifacts') / 'data.csv'

class DataIngestion:
    """
    Handles the data ingestion process. Reads the dataset, splits it into 
    training and testing sets, and saves them to CSV files.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple:
        """
        Reads the dataset, performs a train-test split, and saves the 
        training and testing sets to separate CSV files.

        Returns:
            tuple: A tuple containing the file paths to the training and 
                   testing data CSV files.
        """
        logging.info('Entered the Data ingestion method or Component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Consider making this path configurable
            logging.info('Read the Dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=36)  # More descriptive random_state name?

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))