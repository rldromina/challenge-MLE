import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

from typing import Tuple, Union, List

class DelayModel:
    THRESHOLD_IN_MINUTES = 15 

    FEATURE_COLS = [
    "OPERA_Latin American Wings",  
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
    ]

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float: 
      fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
      fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
      min_diff = ((fecha_o - fecha_i).total_seconds())/60
      return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1 )
        
        for col in self.FEATURE_COLS: 
            if col not in features.columns:
                features[col] = 0

        features = features[self.FEATURE_COLS]                                                                                                                               
        data = data.copy()

        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)  
            data['delay'] = np.where(data['min_diff'] > self.THRESHOLD_IN_MINUTES, 1, 0)  
            target = data[[target_column]]  
            return features,target

        return features 

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target.iloc[:, 0]                                                                                               
        n_y0 = (y == 0).sum()                                                                                               
        n_y1 = (y == 1).sum()                                                                                               

        self._model = LogisticRegression(class_weight={1: n_y0/len(y), 0: n_y1/len(y)}, max_iter=1000)
        self._model.fit(features, y) 

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * len(features)
        else:
            predictions = self._model.predict(features) 
        return [int(p) for p in predictions]