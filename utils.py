# Imports
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.linear_model import LinearRegression

def plot_series(index: pd.Series, y_values_dict: dict[str, pd.Series], title: str) -> None:
    fig = go.Figure()

    for label, y_values in y_values_dict.items():
        fig.add_trace(go.Scatter(x=index, y=y_values, mode='lines', name=label))
    fig.update_layout(xaxis_title="Date", title=title)
    fig.show()

class MLModel:
    """Classe de base pour les modèles de machine learning sur séries temporelles"""
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
    
    def augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des features temporelles au DataFrame"""
        df_feat = df.copy()
        df_feat["dayofweek"] = df_feat.index.dayofweek
        df_feat["month"] = df_feat.index.month
        df_feat["dayofmonth"] = df_feat.index.day
        df_feat["dayofyear"] = df_feat.index.dayofyear
        df_feat["weekofyear"] = df_feat.index.isocalendar().week
        df_feat["lag_1"] = df_feat["Close"].shift(1)
        return df_feat.dropna()
    
    def fit(self, X_train: pd.DataFrame) -> None:
        """Entraîne le modèle sur les données d'entraînement"""
        self.X_train = self.augment_features(X_train)
        self.y_train = self.X_train["Close"]
        self.X_train = self.X_train.drop(columns=["Close"])
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X_test: pd.DataFrame) -> float:
        """Prédit la valeur suivante"""
        X_test_aug = self.augment_features(X_test)
        X_test_aug = X_test_aug.drop(columns=["Close"])
        return self.model.predict(X_test_aug)[0]

class XGBModel(MLModel):
    """Modèle XGBoost pour la prédiction de séries temporelles"""
    
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.1)

class LinearRegressionModel(MLModel):
    """Modèle de régression linéaire pour la prédiction de séries temporelles"""
    
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

