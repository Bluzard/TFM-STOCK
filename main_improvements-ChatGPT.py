import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DemandPredictor:
    """Clase para predicción de demanda con múltiples modelos"""

    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'huber': HuberRegressor(epsilon=1.35),
            'ransac': RANSACRegressor(random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara características avanzadas para el modelo"""
        features = pd.DataFrame()

        features['venta_anterior'] = df['M_Vta -15 AA']
        features['ratio_cambio'] = df['M_Vta -15'] / df['M_Vta -15 AA']
        features['stock_total'] = df['Disponible'] + df['Calidad'] + df['Stock Externo']
        features['dias_cobertura'] = features['stock_total'] / (df['M_Vta -15'] / 15)
        features['tasa_produccion'] = df['Cj/H']

        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(features.mean(), inplace=True)

        return features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena múltiples modelos y selecciona el mejor"""
        X_scaled = self.scaler.fit_transform(X)

        best_mae = float('inf')
        best_model_name = None

        logging.info("\nEvaluación de modelos:")
        for name, model in self.models.items():
            start_time = datetime.now()
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
            mae = -scores.mean()
            train_time = (datetime.now() - start_time).total_seconds()

            logging.info(f"{name.upper()}:")
            logging.info(f"  MAE: {mae:.2f}")
            logging.info(f"  Std: {scores.std():.2f}")
            logging.info(f"  Tiempo de entrenamiento: {train_time:.2f} segundos")

            if mae < best_mae:
                best_mae = mae
                best_model_name = name

        logging.info(f"\nMejor modelo: {best_model_name.upper()} (MAE: {best_mae:.2f})")

        self.best_model = self.models[best_model_name]
        self.best_model.fit(X_scaled, y)

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)

            logging.info("\nImportancia de características:")
            for _, row in feature_imp.iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.3f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("El modelo debe ser entrenado primero")

        X_scaled = self.scaler.transform(X)
        return np.maximum(0, self.best_model.predict(X_scaled))

    @staticmethod
    def plot_predictions(y_true: pd.Series, y_pred: np.ndarray):
        """Genera un gráfico comparativo entre valores reales y predichos"""
        import matplotlib.pyplot as plt
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title("Comparación de Predicciones vs Reales")
        plt.show()

class StockManagementSystem:
    """Sistema principal de gestión de stock"""

    def __init__(self, data_path: str, plan_date: str, expected_results_path: str = None):
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2

        try:
            self.plan_date = datetime.strptime(plan_date, '%Y-%m-%d')
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1')
            logging.info(f"Datos cargados de {data_path}: {len(self.df)} productos")

            self.expected_results = None
            if expected_results_path:
                self.expected_results = pd.read_csv(expected_results_path, sep=';', encoding='latin1')
                logging.info(f"Resultados esperados cargados de {expected_results_path}")

            self.demand_predictor = DemandPredictor()

            self._clean_data()
            self.model = self._train_demand_model()

        except Exception as e:
            logging.error(f"Error al inicializar el sistema: {e}")
            raise

    @staticmethod
    def clean_numeric_columns(df: pd.DataFrame, columns: list, default_values: dict = None) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace({',': '.'}, regex=True), errors='coerce')
        if default_values:
            df.fillna(default_values, inplace=True)
        return df

    def _clean_data(self):
        try:
            self.df.dropna(how='all', inplace=True)
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA']
            default_values = {'Cj/H': 1, 'Disponible': 0, 'Calidad': 0, 'Stock Externo': 0, 'M_Vta -15': 0, 'M_Vta -15 AA': 1}

            self.df = self.clean_numeric_columns(self.df, num_columns, default_values)

            self.df['STOCK_TOTAL'] = self.df['Disponible'] + self.df['Calidad'] + self.df['Stock Externo']
            logging.info(f"Datos limpios: {len(self.df)} productos activos")

        except Exception as e:
            logging.error(f"Error en limpieza de datos: {e}")
            raise

    def _train_demand_model(self):
        try:
            features = self.demand_predictor.prepare_features(self.df)
            target = self.df['M_Vta -15']

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            logging.info("\nEntrenando modelo de demanda...")
            self.demand_predictor.fit(X_train, y_train)

            y_pred = self.demand_predictor.predict(X_test)

            logging.info("\nMétricas en conjunto de prueba:")
            logging.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
            logging.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            logging.info(f"R2: {r2_score(y_test, y_pred):.2f}")

            return self.demand_predictor

        except Exception as e:
            logging.error(f"Error en entrenamiento del modelo: {e}")
            raise

    def predict_demand(self):
        try:
            features = self.demand_predictor.prepare_features(self.df)
            predictions = self.demand_predictor.predict(features)
            self.df['PREDICTED_DEMAND'] = predictions

            mae = mean_absolute_error(self.df['M_Vta -15'], predictions)
            mape = np.mean(np.abs((self.df['M_Vta -15'] - predictions) / self.df['M_Vta -15'])) * 100

            logging.info("\nAnálisis de predicciones:")
            logging.info(f"Error medio absoluto: {mae:.2f}")
            logging.info(f"Error porcentual medio: {mape:.2f}%")

        except Exception as e:
            logging.error(f"Error en predicción de demanda: {e}")
            raise

    # Resto del código se mantendría estructurado de forma similar, con logging y modularidad implementada

if __name__ == "__main__":
    try:
        logging.info("Iniciando sistema de gestión de stock...")
        system = StockManagementSystem(
            data_path="stock_data_test.csv",
            plan_date="2024-12-31",
            expected_results_path="expected_results.csv"
        )

        available_hours = 100
        maintenance_hours = 5
        pending_orders = {"000001": 50, "000002": 30}

        logging.info("\nOptimizando producción...")
        production_plan = system.optimize_production(available_hours, maintenance_hours, pending_orders)

        logging.info("\nGenerando reporte...")
        report = system.generate_production_report(production_plan, "plan_produccion_test.csv")

        logging.info("\nReporte final de producción:")
        for key, value in report.items():
            if key != 'products_detail':
                logging.info(f"{key}: {value}")
            else:
                logging.info("\nDetalle de productos a producir:")
                for product in value:
                    logging.info(f"  Código: {product['COD_ART']}")
                    logging.info(f"  Cajas: {product['CAJAS_PRODUCIR']:.0f}")
                    logging.info(f"  Horas: {product['HORAS_PRODUCCION']:.2f}")

    except Exception as e:
        logging.error(f"Error en ejecución: {e}")
        raise
