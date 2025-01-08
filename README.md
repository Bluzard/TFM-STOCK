# Sistema de IA para Gestión de Stock y Planificación de Producción

## Descripción
Sistema de inteligencia artificial diseñado para optimizar la planificación de producción y gestión de stock, priorizando evitar roturas de stock mientras mantiene niveles de inventario eficientes.

## Características Principales
- Predicción avanzada de demanda utilizando múltiples modelos de ML
- Optimización de producción considerando múltiples restricciones
- Sistema de priorización basado en niveles de stock y demanda
- Validación contra datos históricos y resultados esperados
- Generación de reportes detallados

## Requisitos
```bash
Python 3.8+
pandas
numpy
scikit-learn
```

## Instalación
1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Preparación de Datos
El sistema requiere un archivo CSV con la siguiente estructura:
- COD_ART: Código de artículo
- Cj/H: Cajas por hora de producción
- Disponible: Stock disponible actual
- Calidad: Stock en control de calidad
- Stock Externo: Stock en almacenes externos
- M_Vta -15: Ventas últimos 15 días
- M_Vta -15 AA: Ventas últimos 15 días año anterior

### Ejecución
```python
from stock_management import StockManagementSystem

system = StockManagementSystem(
    data_path="stock_data.csv",
    plan_date="2024-12-31"
)

production_plan = system.optimize_production(
    available_hours=100,
    maintenance_hours=5
)

report = system.generate_production_report(production_plan)
```

### Configuración
Parámetros principales ajustables:
```python
SAFETY_STOCK_DAYS = 3  # Días mínimos de cobertura
MIN_PRODUCTION_HOURS = 2  # Horas mínimas por lote
```

## Estructura del Sistema

### 1. Predicción de Demanda
- Utiliza múltiples modelos de ML (Linear, Huber, RANSAC, Random Forest, GBM)
- Selección automática del mejor modelo basado en MAE
- Validación cruzada para evaluación robusta

### 2. Optimización de Producción
- Priorización basada en:
  * Niveles actuales de stock
  * Demanda predicha
  * Eficiencia de producción
- Restricciones consideradas:
  * Horas disponibles
  * Stock de seguridad
  * Lotes mínimos de producción

### 3. Generación de Reportes
- Plan detallado de producción
- Métricas de rendimiento
- Análisis de cobertura
- Validación contra históricos

## Rendimiento
- MAE típico: ~20-25 unidades
- Error porcentual medio: ~5-6%
- R² en predicciones: >0.95

## Limitaciones y Consideraciones
- El sistema prioriza evitar roturas de stock sobre optimización de inventario
- Se recomienda reejecutar el sistema ante cambios significativos en datos de entrada
- Los tiempos de mantenimiento deben ser considerados en las horas disponibles

## Ejemplos de Uso

### Planificación Básica
```python
system = StockManagementSystem("stock_data.csv", "2024-12-31")
plan = system.optimize_production(100, 5)
report = system.generate_production_report(plan)
```

### Con Pedidos Pendientes
```python
pending_orders = {"001": 50, "002": 30}
plan = system.optimize_production(100, 5, pending_orders)
```

### Validación contra Resultados Esperados
```python
system = StockManagementSystem(
    "stock_data.csv",
    "2024-12-31",
    expected_results_path="expected.csv"
)
```

## Resultados Típicos
```
Métricas del modelo:
- MAE: 21.29
- RMSE: 24.45
- R2: 0.99

Plan de producción:
- Total cajas: 7,821
- Horas asignadas: 95
- Productos programados: 4
```