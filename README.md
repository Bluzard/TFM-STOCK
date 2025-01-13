# Sistema de Planificación de Producción

## Descripción
Sistema de optimización para planificación de producción basado en predicción de demanda y método Simplex, diseñado para minimizar roturas de stock mientras optimiza recursos productivos.

## Requisitos
```
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.2
tkinter (incluido en Python)
joblib >= 1.1.0
```

## Instalación

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto
```
proyecto/
│
├── main.py              # Interfaz principal
├── modelo.py            # Implementación del modelo
├── optimizador.py       # Sistema de optimización
├── utilidades.py        # Funciones auxiliares
│
├── modelos/            # Directorio de modelos guardados
│   └── *.joblib        # Modelos entrenados
│
└── requirements.txt    # Dependencias del proyecto
```

## Uso del Sistema

### Inicio Rápido
```python
python main.py
```

### Archivos de Entrada
Formato CSV/Excel con estructura:
```
COD_ART,Cj/H,Disponible,Calidad,Stock Externo,M_Vta -15,M_Vta -15 AA
001,120,500,50,0,450,400
002,80,300,0,100,200,180
...
```

### Parámetros Configurables
```python
# config.py
DIAS_STOCK_SEGURIDAD = 3    # Cobertura mínima
HORAS_MIN_PRODUCCION = 2    # Lote mínimo
HORAS_DISPONIBLES = 100     # Capacidad diaria
HORAS_MANTENIMIENTO = 5     # Tiempo reservado
```

## Componentes Principales

### 1. Sistema de Predicción
```python
from modelo import ModeloPrediccion

modelo = ModeloPrediccion()
modelo.entrenar(X_train, y_train)
predicciones = modelo.predecir(X_test)
```

### 2. Optimizador de Producción
```python
from optimizador import Optimizador

opt = Optimizador(dias_stock_seguridad=3, horas_min_produccion=2)
plan = opt.optimizar_produccion(df, horas_disponibles=100)
```

### 3. Utilidades y Procesamiento
```python
from utilidades import Utilidades

util = Utilidades()
df = util.procesar_datos(df_input)
util.validar_datos(df)
```

## Salidas del Sistema

### Plan de Producción
Archivo CSV con:
- Plan detallado por producto
- Horas asignadas
- Predicciones de demanda
- Análisis de cobertura

### Métricas de Rendimiento
```
MAE Típico: 21.29
RMSE: 24.45
R²: 0.99
Error porcentual: ~5.53%
```

## Mantenimiento y Mejoras

### Entrenamiento de Modelos
```python
# Entrenar nuevo modelo
modelo.entrenar(X, y)

# Cargar modelo específico
modelo.cargar_modelo("modelo_20250112_204324")
```

### Validación de Resultados
```python
metricas = modelo.validar_predicciones(pred, real)
print(f"Error medio: {metricas['error_medio']:.2f}")
```

## Notas Importantes
- Mantener datos de entrada actualizados
- Reentrenar modelo ante cambios significativos
- Validar parámetros de optimización regularmente
- Considerar tiempos de mantenimiento en planificación