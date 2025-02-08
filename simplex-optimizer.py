import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from scipy.optimize import linprog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlanificadorProduccion:
    def __init__(self):
        self.COBERTURA_MIN = 3  # Cobertura stock de seguridad mínima (días)
        self.COBERTURA_MAX = 60  # Máxima cobertura de stock permitida (días)
        self.DEMANDA_60D_MIN = 0  # Umbral mínimo de ventas en 60 días (Cj)
        self.TASA_PRODUCCION_MIN = 0  # Tasa mínima de producción (Cj/h)
        self.UMBRAL_VARIACION = 0.20  # Umbral de variación para ajuste de demanda (%)

    def simplex(self, df, horas_disponibles, dias_planificacion, dias_no_habiles):
        try:
            df_work = df.copy()
            
            # Filtrar productos con demanda válida
            df_work = df_work[
                (df_work['Vta -60'] > self.DEMANDA_60D_MIN) &
                (df_work['demanda_media'] > 0)
            ].copy()
            
            # Calcular días hábiles y demanda del periodo
            dias_habiles = dias_planificacion - dias_no_habiles
            
            # Calcular cobertura inicial y stock de seguridad
            df_work['demanda_periodo'] = (df_work['demanda_media'] * dias_planificacion).round(0)
            df_work['stock_seguridad'] = (df_work['demanda_media'] * self.COBERTURA_MIN).round(0)
            df_work['cobertura_inicial'] = (df_work['stock_inicial'] / df_work['demanda_media']).round(1)
            
            # Filtrar productos que necesitan producción
            df_work = df_work[df_work['cobertura_inicial'] < self.COBERTURA_MAX]
            
            logger.info(f"Lista entrada al Simplex con {len(df_work)} filas:\n{df_work[['COD_ART', 'stock_inicial', 'demanda_media', 'Cj/H','cobertura_inicial']]}")
            
            if df_work.empty:
                logger.info("No hay productos que requieran producción")
                return None
                
            n_productos = len(df_work)
            
            # Función objetivo: minimizar horas totales de producción
            c = 1/df_work['Cj/H'].values  # Coeficiente es horas por unidad producida
            
            # 1. Restricción de cobertura mínima de 3 días
            A_cob = np.zeros((n_productos, n_productos))
            for i in range(n_productos):
                # Consideramos la demanda del periodo total para calcular cobertura
                A_cob[i, i] = 1/df_work['demanda_media'].iloc[i]
            b_cob = 3 - df_work['stock_inicial'].values/df_work['demanda_media'].values
            
            # 2. Restricción de horas disponibles (solo en días hábiles)
            A_horas = np.zeros((1, n_productos))
            A_horas[0] = 1/df_work['Cj/H'].values
            b_horas = [horas_disponibles]  # horas_disponibles ya viene calculado con días hábiles
            
            # 3. Restricción de horas mínimas por producto (2 horas)
            A_min_horas = np.zeros((n_productos, n_productos))
            for i in range(n_productos):
                A_min_horas[i, i] = -1/df_work['Cj/H'].iloc[i]
            b_min_horas = np.full(n_productos, -2)  # Mínimo 2 horas
            
            # Combinar todas las restricciones
            A = np.vstack([-A_cob, A_horas, A_min_horas])
            b = np.concatenate([-b_cob, b_horas, b_min_horas])
            
            # Resolver con Simplex
            result = linprog(
                c,
                A_ub=A, 
                b_ub=b,
                method='highs'
            )
            
            if not result.success:
                logger.error(f"No se encontró solución óptima: {result.message}")
                return None
                
            # Extraer las cantidades a producir
            produccion_optima = pd.Series(np.round(result.x, 0), index=df_work.index)
            
            # Calcular y loguear métricas
            horas_por_producto = produccion_optima / df_work['Cj/H']
            cobertura_final = (df_work['stock_inicial'] + produccion_optima) / df_work['demanda_media']
            
            logger.info(f"Métricas de la solución (considerando {dias_habiles} días hábiles de {dias_planificacion} días totales):")
            logger.info(f"Horas totales necesarias: {horas_por_producto.sum():.2f}")
            logger.info(f"Horas restantes: {horas_disponibles - horas_por_producto.sum():.2f}")
            logger.info("\nCobertura por producto:")
            for idx in df_work.index:
                logger.info(f"Producto {df_work.loc[idx, 'COD_ART']}: {cobertura_final[idx]:.1f} días")
            
            return produccion_optima
            
        except Exception as e:
            logger.error(f"Error en optimización Simplex: {str(e)}")
            logger.error("Traza completa:", exc_info=True)
            return None

    def aplicar_filtros(self, df):
        # Filtros para selección de productos
        df_filtrado = df.copy()
        df_filtrado = df_filtrado[df_filtrado['Vta -60'] > self.DEMANDA_60D_MIN]
        df_filtrado = df_filtrado[df_filtrado['Cj/H'] > self.TASA_PRODUCCION_MIN]
        
        return df_filtrado.drop_duplicates(subset=['COD_ART'], keep='last')

    def calcular_demanda(self, df):
        """
        Calcula la demanda media basándose en históricos y variaciones.
        """
        # Asegurar que no haya valores nulos
        df['M_Vta -15'] = df['M_Vta -15'].fillna(0)
        df['M_Vta -15 AA'] = df['M_Vta -15 AA'].fillna(0)
        df['M_Vta +15 AA'] = df['M_Vta +15 AA'].fillna(0)

        # Calcular variación interanual
        with np.errstate(divide='ignore', invalid='ignore'):
            df['variacion_aa'] = np.abs(1 - df['M_Vta +15 AA'] / df['M_Vta -15 AA'])
        df['variacion_aa'] = df['variacion_aa'].fillna(0)

        # Calcular demanda media
        condicion_sin_datos_aa = (df['M_Vta -15 AA'] == 0) | (df['M_Vta +15 AA'] == 0)
        condicion_variacion = (df['variacion_aa'] > self.UMBRAL_VARIACION) & (df['variacion_aa'] < 1)

        df['demanda_media'] = np.where(
            condicion_sin_datos_aa,
            df['M_Vta -15'],
            np.where(
                condicion_variacion,
                df['M_Vta -15'] * (df['M_Vta +15 AA'] / df['M_Vta -15 AA']),
                df['M_Vta -15']
            )
        )

        # Asegurar que la demanda media no sea negativa
        df['demanda_media'] = np.maximum(df['demanda_media'], 0)

        return df

    def cargar_datos(self, carpeta="Dataset", fecha_dataset=None, fecha_inicio=None):
        try:
            # Validar fecha dataset
            if not fecha_dataset:
                raise ValueError("Debe proporcionar una fecha del dataset a utilizar.")
            
            # Normalizar fecha para búsqueda de archivo
            fecha_normalizada = datetime.strptime(
                fecha_dataset.replace("/", "-"), 
                "%d-%m-%Y"
            ).strftime("%d-%m-%y")
            
            # Encontrar archivo correspondiente
            archivo = next(
                (f for f in os.listdir(carpeta) 
                 if f.endswith('.csv') and fecha_normalizada in f),
                None
            )
            
            if not archivo:
                raise FileNotFoundError(
                    f"No se encontró archivo para fecha: {fecha_normalizada}"
                )
            
            ruta_archivo = os.path.join(carpeta, archivo)
            logger.info(f"Cargando archivo: {ruta_archivo}")
            
            # Cargar archivo CSV
            df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1', skiprows=4)
            df = df[df['COD_ART'].notna()]
            
            # Columnas numéricas a convertir
            columnas_numericas = ['Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 
                                'Stock Externo', 'M_Vta -15 AA', 'M_Vta +15 AA', 
                                'Vta -60', 'OF']
            
            # Convertir columnas a numérico
            for col in columnas_numericas:
                df[col] = pd.to_numeric(
                    df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True),
                    errors='coerce'
                ).fillna(0)

            fecha_dataset = pd.to_datetime(fecha_dataset, format='%d-%m-%Y')
            fecha_inicio = pd.to_datetime(fecha_inicio, format='%d-%m-%Y')

            # Calcular incremento de stock antes de inicio de programación
            df['1ª OF'] = pd.to_datetime(df['1ª OF'].replace('(en blanco)', pd.NaT), format='%d/%m/%Y', errors='coerce')

            # Actualizar disponible inicial considerando OFs programadas
            if not df['1ª OF'].isna().all():
                mask = (df['1ª OF'] >= fecha_dataset) & (df['1ª OF'] < fecha_inicio)
                df['Disponible_Inicial'] = df['Disponible']
                df.loc[mask, 'Disponible_Inicial'] = df.loc[mask, 'Disponible'].fillna(0) + df.loc[mask, 'OF'].fillna(0)
            else:
                df['Disponible_Inicial'] = df['Disponible']

            # Aplicar filtros y calcular demanda
            df = self.aplicar_filtros(df)
            df = self.calcular_demanda(df)

            # Calcular demanda provisional y stock inicial
            df['demanda_prov'] = (fecha_inicio - fecha_dataset).days * df['demanda_media'].fillna(0)
            df['stock_inicial'] = (
                df['Disponible_Inicial'].fillna(0) + 
                df['Calidad'].fillna(0) + 
                df['Stock Externo'].fillna(0) -
                df['demanda_prov'].fillna(0)
            )

            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None

    def generar_plan_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles, fecha_inicio=None, usar_simplex=False):
        try:
            logger.info("=== Iniciando generación de plan ===")
            logger.info(f"{'Usando Simplex' if usar_simplex else 'Usando método propio'}")
            logger.info(f"Columnas disponibles: {df.columns.tolist()}")
            logger.info(f"Datos iniciales:\n{df[['COD_ART', 'Disponible','stock_inicial', 'demanda_media', 'Cj/H','1ª OF', 'OF']]}")
            
            if isinstance(fecha_inicio, str):
                fecha_inicio = datetime.strptime(fecha_inicio, "%d-%m-%Y")
            
            plan = df.copy()
            # Generar producción óptima
            produccion_optima = (
                self.simplex(plan, horas_disponibles, dias_planificacion, dias_no_habiles)
                if usar_simplex else
                self.optimizar_produccion(plan, horas_disponibles, dias_planificacion, dias_no_habiles)
            )
            
            # Validar producción óptima
            if produccion_optima is None:
                logger.warning("No se obtuvo producción óptima")
                return None, None, None
                
            if produccion_optima.empty:
                logger.warning("Producción óptima está vacía")
                return None, None, None
            
            # Asignar cajas y horas de producción
            plan.loc[produccion_optima.index, 'cajas_a_producir'] = produccion_optima
            plan['horas_necesarias'] = (plan['cajas_a_producir'] / plan['Cj/H']).round(1)
            
            # Filtrar productos con producción
            plan = plan[plan['cajas_a_producir'] > 0].copy()
            
            if plan.empty:
                logger.warning("Plan está vacío después de filtrar cajas a producir")
                return None, None, None
                
            # Calcular fechas de planificación
            fecha_fin = fecha_inicio + timedelta(days=dias_planificacion)
            return plan, fecha_inicio, fecha_fin
            
        except Exception as e:
            logger.error(f"Error generando plan: {str(e)}", exc_info=True)
            return None, None, None

    def generar_reporte(self, plan, fecha_inicio, fecha_fin):
        try:
            if plan is None or len(plan) == 0:
                logger.error("No hay plan para reportar")
                return
                
            print(f"\nPLAN DE PRODUCCIÓN ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
            print("-" * 80)
            print(f"Total productos: {len(plan)}")
            print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
            print("\nDetalle por producto:")
            
            columnas = ['COD_ART', 'NOM_ART', 'Disponible_Inicial', 'Calidad', 'Stock Externo', 
                       'stock_inicial', 'demanda_media', 'cajas_a_producir', 'horas_necesarias']
            print(plan[columnas].to_string(index=False))
            
            # Guardar reporte en archivo CSV
            ruta_reporte = f"plan_produccion_{fecha_inicio.strftime('%Y%m%d')}_{fecha_fin.strftime('%Y%m%d')}.csv"
            plan[columnas].to_csv(ruta_reporte, index=False)
            logger.info(f"Reporte guardado en {ruta_reporte}")
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")


def main():
    try:
        fecha_dataset = input("Ingrese fecha de dataset (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        fecha_inicio = input("Ingrese fecha inicio planificación (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        dias_planificacion = int(input("Ingrese días de planificación: "))
        dias_no_habiles = float(input("Ingrese días no hábiles en el periodo: "))
        horas_mantenimiento = int(input("Ingrese horas de mantenimiento: "))
        usar_simplex = input("¿Usar método Simplex? (s/n): ").strip().lower() == 's'
        
        # Validaciones de entrada
        if dias_planificacion <= 0:
            raise ValueError("Los días de planificación deben ser positivos")
        if dias_no_habiles < 0 or dias_no_habiles >= dias_planificacion:
            raise ValueError("Días no hábiles inválidos")
        if horas_mantenimiento < 0:
            raise ValueError("Las horas de mantenimiento no pueden ser negativas")
        
        # Convertir fechas para comparación
        fecha_dataset_dt = datetime.strptime(fecha_dataset.replace("/", "-"), "%d-%m-%Y")
        fecha_inicio_dt = datetime.strptime(fecha_inicio.replace("/", "-"), "%d-%m-%Y")
        if fecha_inicio_dt < fecha_dataset_dt:
            raise ValueError("La fecha de inicio debe ser mayor o igual a la fecha de dataset")
            
        # Calcular horas disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        logger.info(f"Horas disponibles para producción: {horas_disponibles}")
        
        # Inicializar planificador y cargar datos
        planificador = PlanificadorProduccion()
        
        logger.info("Cargando datos...")
        df = planificador.cargar_datos(fecha_dataset=fecha_dataset, fecha_inicio=fecha_inicio)
        if df is None:
            logger.error("Error al cargar datos")
            return
            
        logger.info(f"Datos cargados y filtrados: {len(df)} productos")
        
        # Generar plan de producción
        logger.info("Generando plan de producción...")
        plan, fecha_inicio_dt, fecha_fin = planificador.generar_plan_produccion(
            df, 
            horas_disponibles,
            dias_planificacion,
            dias_no_habiles,
            fecha_inicio,
            usar_simplex
        )
        
        if plan is None:
            logger.error("No se pudo generar el plan")
            return
            
        # Generar reporte final
        logger.info("Generando reporte...")
        planificador.generar_reporte(plan, fecha_inicio_dt, fecha_fin)
        
        logger.info("Proceso completado exitosamente")
        
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}")
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()