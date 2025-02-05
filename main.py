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
        ## Parámetros base para la planificación
        self.DIAS_STOCK_SEGURIDAD = 3
        self.MIN_VENTA_60D = 0  ## Umbral mínimo de ventas en 60 días
        self.MIN_TASA_PRODUCCION = 0  ## Tasa mínima de producción
        self.UMBRAL_VARIACION = 0.20  ## Umbral de variación para ajuste de demanda
        self.COBERTURA_MAX_FINAL = 60  ## Máxima cobertura de stock permitida

    def simplex(self, df, horas_disponibles, dias_planificacion, dias_no_habiles):
        try:
            df_work = df.copy()
            
            # Filtrar productos con demanda válida
            df_work = df_work[
                (df_work['Vta -60'] > self.MIN_VENTA_60D) &
                (df_work['demanda_media'] > 0)
            ].copy()
            
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
            
            # 1. Restricción de cobertura mínima de 3 días (desigualdad)
            A_cob = np.zeros((n_productos, n_productos))
            for i in range(n_productos):
                A_cob[i, i] = 1/df_work['demanda_media'].iloc[i]  # Conversión a días de cobertura
            b_cob = 3 - df_work['stock_inicial'].values/df_work['demanda_media'].values
            
            # 2. Restricción de horas disponibles
            A_horas = np.zeros((1, n_productos))
            A_horas[0] = 1/df_work['Cj/H'].values
            b_horas = [horas_disponibles]
            
            # 3. Restricción de horas mínimas por producto (2 horas)
            A_min_horas = np.zeros((n_productos, n_productos))
            for i in range(n_productos):
                A_min_horas[i, i] = -1/df_work['Cj/H'].iloc[i]  # Convertir cajas a horas
            b_min_horas = np.full(n_productos, -2)  # Mínimo 2 horas
            
            # Combinar todas las restricciones
            A = np.vstack([-A_cob, A_horas, A_min_horas])  # -A_cob porque queremos >= en lugar de <=
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
            
            logger.info("Métricas de la solución:")
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

    def optimizar_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles):
        try:
            dias_habiles = dias_planificacion - dias_no_habiles
            logger.info(f"Días hábiles para planificación: {dias_habiles}")
            
            df_work = df.copy()
            ## Filtrar productos con demanda positiva
            df_work = df_work[df_work['demanda_media'] > 0].copy()
            
            ## Calcular cobertura y necesidades
            df_work['cobertura_actual'] = (df_work['stock_total'] / df_work['demanda_media']).round(2)
            df_work['dias_cobertura'] = (df_work['stock_total'] / df_work['demanda_media']).round(2)
            df_work['demanda_periodo'] = (df_work['demanda_media'] * dias_habiles).round(0)
            df_work['stock_seguridad'] = (df_work['demanda_media'] * self.DIAS_STOCK_SEGURIDAD).round(0)
            
            ## Calcular necesidad base de producción
            df_work['necesidad_base'] = df_work.apply(lambda row: max(
                row['stock_seguridad'] - row['stock_total'],
                row['demanda_periodo'] - (row['stock_total'] - row['stock_seguridad'])
            ), axis=1)
            
            ## Filtrar productos que necesitan producción
            df_work = df_work[
                (df_work['cobertura_actual'] <= self.COBERTURA_MAX_FINAL) &
                (df_work['necesidad_base'] > 0)
            ]
            
            if df_work.empty:
                logger.error("No hay productos que requieran producción")
                return None

            ## Ordenar por días de cobertura
            df_work = df_work.sort_values('dias_cobertura', ascending=True)
            
            ## Inicializar producción
            produccion_optima = pd.Series(0.0, index=df_work.index)
            horas_totales = 0.0
            horas_restantes = horas_disponibles
            
            ## Calcular producción por producto
            for idx in df_work.index:
                if horas_restantes < 2:
                    break
                    
                producto = df_work.loc[idx]
                necesidad = producto['necesidad_base']
                
                ## Calcular horas necesarias
                horas_necesarias = necesidad / producto['Cj/H']
                
                ## Ajustar a mínimo 2 horas de producción
                if 0 < horas_necesarias < 2:
                    horas_necesarias = 2
                    necesidad = horas_necesarias * producto['Cj/H']
                
                ## Ajustar si excede horas disponibles
                if horas_necesarias > horas_restantes:
                    necesidad = horas_restantes * producto['Cj/H']
                    horas_necesarias = horas_restantes
                
                produccion_optima[idx] = round(necesidad, 0)
                horas_totales += horas_necesarias
                horas_restantes -= horas_necesarias
                
                logger.info(f"Producto {producto['COD_ART']}: {necesidad:.0f} cajas, {horas_necesarias:.2f} horas")

            logger.info(f"Horas totales necesarias: {horas_totales:.2f}")
            logger.info(f"Horas restantes: {horas_restantes:.2f}")
            return produccion_optima

        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            logger.error("Traza completa:", exc_info=True)
            return None

    def aplicar_filtros(self, df, fecha_inicio):
        ## Filtros para selección de productos
        df_filtrado = df.copy()
        df_filtrado = df_filtrado[df_filtrado['Vta -60'] > self.MIN_VENTA_60D]
        df_filtrado = df_filtrado[df_filtrado['Cj/H'] > self.MIN_TASA_PRODUCCION]
        
        ## Filtro de primera orden de fabricación
        df_filtrado['1ª OF'] = df_filtrado['1ª OF'].replace('(en blanco)', pd.NA)
        mask_of = ~df_filtrado['1ª OF'].notna() | (
            pd.to_datetime(df_filtrado['1ª OF'], format='%d/%m/%Y') > pd.to_datetime(fecha_inicio)
        )
        df_filtrado = df_filtrado[mask_of]
        
        return df_filtrado.drop_duplicates(subset=['COD_ART'], keep='last')

    def calcular_demanda(self, df):
        """
        Calcula la demanda media basándose en la lógica definida.
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

    def cargar_datos(self, carpeta="Dataset", fecha_inicio=None):
        try:
            ## Validar fecha de inicio
            if not fecha_inicio:
                raise ValueError("Debe proporcionar una fecha de inicio.")
            
            ## Normalizar fecha para búsqueda de archivo
            fecha_normalizada = datetime.strptime(
                fecha_inicio.replace("/", "-"), 
                "%d-%m-%Y"
            ).strftime("%d-%m-%y")
            
            ## Encontrar archivo correspondiente
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
            
            ## Cargar archivo CSV
            df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1', skiprows=4)
            df = df[df['COD_ART'].notna()]
            
            ## Columnas numéricas a convertir
            columnas_numericas = ['Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 
                                'Stock Externo', 'M_Vta -15 AA', 'M_Vta +15 AA', 
                                'Vta -60']
            
            ## Convertir columnas a numérico
            for col in columnas_numericas:
                df[col] = pd.to_numeric(
                    df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True),
                    errors='coerce'
                ).fillna(0)
            
            ## Calcular stock total
            df['stock_total'] = (
                df['Disponible'].fillna(0) + 
                df['Calidad'].fillna(0) + 
                df['Stock Externo'].fillna(0)
            )
            
            ## Aplicar filtros y calcular demanda
            df = self.aplicar_filtros(df, fecha_inicio)
            df = self.calcular_demanda(df)
            
            logger.info(f"Datos cargados: {len(df)} productos")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None

    def generar_plan_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles, fecha_inicio=None, usar_simplex=False):
        try:
            logger.info("=== Iniciando generación de plan ===")
            logger.info(f"{'Usando Simplex' if usar_simplex else 'Usando método propio'}")
            logger.info(f"Columnas disponibles: {df.columns.tolist()}")
            logger.info(f"Datos iniciales:\n{df[['COD_ART', 'stock_total', 'demanda_media', 'Cj/H']].head()}")
            
            if isinstance(fecha_inicio, str):
                fecha_inicio = datetime.strptime(fecha_inicio, "%d-%m-%Y")
            
            plan = df.copy()
            ## Seleccionar método de optimización
            produccion_optima = (
                self.simplex(plan, horas_disponibles, dias_planificacion, dias_no_habiles)
                if usar_simplex else
                self.optimizar_produccion(plan, horas_disponibles, dias_planificacion, dias_no_habiles)
            )
            
            ## Validar producción óptima
            if produccion_optima is None:
                logger.warning("No se obtuvo producción óptima")
                return None, None, None
                
            if produccion_optima.empty:
                logger.warning("Producción óptima está vacía")
                return None, None, None
                
            logger.info(f"Producción óptima calculada:\n{produccion_optima}")
            
            ## Asignar cajas y horas de producción
            plan.loc[produccion_optima.index, 'cajas_a_producir'] = produccion_optima
            plan['horas_necesarias'] = (plan['cajas_a_producir'] / plan['Cj/H']).round(3)
            
            logger.info(f"Plan antes de filtrar:\n{plan[plan['cajas_a_producir'] > 0][['COD_ART', 'cajas_a_producir', 'horas_necesarias']]}")
            
            ## Filtrar productos con producción
            plan = plan[plan['cajas_a_producir'] > 0].copy()
            
            if plan.empty:
                logger.warning("Plan está vacío después de filtrar cajas a producir")
                return None, None, None
                
            ## Calcular fechas de planificación
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
                
            ## Generar reporte detallado de producción
            print(f"\nPLAN DE PRODUCCIÓN ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
            print("-" * 80)
            print(f"Total productos: {len(plan)}")
            print(f"Total cajas: {plan['cajas_a_producir'].sum():.0f}")
            print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
            print("\nDetalle por producto:")
            
            columnas = ['COD_ART', 'NOM_ART', 'stock_total', 'demanda_media', 
                      'cajas_a_producir', 'horas_necesarias']
            print(plan[columnas].to_string(index=False))
            
            ## Opcional: Guardar reporte en archivo CSV
            ruta_reporte = f"plan_produccion_{fecha_inicio.strftime('%Y%m%d')}.csv"
            plan[columnas].to_csv(ruta_reporte, index=False)
            logger.info(f"Reporte guardado en {ruta_reporte}")
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")

def main():
    try:
        fecha_inicio = input("Ingrese fecha inicio (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        dias_planificacion = int(input("Ingrese días de planificación: "))
        dias_no_habiles = int(input("Ingrese días no hábiles en el periodo: "))
        horas_mantenimiento = int(input("Ingrese horas de mantenimiento: "))
        usar_simplex = input("¿Usar método Simplex? (s/n): ").strip().lower() == 's'
        
        if dias_planificacion <= 0:
            raise ValueError("Los días de planificación deben ser positivos")
        if dias_no_habiles < 0 or dias_no_habiles >= dias_planificacion:
            raise ValueError("Días no hábiles inválidos")
        if horas_mantenimiento < 0:
            raise ValueError("Las horas de mantenimiento no pueden ser negativas")
            
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        logger.info(f"Horas disponibles para producción: {horas_disponibles}")
        
        planificador = PlanificadorProduccion()
        
        logger.info("Cargando datos...")
        df = planificador.cargar_datos(fecha_inicio=fecha_inicio)
        if df is None:
            logger.error("Error al cargar datos")
            return
            
        logger.info(f"Datos cargados: {len(df)} productos")
        
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
            
        logger.info("Generando reporte...")
        planificador.generar_reporte(plan, fecha_inicio_dt, fecha_fin)
        
        logger.info("Proceso completado exitosamente")
        
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}")
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()