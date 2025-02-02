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
        self.COBERTURA_MIN = 3 #Cobertura stock de seguridad mínima (días)
        self.COBERTURA_MAX = 60  ## Máxima cobertura de stock permitida e(días)
        self.DEMANDA_60D_MIN = 0  ## Umbral mínimo de ventas en 60 días (Cj)
        self.TASA_PRODUCCION_MIN = 0  ## Tasa mínima de producción (Cj/h)
        self.UMBRAL_VARIACION = 0.20  ## Umbral de variación para ajuste de demanda (%)
        

    def simplex(self, df, horas_disponibles, dias_planificacion):
        try:
            #dias_habiles = dias_planificacion - dias_no_habiles #demanda media considera todos los días del año
            df_work = df.copy()
            
            ## Filtrar productos con demanda válida
            df_work = df_work[
                (df_work['Vta -60'] > self.DEMANDA_60D_MIN) &
                (df_work['demanda_media'] > 0)
            ].copy()
            
            ## Calcular cobertura inicial y stock de seguridad
            df_work['demanda_periodo'] = (df_work['demanda_media'] * dias_planificacion).round(0)
            df_work['stock_seguridad'] = (df_work['demanda_media'] * self.COBERTURA_MIN).round(0)

            df_work['cobertura_inicial'] = (df_work['stock_inicial'] / df_work['demanda_media']).round(1)
            df_work['cobertura_final_est'] = ((df_work['stock_inicial'] - df_work['demanda_periodo'])/ df_work['demanda_media']).round(1)
            
            
            ## Filtrar productos que necesitan producción
            df_work = df_work[
                (df_work['cobertura_final_est'] < self.COBERTURA_MAX) 
            #    (df_work['cobertura_final_est'] >= self.COBERTURA_MIN)
            ]
            logger.info(f"Lista entrada al Simplex con {len(df_work)} filas:\n{df_work[['COD_ART', 'stock_inicial', 'demanda_media', 'Cj/H','cobertura_inicial', 'cobertura_final_est']]}")
            
            if df_work.empty:
                logger.info("No hay productos que requieran producción")
                return None
                
            n_productos = len(df_work)
            
            #

            ## Función objetivo: minimizar diferencia entre stock actual y stock de seguridad

            #c = abs(df_work['stock_seguridad'].values - df_work['Cj/H'].values*df_work['horas_produccion'])    ## Priorizar productos con mayor demanda
            c = -(df_work['demanda_media'].values) # Priorizar productos con mayor demanda
            
            ## Restricción de horas disponibles
            A_ub = np.zeros((1, n_productos))
            A_ub[0] = 1/df_work['Cj/H'].values
            b_ub = [horas_disponibles]
            
            ## Restricción de stock mínimo
            A_lb = np.identity(n_productos)
            b_lb = df_work['stock_seguridad'].values - df_work['stock_inicial'].values
            
            A = np.vstack([A_ub, A_lb])
            b = np.concatenate([b_ub, b_lb])

            #Restricción horas disponibles con igualdad
            A_eq = np.ones((1, n_productos))  # Una fila con coeficientes 1 para todos los productos
            b_eq = horas_disponibles

            ## Chequeo previo a uso de simplex
            print("A_ub:\n", A)
            print("b_ub:\n", b)
            print("c:\n", c)
            
            ## Resolver con Simplex
            result = linprog(
                c,
                A_ub=A, 
                b_ub=b,
                A_eq=A_eq,
                b_eq=b_eq,
                method='highs'
            )
            
            if not result.success:
                logger.error(f"No se encontró solución óptima: {result.message}")
                return None
                
            ## Obtener producción óptima
            produccion_optima = pd.Series(np.round(result.x, 0), index=df_work.index)
            
            ## Calcular horas utilizadas
            horas_por_producto = produccion_optima / df_work['Cj/H']
            logger.info(f"Horas totales necesarias: {horas_por_producto.sum():.2f}")
            logger.info(f"Horas restantes: {horas_disponibles - horas_por_producto.sum():.2f}")
            
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
            df_work['cobertura_inicial'] = (df_work['stock_inicial'] / df_work['demanda_media']).round(2)
            df_work['dias_cobertura'] = (df_work['stock_inicial'] / df_work['demanda_media']).round(2)
            df_work['demanda_periodo'] = (df_work['demanda_media'] * dias_habiles).round(0)
            df_work['stock_seguridad'] = (df_work['demanda_media'] * self.COBERTURA_MIN).round(0)
            
            ## Calcular necesidad base de producción
            df_work['necesidad_base'] = df_work.apply(lambda row: max(
                row['stock_seguridad'] - row['stock_inicial'],
                row['demanda_periodo'] - (row['stock_inicial'] - row['stock_seguridad'])
            ), axis=1)
            
            ## Filtrar productos que necesitan producción
            df_work = df_work[
                (df_work['cobertura_inicial'] <= self.COBERTURA_MAX) &
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

    def aplicar_filtros(self, df):
        ## Filtros para selección de productos
        df_filtrado = df.copy()
        df_filtrado = df_filtrado[df_filtrado['Vta -60'] > self.DEMANDA_60D_MIN]
        df_filtrado = df_filtrado[df_filtrado['Cj/H'] > self.TASA_PRODUCCION_MIN] #no parece filtrar nada

        """"  NO se debe filtrar por contenido en esta columna
        ## Filtro de primera orden de fabricación
        df_filtrado['1ª OF'] = df_filtrado['1ª OF'].replace('(en blanco)', pd.NA)
        mask_of = ~df_filtrado['1ª OF'].notna() | (
            pd.to_datetime(df_filtrado['1ª OF'], format='%d/%m/%Y') > pd.to_datetime(fecha_inicio)
        )
        df_filtrado = df_filtrado[mask_of]
        """
        
        
        return df_filtrado.drop_duplicates(subset=['COD_ART'], keep='last')

    def calcular_demanda(self,df):
        """
        Calcula la demanda media basándose en la lógica definida en la función original.
        
        Parámetros:
        df (pd.DataFrame): DataFrame con los datos necesarios para el cálculo.
        umbral_variacion (float): Umbral de variación para aplicar ajuste de demanda.
        
        Retorna:
        pd.DataFrame: DataFrame con la columna 'demanda_media' actualizada.
        """
        # Asegurar que no haya valores nulos
        df['M_Vta -15'] = df['M_Vta -15'].fillna(0)
        df['M_Vta -15 AA'] = df['M_Vta -15 AA'].fillna(0)  # Se corrige aquí
        df['M_Vta +15 AA'] = df['M_Vta +15 AA'].fillna(0)

        # Calcular variación interanual (sin valor absoluto para permitir aumento/disminución)
        with np.errstate(divide='ignore', invalid='ignore'):  # Evita warnings por división por 0
            df['variacion_aa'] = np.abs(1 - df['M_Vta +15 AA'] / df['M_Vta -15 AA'])
        df['variacion_aa'] = df['variacion_aa'].fillna(0)

        # Calcular demanda media
        df['demanda_media'] = np.where(
            (df['M_Vta -15 AA'] == 0) | (df['M_Vta +15 AA'] == 0),  # Condición de Excel
            df['M_Vta -15'],
            np.where(
                (df['variacion_aa'] > self.UMBRAL_VARIACION & df['variacion_aa'] <1),
                df['M_Vta -15'] * (df['M_Vta +15 AA'] / df['M_Vta -15 AA']),
                df['M_Vta -15']
            )
        )

        # Asegurar que la demanda media no sea negativa
        df['demanda_media'] = np.maximum(df['demanda_media'], 0)

        return df

    def cargar_datos(self, carpeta="Dataset", fecha_dataset=None, fecha_inicio=None):
        try:
            ## Validar fecha dataset
            if not fecha_dataset:
                raise ValueError("Debe proporcionar una fecha del dataset a utilizar.")
            
            ## Normalizar fecha para búsqueda de archivo
            fecha_normalizada = datetime.strptime(
                fecha_dataset.replace("/", "-"), 
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
                                'Vta -60', 'OF']
            
            ## Convertir columnas a numérico
            for col in columnas_numericas:
                df[col] = pd.to_numeric(
                    df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True),
                    errors='coerce'
                ).fillna(0)
            

            fecha_dataset = pd.to_datetime(fecha_dataset, format='%d-%m-%Y')
            fecha_inicio = pd.to_datetime(fecha_inicio, format='%d-%m-%Y')

            ## Calcular incremento de stock antes de inicio de programación con OF=Orden de fabricación

            # Convertir '1ª OF' a datetime y manejar valores faltantes
            df['1ª OF'] = pd.to_datetime(df['1ª OF'].replace('(en blanco)', pd.NaT), format='%d/%m/%Y', errors='coerce')

            # Crear la nueva columna 'Disponible_Inicial' con los valores actualizados según la condición
            if not df['1ª OF'].isna().all():
                # Aplicar la condición de rango para la fecha
                mask = (df['1ª OF'] >= fecha_dataset) & (df['1ª OF'] < fecha_inicio)
                # Crear la nueva columna basándose en 'Disponible' y 'OF', respetando la condición
                df['Disponible_Inicial'] = df['Disponible']  # Copiar los valores originales
                df.loc[mask, 'Disponible_Inicial'] = df.loc[mask, 'Disponible'].fillna(0) + df.loc[mask, 'OF'].fillna(0)
            else:
                # Si todas las fechas son NaT, simplemente copiar 'Disponible'
                df['Disponible_Inicial'] = df['Disponible']

            ## Aplicar filtros y calcular demanda
            df = self.aplicar_filtros(df)
            df = self.calcular_demanda(df)

            df['demanda_prov'] = (fecha_inicio - fecha_dataset).days*df['demanda_media'].fillna(0)
            
            ## Calcular stock inicial
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
            ## Seleccionar método de optimización
            produccion_optima = (
                self.simplex(plan, horas_disponibles, dias_planificacion)
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
                
            #logger.info(f"Producción óptima calculada:\n{produccion_optima}") #se comenta porque luego aparece en salida las cajas a producir
            
            ## Asignar cajas y horas de producción
            plan.loc[produccion_optima.index, 'cajas_a_producir'] = produccion_optima
            plan['horas_necesarias'] = (plan['cajas_a_producir'] / plan['Cj/H']).round(1)
            
            #logger.info(f"Plan antes de filtrar:\n{plan[plan['cajas_a_producir'] > 0][['COD_ART', 'cajas_a_producir', 'horas_necesarias']]}") #info que se ve en salida final también, y ya está filtrado acá
            
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
            #print(f"Total cajas: {plan['cajas_a_producir'].sum():.0f}")
            print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
            print("\nDetalle por producto:")
            
            columnas = ['COD_ART', 'NOM_ART', 'Disponible_Inicial', 'Calidad', 'Stock Externo', 'stock_inicial', 'demanda_media', 
                      'cajas_a_producir', 'horas_necesarias']
            print(plan[columnas].to_string(index=False))
            
            ## Opcional: Guardar reporte en archivo CSV
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
        
        if dias_planificacion <= 0:
            raise ValueError("Los días de planificación deben ser positivos")
        if dias_no_habiles < 0 or dias_no_habiles >= dias_planificacion:
            raise ValueError("Días no hábiles inválidos")
        if horas_mantenimiento < 0:
            raise ValueError("Las horas de mantenimiento no pueden ser negativas")
        if fecha_inicio < fecha_dataset:
            raise ValueError("La fecha de inicio debe ser mayor o igual a la fecha de dataset")
            
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        logger.info(f"Horas disponibles para producción: {horas_disponibles}")
        
        planificador = PlanificadorProduccion()
        
        logger.info("Cargando datos...")
        df = planificador.cargar_datos(fecha_inicio=fecha_inicio, fecha_dataset=fecha_dataset)
        if df is None:
            logger.error("Error al cargar datos")
            return
            
        logger.info(f"Datos cargados y filtrados: {len(df)} productos")
        
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