import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlanificadorProduccion:
    def __init__(self):
        self.DIAS_STOCK_SEGURIDAD = 3
        self.DIAS_MAX_COBERTURA = 15
        self.MIN_VENTA_60D = 0
        self.MIN_TASA_PRODUCCION = 0
        self.UMBRAL_VARIACION = 0.20
        self.COBERTURA_MAX_FINAL = 60

    def optimizar_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles):
        try:
            dias_habiles = dias_planificacion - dias_no_habiles
            logger.info(f"Días hábiles para planificación: {dias_habiles}")
            
            df_work = df.copy()
            df_work['cobertura_actual'] = (df_work['stock_total'] / df_work['demanda_diaria']).round(2)
            df_work['dias_cobertura'] = (df_work['stock_total'] / df_work['demanda_diaria']).round(2)
            df_work['demanda_periodo'] = (df_work['demanda_diaria'] * dias_habiles).round(0)
            df_work['stock_seguridad'] = (df_work['demanda_diaria'] * self.DIAS_STOCK_SEGURIDAD).round(0)
            
            df_work['necesidad_base'] = df_work.apply(lambda row: max(
                row['stock_seguridad'] - row['stock_total'],
                row['demanda_periodo'] - (row['stock_total'] - row['stock_seguridad'])
            ), axis=1)
            
            # Mantener índice original
            df_work = df_work[
                (df_work['cobertura_actual'] <= self.COBERTURA_MAX_FINAL) &
                (df_work['necesidad_base'] > 0)
            ]
            
            if df_work.empty:
                logger.error("No hay productos que requieran producción")
                return None

            df_work = df_work.sort_values('dias_cobertura', ascending=True)
            
            produccion_optima = pd.Series(0.0, index=df_work.index)
            horas_totales = 0.0
            horas_restantes = horas_disponibles
            
            for idx in df_work.index:
                if horas_restantes < 2:
                    break
                    
                producto = df_work.loc[idx]
                necesidad = producto['necesidad_base']
                
                horas_necesarias = necesidad / producto['Cj/H']
                
                if 0 < horas_necesarias < 2:
                    horas_necesarias = 2
                    necesidad = horas_necesarias * producto['Cj/H']
                
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
        df_filtrado = df.copy()
        df_filtrado = df_filtrado[df_filtrado['Vta -60'] > self.MIN_VENTA_60D]
        df_filtrado = df_filtrado[df_filtrado['Cj/H'] > self.MIN_TASA_PRODUCCION]
        
        df_filtrado['1ª OF'] = df_filtrado['1ª OF'].replace('(en blanco)', pd.NA)
        mask_of = ~df_filtrado['1ª OF'].notna() | (
            pd.to_datetime(df_filtrado['1ª OF'], format='%d/%m/%Y') > pd.to_datetime(fecha_inicio)
        )
        df_filtrado = df_filtrado[mask_of]
        
        return df_filtrado.drop_duplicates(subset=['COD_ART'], keep='last')

    def calcular_demanda(self, df):
        df['M_Vta -15 AA'] = df['M_Vta -15 AA'].replace(0, 1)
        df['variacion_aa'] = np.abs(1 - df['M_Vta +15 AA'] / df['M_Vta -15 AA']).fillna(0)
        
        df['demanda_diaria'] = np.where(
            df['variacion_aa'] > self.UMBRAL_VARIACION,
            df['M_Vta -15'] * (1 + df['variacion_aa']),
            df['M_Vta -15'] / 15
        )
        
        df['demanda_diaria'] = np.maximum(df['demanda_diaria'], 0)
        return df

    def cargar_datos(self, carpeta="Dataset", fecha_inicio=None):
        try:
            if not fecha_inicio:
                raise ValueError("Debe proporcionar una fecha de inicio.")
            
            fecha_normalizada = datetime.strptime(
                fecha_inicio.replace("/", "-"), 
                "%d-%m-%Y"
            ).strftime("%d-%m-%y")
            
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
            
            df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1', skiprows=4)
            df = df[df['COD_ART'].notna()]
            
            columnas_numericas = ['Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 
                                'Stock Externo', 'M_Vta -15 AA', 'M_Vta +15 AA', 
                                'Vta -60']
            
            for col in columnas_numericas:
                df[col] = pd.to_numeric(
                    df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True),
                    errors='coerce'
                )
            
            df['stock_total'] = (
                df['Disponible'].fillna(0) + 
                df['Calidad'].fillna(0) + 
                df['Stock Externo'].fillna(0)
            )
            
            df = self.aplicar_filtros(df, fecha_inicio)
            df = self.calcular_demanda(df)
            
            logger.info(f"Datos cargados: {len(df)} productos")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None

    def generar_plan_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles, fecha_inicio=None):
        try:
            logger.info("=== Iniciando generación de plan ===")
            logger.info(f"Columnas disponibles: {df.columns.tolist()}")
            logger.info(f"Datos iniciales:\n{df[['COD_ART', 'stock_total', 'demanda_diaria', 'Cj/H']].head()}")
            
            if isinstance(fecha_inicio, str):
                fecha_inicio = datetime.strptime(fecha_inicio, "%d-%m-%Y")
            
            plan = df.copy()
            produccion_optima = self.optimizar_produccion(
                plan, horas_disponibles, dias_planificacion, dias_no_habiles
            )
            
            if produccion_optima is None:
                logger.warning("No se obtuvo producción óptima")
                return None, None, None
                
            if produccion_optima.empty:
                logger.warning("Producción óptima está vacía")
                return None, None, None
                
            logger.info(f"Producción óptima calculada:\n{produccion_optima}")
            
            plan.loc[produccion_optima.index, 'cajas_a_producir'] = produccion_optima
            plan['horas_necesarias'] = (plan['cajas_a_producir'] / plan['Cj/H']).round(3)
            
            logger.info(f"Plan antes de filtrar:\n{plan[plan['cajas_a_producir'] > 0][['COD_ART', 'cajas_a_producir', 'horas_necesarias']]}")
            
            plan = plan[plan['cajas_a_producir'] > 0].copy()
            
            if plan.empty:
                logger.warning("Plan está vacío después de filtrar cajas a producir")
                return None, None, None
                
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
            print(f"Total cajas: {plan['cajas_a_producir'].sum():.0f}")
            print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
            print("\nDetalle por producto:")
            
            columnas = ['COD_ART', 'NOM_ART', 'stock_total', 'demanda_diaria', 
                      'cajas_a_producir', 'horas_necesarias']
            print(plan[columnas].to_string(index=False))
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")

def main():
    try:
        fecha_inicio = input("Ingrese fecha inicio (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        dias_planificacion = int(input("Ingrese días de planificación: "))
        dias_no_habiles = int(input("Ingrese días no hábiles en el periodo: "))
        horas_mantenimiento = int(input("Ingrese horas de mantenimiento: "))
        
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
            fecha_inicio
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