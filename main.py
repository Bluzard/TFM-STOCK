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
        self.DIAS_STOCK_SEGURIDAD = 3
        self.HORAS_MIN_LOTE = 2
        self.DIAS_MAX_COBERTURA = 15
        self.MIN_VENTA_60D = 0
        self.MIN_TASA_PRODUCCION = 0
        self.UMBRAL_VARIACION = 0.20
        self.COBERTURA_MAX_FINAL = 60  # Máxima cobertura después de producción

    def aplicar_filtros(self, df, fecha_inicio):
        """Aplica los filtros según documento de requerimientos"""
        df_filtrado = df.copy()
        total_inicial = len(df_filtrado)
        
        # 1. Productos activos (ventas en últimos 60 días)
        df_filtrado = df_filtrado[df_filtrado['Vta -60'] > self.MIN_VENTA_60D]
        logger.info(f"Productos activos: {len(df_filtrado)} de {total_inicial}")
        
        # 2. Tasa de producción válida
        df_filtrado = df_filtrado[df_filtrado['Cj/H'] > self.MIN_TASA_PRODUCCION]
        logger.info(f"Con tasa producción válida: {len(df_filtrado)}")
        
        # 3. Verificar OF previas
        df_filtrado['1ª OF'] = df_filtrado['1ª OF'].replace('(en blanco)', pd.NA)
        mask_of = ~df_filtrado['1ª OF'].notna() | (
            pd.to_datetime(df_filtrado['1ª OF'], format='%d/%m/%Y') > pd.to_datetime(fecha_inicio)
        )
        df_filtrado = df_filtrado[mask_of]
        logger.info(f"Sin OF previa: {len(df_filtrado)}")
        
        return df_filtrado.drop_duplicates(subset=['COD_ART'], keep='last')

    def calcular_demanda(self, df):
        """Calcula demanda considerando variación año anterior"""
        df['variacion_aa'] = abs(1 - df['M_Vta +15 AA'] / df['M_Vta -15 AA']).fillna(0)
        df['demanda_diaria'] = np.where(
            df['variacion_aa'] > self.UMBRAL_VARIACION,
            df['M_Vta -15'] * (1 - df['M_Vta +15 AA'] / df['M_Vta -15 AA']),
            df['M_Vta -15'] / 15
        )
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
            logger.info(f"Cargando archivo: {archivo}")
            
            with open(ruta_archivo, 'r', encoding='latin1') as f:
                fecha_archivo = f.readline().split(';')[1].strip().split()[0]
                logger.info(f"Fecha del archivo: {fecha_archivo}")
            
            df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1', skiprows=4)
            df = df[df['COD_ART'].notna()]
            logger.info(f"Columnas disponibles: {df.columns.tolist()}")
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None

    def optimizar_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles):
        try:
            if len(df) == 0:
                logger.error("No hay productos para optimizar")
                return None
                
            logger.info(f"Optimizando {len(df)} productos")
            logger.info(f"Horas disponibles: {horas_disponibles}")
            logger.info(f"Días planificación: {dias_planificacion}")
            logger.info(f"Días no hábiles: {dias_no_habiles}")
            n_productos = len(df)
            dias_habiles = dias_planificacion - dias_no_habiles
            
            # Vector de coeficientes objetivo (minimizar producción total)
            c = np.ones(n_productos)
            
            # Matriz de restricciones
            A_ub = []
            b_ub = []
            
            # 1. Restricción de horas totales
            horas_por_caja = np.array([1/producto['Cj/H'] if producto['Cj/H'] > 0 else 1e6 for _, producto in df.iterrows()])
            A_ub.append(horas_por_caja)
            b_ub.append(horas_disponibles)
            
            # 2. Stock mínimo de seguridad
            stock_actual = df['stock_total'].values
            demanda_diaria = df['demanda_diaria'].replace([np.inf, -np.inf], 0).fillna(0).values
            demanda_total = demanda_diaria * (self.DIAS_STOCK_SEGURIDAD + dias_habiles)
            
            for i in range(n_productos):
                fila = np.zeros(n_productos)
                fila[i] = -1
                A_ub.append(fila)
                b_ub.append(float(stock_actual[i] - demanda_total[i]))
            
            # 3. Cobertura máxima
            for i in range(n_productos):
                if demanda_diaria[i] > 0:
                    fila = np.zeros(n_productos)
                    fila[i] = 1
                    A_ub.append(fila)
                    cobertura_max = demanda_diaria[i] * self.COBERTURA_MAX_FINAL
                    b_ub.append(float(cobertura_max - stock_actual[i]))
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Límites de producción
            limites = []
            for _, producto in df.iterrows():
                cajas_min = max(0, self.HORAS_MIN_LOTE * producto['Cj/H'])
                limites.append((float(cajas_min), None))
            
            resultado = linprog(
                c, 
                A_ub=A_ub, 
                b_ub=b_ub, 
                bounds=limites, 
                method='highs'
            )
            
            if resultado.success:
                logger.info(f"Optimización exitosa: {resultado.message}")
                return resultado.x
            else:
                logger.error(f"Optimización fallida: {resultado.message}")
                return None
            
            return resultado.x if resultado.success else None
            
        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            return None

    def generar_plan_produccion(self, df, horas_disponibles, dias_planificacion, dias_no_habiles, fecha_inicio=None):
        try:
            if isinstance(fecha_inicio, str):
                fecha_inicio = datetime.strptime(fecha_inicio, "%d-%m-%Y")
            
            plan = df.copy()
            produccion_optima = self.optimizar_produccion(
                plan, horas_disponibles, dias_planificacion, dias_no_habiles
            )
            
            if produccion_optima is None:
                return None, None, None
            
            plan['cajas_a_producir'] = produccion_optima
            plan['horas_necesarias'] = plan['cajas_a_producir'] / plan['Cj/H']
            plan = plan[plan['cajas_a_producir'] > 0].copy()
            
            plan['cobertura_final'] = (
                (plan['stock_total'] + plan['cajas_a_producir']) / 
                plan['demanda_diaria']
            )
            
            if not plan.empty:
                fecha_fin = fecha_inicio + timedelta(days=dias_planificacion)
                nombre_archivo = f"plan_produccion_{fecha_inicio.strftime('%Y%m%d')}_{fecha_fin.strftime('%Y%m%d')}.csv"
                
                with open(nombre_archivo, 'w', encoding='latin1') as f:
                    f.write(f"Fecha Inicio;{fecha_inicio.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(f"Fecha Fin;{fecha_fin.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(f"Días no hábiles;{dias_no_habiles};;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                
                columnas = ['COD_ART', 'NOM_ART', 'stock_total', 'demanda_diaria', 
                           'cajas_a_producir', 'horas_necesarias', 'cobertura_final']
                plan[columnas].to_csv(nombre_archivo, sep=';', index=False, mode='a', 
                                    encoding='latin1')
                logger.info(f"Plan guardado en: {nombre_archivo}")
                
                return plan, fecha_inicio, fecha_fin
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error generando plan: {str(e)}")
            return None, None, None

    def generar_reporte(self, plan, fecha_inicio, fecha_fin):
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
                    'cajas_a_producir', 'horas_necesarias', 'cobertura_final']
        print(plan[columnas].to_string(index=False))

def main():
    try:
        fecha_inicio = input("Ingrese fecha inicio (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        dias_planificacion = int(input("Ingrese días de planificación: "))
        dias_no_habiles = int(input("Ingrese días no hábiles en el periodo: "))
        horas_mantenimiento = int(input("Ingrese horas de mantenimiento: "))
        
        planificador = PlanificadorProduccion()
        
        df = planificador.cargar_datos(fecha_inicio=fecha_inicio)
        if df is None:
            logger.error("Error al cargar datos")
            return
        
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        plan, fecha_inicio_dt, fecha_fin = planificador.generar_plan_produccion(
            df, horas_disponibles, dias_planificacion, dias_no_habiles, fecha_inicio
        )
        
        if plan is None:
            logger.error("No se pudo generar el plan")
            return
        
        planificador.generar_reporte(plan, fecha_inicio_dt, fecha_fin)
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()