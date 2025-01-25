import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionPlanner:
    def __init__(self, safety_stock_days=3, min_production_hours=2):
        self.safety_stock_days = safety_stock_days
        self.min_production_hours = min_production_hours

    def load_data(self, folder_path="Dataset", fecha_inicio=None):
        """    
        Carga el archivo CSV que corresponde a la fecha de inicio proporcionada.
        Args:
        folder_path (str): Ruta del directorio con los archivos CSV.
        fecha_inicio (str): Fecha de inicio en formato 'DD-MM-YYYY' o 'DD/MM/YYYY'.    
        Returns:        pd.DataFrame: Datos procesados, o None si no se encuentra el archivo.
        """
        try:
            # Normalizar la fecha para buscar en el nombre del archivo
            if not fecha_inicio:
                raise ValueError("Debe proporcionar una fecha de inicio.")
            
            # Convertir la fecha al formato esperado en los nombres de archivo
            try:
                fecha_normalizada = datetime.strptime(fecha_inicio.replace("/", "-"), "%d-%m-%Y").strftime("%d-%m-%y")
            except ValueError as e:
                raise ValueError("El formato de la fecha debe ser 'DD-MM-YYYY' o 'DD/MM/YYYY'.")
            
            # Buscar el archivo correspondiente
            archivo_encontrado = None
            for file in os.listdir(folder_path):
                if file.endswith('.csv') and fecha_normalizada in file:
                    archivo_encontrado = file
                    break
            
            if not archivo_encontrado:
                raise FileNotFoundError(f"No se encontró un archivo con la fecha: {fecha_normalizada}.")
            
            logger.info(f"Cargando archivo: {archivo_encontrado}")
            file_path = os.path.join(folder_path, archivo_encontrado)
            
            # Leer la fecha del archivo
            with open(file_path, 'r', encoding='latin1') as f:
                date_str = f.readline().split(';')[1].strip().split()[0]
                logger.info(f"Fecha del archivo: {date_str}")
            
            # Leer el CSV
            df = pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=4)
            df = df[df['COD_ART'].notna()]  # Eliminar filas sin código
            
            # Convertir columnas numéricas
            for col in ['Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15 AA', 'M_Vta +15 AA']:
                df[col] = pd.to_numeric(df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True), errors='coerce')
            
            # Eliminar duplicados
            df = df.drop_duplicates(subset=['COD_ART'], keep='last')
            logger.info(f"Datos cargados: {len(df)} productos únicos")
            return df
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None, None, None

    def generate_production_plan(self, df, available_hours, planning_days,fecha_inicio=None):
        try:
            # Revisar formato de fecha_inicio
            if isinstance(fecha_inicio, str):
                fecha_inicio = datetime.strptime(fecha_inicio, "%d-%m-%Y")

            # Copiar solo las columnas necesarias
            plan = df.copy()[['COD_ART', 'NOM_ART', 'Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15 AA', 'M_Vta +15 AA' ]]
            
            # Reemplazar comas en la columna NOM_ART
            plan['NOM_ART'] = plan['NOM_ART'].str.replace(',', '?', regex=False) 

            # Calcular campos básicos
            plan['cajas_hora'] = plan['Cj/H']
            plan['stock_total'] = plan['Disponible'].fillna(0) + plan['Calidad'].fillna(0) + plan['Stock Externo'].fillna(0)
            
            #Completar demanda diaria con fórmula experta
            plan['var_ex_AA_abs'] = abs(1 - plan['M_Vta +15 AA'] / plan['M_Vta -15 AA']).fillna(0)

            # Calcular demanda_diaria usando np.where para aplicar la lógica basada en var_ex_AA_abs
            plan['demanda_diaria'] = np.where(
                plan['var_ex_AA_abs'] > 0.2,
                plan['M_Vta -15'] * (1 - plan['M_Vta +15 AA'] / plan['M_Vta -15 AA']),
                plan['M_Vta -15']
            )
            # Filtrar productos válidos
            plan = plan[plan['cajas_hora'] > 0].copy()
            logger.info(f"Planificando producción para {len(plan)} productos")
            
            # Calcular necesidades
            plan['cajas_necesarias'] = np.maximum(
                0, 
                ((self.safety_stock_days + planning_days) * plan['demanda_diaria']) - plan['stock_total']
            )
            
            # Aplicar lote mínimo
            plan['cajas_a_producir'] = np.maximum(
                plan['cajas_necesarias'],
                self.min_production_hours * plan['cajas_hora']
            )
            
            # Calcular horas
            plan['horas_necesarias'] = plan['cajas_a_producir'] / plan['cajas_hora']
            
            # Filtrar y ordenar
            plan = plan[plan['cajas_a_producir'] > 0].sort_values('stock_total')
            
            # Ajustar al tiempo disponible
            horas_totales = 0
            productos_final = []
            
            for _, row in plan.iterrows():
                if horas_totales + row['horas_necesarias'] <= available_hours:
                    productos_final.append(row)
                    horas_totales += row['horas_necesarias']
            
            plan_final = pd.DataFrame(productos_final)
            
            if not plan_final.empty:
                fecha_fin = fecha_inicio + timedelta(days=planning_days)
                
                # Guardar plan con fechas
                filename = f"plan_produccion_{fecha_inicio.strftime('%Y%m%d')}_{fecha_fin.strftime('%Y%m%d')}.csv"
                
                # Crear encabezado con fechas
                with open(filename, 'w', encoding='latin1') as f:
                    f.write(f"Fecha Inicio;{fecha_inicio.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(f"Fecha Fin;{fecha_fin.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                
                # Guardar el plan después del encabezado
                plan_final.to_csv(filename, sep=';', index=False, mode='a', encoding='latin1')
                logger.info(f"Plan guardado en: {filename}")
                
                return plan_final, fecha_inicio, fecha_fin
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error en plan de producción: {str(e)}")
            return None, None, None

    def generate_report(self, plan, fecha_inicio, fecha_fin):
        if plan is None or len(plan) == 0:
            logger.error("No hay plan de producción para reportar")
            return
            
        print(f"\nPLAN DE PRODUCCIÓN ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        print("-" * 80)
        print(f"Total productos: {len(plan)}")
        print(f"Total cajas: {plan['cajas_a_producir'].sum():.0f}")
        print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
        print("\nDetalle por producto:")
        print(plan[['COD_ART', 'NOM_ART', 'stock_total', 'cajas_a_producir', 'horas_necesarias']].to_string(index=False))

def main():
    try:
        # Solicitar parámetros al usuario
        fecha_inicio = input("Ingrese la fecha de inicio (DD-MM-YYYY o DD/MM/YYYY): ").strip()
        planning_days = int(input("Ingrese los días de planificación: "))
        mantenimiento_hs = int(input("Ingrese las horas de mantenimiento en el período a planificar: "))
        no_laborable_days = int(input("Ingrese los días no laborables en el período a planificar (inlcuyendo domingos): "))
        
        planner = ProductionPlanner()
        df = planner.load_data(fecha_inicio=fecha_inicio)
        
        if df is None or df.empty:
            logger.error("No se cargaron datos. Verifique que el archivo y la fecha sean correctos.")
            return
        
        available_hours = 24 * (planning_days - no_laborable_days) - mantenimiento_hs
        plan, fecha_inicio_dt, fecha_fin = planner.generate_production_plan(df, available_hours, planning_days, fecha_inicio)
        
        if plan is None or plan.empty:
            logger.error("No se generó un plan de producción. Verifique los datos y parámetros.")
            return
        
        planner.generate_report(plan, fecha_inicio_dt, fecha_fin)
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()