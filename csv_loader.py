# CARGA DE ARCHIVOS, INICIALIZACIÓN Y CLASS PRODUCTO

import logging
from datetime import datetime
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Producto:
    def __init__(self, cod_art, nom_art, cod_gru, cajas_hora, disponible, calidad, 
                 stock_externo, pedido, primera_of, of, vta_60, vta_15, m_vta_15, 
                 vta_15_aa, m_vta_15_aa, vta_15_mas_aa, m_vta_15_mas_aa, orden_planificacion=''):
        # Datos básicos
        self.cod_art = cod_art
        self.nom_art = nom_art
        self.cod_gru = cod_gru
        self.of = self._convertir_float(of)
        self.cajas_hora = self._convertir_float(cajas_hora)

        # Añadir_of_reales (80% de la estimación)
        if self.of is not None:
            self.of_reales = self.of * 0.8   # Hay una "merma productiva" del 20% con respecto a la estimación.
        else:
            self.of_reales = None
            logger.error(f"Producto {self.cod_art}: of_reales es None")

        # Añadir cajas_hora_reales (90% de cajas_hora)
        if self.cajas_hora is not None:
            self.cajas_hora_reales = self.cajas_hora * 0.9   # Hay una "merma productiva" del 10% con respecto a la estimación.
        else:
            self.cajas_hora_reales = None
            logger.error(f"Producto {self.cod_art}: cajas_hora es None")  
        
        # Stocks
        self.disponible = self._convertir_float(disponible)
        self.calidad = self._convertir_float(calidad)
        self.stock_externo = self._convertir_float(stock_externo) if stock_externo != '(en blanco)' else 0
        
        # Órdenes y pedidos
        self.pedido = self._convertir_float(pedido)
        self.primera_of = primera_of
        
        # Datos de ventas
        self.vta_60 = self._convertir_float(vta_60)
        self.vta_15 = self._convertir_float(vta_15)
        self.m_vta_15 = self._convertir_float(m_vta_15)
        self.vta_15_aa = self._convertir_float(vta_15_aa)
        self.m_vta_15_aa = self._convertir_float(m_vta_15_aa)
        self.vta_15_mas_aa = self._convertir_float(vta_15_mas_aa)
        self.m_vta_15_mas_aa = self._convertir_float(m_vta_15_mas_aa)
        
        # Campo de orden de planificación
        self.orden_planificacion = 0  # Inicialmente 0, se asignará en la función ordenar_planificacion
        
        # Campos calculados (inicialmente 0)
        self.demanda_media = 0
        self.stock_inicial = 0
        self.cobertura_inicial = 0
        self.stock_seguridad = 0
        self.cajas_a_producir = 0
        self.horas_necesarias = 0
        self.cobertura_final_est = 0
        self.cobertura_final_plan = 0

    
    def _convertir_float(self, valor):
        if isinstance(valor, (int, float)):
            return float(valor)
        if isinstance(valor, str):
            if valor.strip() == '' or valor == '(en blanco)':
                return 0.0
            try:
                valor = valor.replace(".", "")
                return float(valor.replace(',', '.'))
            except ValueError:
                return 0.0
        return 0.0

def leer_dataset(nombre_archivo):
    try:
        ruta_completa = os.path.join('Dataset', nombre_archivo)
        productos = []
        with open(ruta_completa, 'r', encoding='latin1') as file:
            for _ in range(5):
                next(file)
            
            for linea in file:
                if not linea.strip() or linea.startswith('Total general'):
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) >= 15:
                    producto = Producto(
                        cod_art=campos[0],          # COD_ART
                        nom_art=campos[1],          # NOM_ART
                        cod_gru=campos[2],          # COD_GRU
                        cajas_hora=campos[3],       # Cj/H
                        disponible=campos[4],       # Disponible
                        calidad=campos[5],          # Calidad
                        stock_externo=campos[6],    # Stock Externo
                        pedido=campos[7],          # Pedido
                        primera_of=campos[8],       # 1ª OF
                        of=campos[9],               # OF
                        vta_60=campos[10],         # Vta -60
                        vta_15=campos[11],         # Vta -15
                        m_vta_15=campos[12],       # M_Vta -15
                        vta_15_aa=campos[15],      # Vta -15 AA
                        m_vta_15_aa=campos[16],    # M_Vta -15 AA
                        vta_15_mas_aa=campos[17],  # Vta +15 AA
                        m_vta_15_mas_aa=campos[18] # M_Vta +15 AA
                    )
                    productos.append(producto)
        return productos
    except Exception as e:
        logger.error(f"Error leyendo dataset: {str(e)}")
        return None

def leer_indicaciones_articulos():
    try:
        productos_info = {}
        with open('Indicaciones articulos.csv', 'r', encoding='latin1') as file:
            header = file.readline().strip().split(';')
            try:
                idx_info = header.index('Info extra')
                idx_cod = header.index('COD_ART')
                idx_orden = header.index('ORDEN PLANIFICACION')
                
                # Usar 'cj/palet' en lugar de 'CAJAS_PALET'
                idx_cajas_palet = header.index('cj/palet') 
            except ValueError:
                logger.error("No se encontraron las columnas requeridas en el archivo de indicaciones")
                return {}, set()
            
            productos_omitir = set()
            for linea in file:
                if not linea.strip():
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) > max(idx_info, idx_cod, idx_orden, idx_cajas_palet):
                    info_extra = campos[idx_info].strip()
                    cod_art = campos[idx_cod].strip()
                    orden = campos[idx_orden].strip() if idx_orden < len(campos) else ''
                    
                    # Convertir cajas por palet a entero, con valor por defecto
                    try:
                        cajas_palet = int(campos[idx_cajas_palet]) if campos[idx_cajas_palet].strip() else 40
                    except ValueError:
                        cajas_palet = 40
                    
                    if info_extra in ['DESCATALOGADO']: # Al incluir la mejora de pedidos pendientes solo descartamos los DESCATALOGADOS
                        productos_omitir.add(cod_art)
                    
                    productos_info[cod_art] = {
                        'info_extra': info_extra,
                        'orden_planificacion': orden,
                        'cajas_palet': cajas_palet
                    }
        
        logger.info(f"Información de productos cargada: {len(productos_info)}")
        return productos_info, productos_omitir
    except Exception as e:
        logger.error(f"Error leyendo indicaciones: {str(e)}")
        return {}, set()

def verificar_archivo_existe(nombre_archivo):
    try:
        if not os.path.exists(nombre_archivo):
            print(f"\n⚠️  ADVERTENCIA: No se encuentra el archivo '{nombre_archivo}'")
            print("Verifique que el archivo existe en el directorio actual con ese nombre exacto.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error verificando archivo: {str(e)}")
        return False
def leer_pedidos_pendientes(fecha_dataset):
    try:
        fecha_dataset_str = fecha_dataset.strftime('%d-%m-%y')
        archivo_pedidos = f'Pedidos pendientes {fecha_dataset_str}.csv'
        
        # Leer el archivo saltando las dos primeras filas (encabezados no necesarios)
        df_pedidos = pd.read_csv(archivo_pedidos, sep=';', encoding='latin1',skiprows=1)  # Saltar primera fila)
        
        # Renombrar las columnas usando la primera fila real de datos
        # Las fechas estarán en formato DD/MM/YYYY
        columnas_fechas = [col for col in df_pedidos.columns if '/' in str(col)]
        
        # Mantener solo las columnas necesarias: COD_ART y fechas
        columnas_mantener = ['COD_ART'] + columnas_fechas
        df_pedidos = df_pedidos[columnas_mantener]
        
        # Asegurar que COD_ART sea string
        df_pedidos['COD_ART'] = df_pedidos['COD_ART'].astype(str)
        
        # Convertir columnas de fechas a números y llenar NaN con 0
        for col in columnas_fechas:
            df_pedidos[col] = pd.to_numeric(df_pedidos[col], errors='coerce').fillna(0)
        
        # Verificar que tenemos datos
        if df_pedidos.empty:
            logger.error("No se encontraron datos en el archivo de pedidos")
            return None
            
        logger.info(f"Lectura exitosa de pedidos. Dimensiones: {df_pedidos.shape}")
        
        return df_pedidos

    except Exception as e:
        logger.error(f"Error leyendo pedidos pendientes: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return None