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
        self.orden_planificacion = orden_planificacion
        
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
                    
                    if info_extra in ['DESCATALOGADO']:
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

def verificar_dataset_existe(nombre_archivo):
    try:
        if not os.path.exists(nombre_archivo):
            print(f"\n⚠️  ADVERTENCIA: No se encuentra el archivo '{nombre_archivo}'")
            print("Verifique que el archivo existe en el directorio actual con ese nombre exacto.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error verificando dataset: {str(e)}")
        return False
def leer_pedidos_pendientes(fecha_dataset):
    try:
        fecha_dataset_str = fecha_dataset.strftime('%d-%m-%y')
        archivo_pedidos = f'Pedidos pendientes {fecha_dataset_str}.csv'
        
        # Verificar si el archivo existe
        if not os.path.exists(archivo_pedidos):
            logger.warning(f"No se encontró el archivo de pedidos: {archivo_pedidos}")
            return None

        # Leer el archivo de pedidos
        df_pedidos = pd.read_csv(archivo_pedidos, sep=';', encoding='latin1')
        
        # Convertir fechas numéricas
        fechas_pedidos = [col for col in df_pedidos.columns if col[0:2] in ['09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']]
        df_pedidos[fechas_pedidos] = df_pedidos[fechas_pedidos].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Convertir COD_ART a string
        df_pedidos['COD_ART'] = df_pedidos['COD_ART'].astype(str)
        
        return df_pedidos

    except Exception as e:
        logger.error(f"Error leyendo pedidos pendientes: {str(e)}")
        return None