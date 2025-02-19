import logging
from typing import NamedTuple, Optional, Dict
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_column(header: list, possible_names: list) -> int:
    """
    Busca una columna en el header por múltiples nombres posibles
    
    Args:
        header: Lista de nombres de columnas
        possible_names: Lista de posibles nombres para la columna
    
    Returns:
        Índice de la columna encontrada o -1 si no se encuentra
    """
    for name in possible_names:
        try:
            return header.index(name)
        except ValueError:
            continue
    return -1

class ConfiguracionArticulo(NamedTuple):
    """Estructura para almacenar la configuración de artículos"""
    info_extra: str
    orden_planificacion: str = ''  # INICIO, FINAL o vacío
    prioridad_semanal: int = 5    # 1-5, siendo 1 la más alta
    grupo_alergenicos: bool = False   # True si contiene alérgenos
    cajas_palet: int = 0          # Cantidad de cajas por palet
    tiempo_cambio: int = 30       # Tiempo en minutos para cambio
    capacidad_almacen: int = 1000 # Capacidad máxima de almacenamiento

class Producto:
    def __init__(self, cod_art, nom_art, cod_gru, cajas_hora, disponible, calidad, 
                 stock_externo, pedido, primera_of, of, vta_60, vta_15, m_vta_15, 
                 vta_15_aa, m_vta_15_aa, vta_15_mas_aa, m_vta_15_mas_aa, 
                 configuracion: Optional[ConfiguracionArticulo] = None):
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
        
        # Configuración adicional
        self.configuracion = configuracion or ConfiguracionArticulo(
            info_extra='',
            orden_planificacion='',
            prioridad_semanal=5,
            grupo_alergenicos=False,
            cajas_palet=0,
            tiempo_cambio=30,
            capacidad_almacen=1000
        )
        
        # Campos calculados (inicialmente 0)
        self.demanda_media = 0
        self.demanda_provisoria = 0
        self.stock_inicial = 0
        self.cobertura_inicial = 0
        self.demanda_periodo = 0
        self.stock_seguridad = 0
        self.cajas_a_producir = 0
        self.horas_necesarias = 0
        self.cobertura_final_est = 0
        self.cobertura_final_plan = 0

    def _convertir_float(self, valor):
        """Convierte un valor a float, manejando diferentes formatos"""
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

def leer_indicaciones_articulos() -> Dict[str, ConfiguracionArticulo]:
    """Lee y procesa el archivo de indicaciones de artículos"""
    try:
        configuraciones = {}
        with open('Indicaciones articulos.csv', 'r', encoding='latin1') as file:
            header = file.readline().strip().split(';')
            
            # Buscar índices de columnas
            idx_cod = find_column(header, ['COD_ART', 'Código'])
            idx_info = find_column(header, ['Info extra', 'Información'])
            idx_orden = find_column(header, ['ORDEN PLANIFICACIÓN', 'Orden'])
            idx_prioridad = find_column(header, ['Prioridad Semanal', 'Prioridad'])
            idx_alerg = find_column(header, ['Grupo Alergénicos', 'Alérgenos'])
            idx_cajas = find_column(header, ['Cajas/Palet', 'Cajas por Palet'])
            idx_tiempo = find_column(header, ['Tiempo Cambio', 'Setup Time'])
            
            # Verificar columnas requeridas
            if idx_cod == -1 or idx_info == -1:
                raise ValueError("Faltan columnas requeridas (COD_ART, Info extra)")
            
            for linea in file:
                if not linea.strip():
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) <= max(idx_cod, idx_info):
                    continue
                
                cod_art = campos[idx_cod].strip()
                info_extra = campos[idx_info].strip()
                
                # Obtener valores con defaults
                orden = campos[idx_orden].strip() if idx_orden != -1 and idx_orden < len(campos) else ''
                prioridad = int(campos[idx_prioridad]) if idx_prioridad != -1 and idx_prioridad < len(campos) and campos[idx_prioridad].strip().isdigit() else 5
                alerg = campos[idx_alerg].strip().upper() == 'SI' if idx_alerg != -1 and idx_alerg < len(campos) else False
                cajas = int(campos[idx_cajas]) if idx_cajas != -1 and idx_cajas < len(campos) and campos[idx_cajas].strip().isdigit() else 0
                tiempo = int(campos[idx_tiempo]) if idx_tiempo != -1 and idx_tiempo < len(campos) and campos[idx_tiempo].strip().isdigit() else 30
                
                configuraciones[cod_art] = ConfiguracionArticulo(
                    info_extra=info_extra,
                    orden_planificacion=orden,
                    prioridad_semanal=prioridad,
                    grupo_alergenicos=alerg,
                    cajas_palet=cajas,
                    tiempo_cambio=tiempo
                )
        
        logger.info(f"Configuraciones cargadas: {len(configuraciones)}")
        return configuraciones
    except Exception as e:
        logger.error(f"Error en lectura de indicaciones: {str(e)}")
        return {}

def leer_dataset(nombre_archivo):
    """Lee el archivo de dataset y crea objetos Producto"""
    try:
        # Cargar configuraciones
        configuraciones = leer_indicaciones_articulos()
        
        productos = []
        with open(nombre_archivo, 'r', encoding='latin1') as file:
            # Saltar encabezado
            for _ in range(5):
                next(file)
            
            for linea in file:
                if not linea.strip() or linea.startswith('Total general'):
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) >= 15:
                    cod_art = campos[0]
                    config = configuraciones.get(cod_art, ConfiguracionArticulo(info_extra=''))
                    
                    producto = Producto(
                        cod_art=cod_art,
                        nom_art=campos[1],
                        cod_gru=campos[2],
                        cajas_hora=campos[3],
                        disponible=campos[4],
                        calidad=campos[5],
                        stock_externo=campos[6],
                        pedido=campos[7],
                        primera_of=campos[8],
                        of=campos[9],
                        vta_60=campos[10],
                        vta_15=campos[11],
                        m_vta_15=campos[12],
                        vta_15_aa=campos[15],
                        m_vta_15_aa=campos[16],
                        vta_15_mas_aa=campos[17],
                        m_vta_15_mas_aa=campos[18],
                        configuracion=config
                    )
                    productos.append(producto)
        
        logger.info(f"Dataset cargado: {len(productos)} productos")
        return productos
    except Exception as e:
        logger.error(f"Error leyendo dataset: {str(e)}")
        return None

def verificar_dataset_existe(nombre_archivo: str) -> bool:
    """Verifica si existe el archivo de dataset"""
    try:
        if not os.path.exists(nombre_archivo):
            logger.warning(f"No se encuentra el archivo: {nombre_archivo}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error verificando dataset: {str(e)}")
        return False