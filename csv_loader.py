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
        
        # Ajuste de producción real (85% de la teórica)
        self.cajas_hora_reales = self.cajas_hora * 0.85
        self.of_reales = self.of * 0.85
        
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
            except ValueError as e:
                logger.error(f"No se encontraron todas las columnas requeridas en el archivo de indicaciones: {str(e)}")
                # Intentar con valores por defecto si no se encuentran todas las columnas
                idx_info = header.index('Info extra') if 'Info extra' in header else -1
                idx_cod = header.index('COD_ART') if 'COD_ART' in header else 0
                idx_orden = header.index('ORDEN PLANIFICACION') if 'ORDEN PLANIFICACION' in header else -1
                idx_cajas_palet = header.index('cj/palet') if 'cj/palet' in header else -1
            
            productos_omitir = set()
            for linea in file:
                if not linea.strip():
                    continue
                    
                campos = linea.strip().split(';')
                
                # Verificar que hay suficientes campos
                if len(campos) <= max(idx for idx in [idx_cod, idx_info, idx_orden, idx_cajas_palet] if idx >= 0):
                    continue
                
                cod_art = campos[idx_cod].strip()
                
                # Extraer información si es posible
                info_extra = campos[idx_info].strip() if idx_info >= 0 and idx_info < len(campos) else ''
                orden = campos[idx_orden].strip() if idx_orden >= 0 and idx_orden < len(campos) else ''
                
                # Convertir cajas por palet a entero, con valor por defecto
                try:
                    cajas_palet = int(campos[idx_cajas_palet]) if (idx_cajas_palet >= 0 and 
                                                                 idx_cajas_palet < len(campos) and 
                                                                 campos[idx_cajas_palet].strip()) else 40
                except ValueError:
                    cajas_palet = 40
                
                if info_extra in ['DESCATALOGADO', 'PEDIDO']:
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
    """
    Lee y procesa el archivo de pedidos pendientes con formato específico "Pedidos pendientes DD-MM-YY".
    
    Args:
        fecha_dataset: Fecha del dataset (datetime o string en formato DD-MM-YY)
        
    Returns:
        DataFrame con los pedidos pendientes procesados, o None si hay un error
    """
    try:
        # Formatear la fecha para el nombre de archivo
        if isinstance(fecha_dataset, datetime):
            fecha_str = fecha_dataset.strftime('%d-%m-%y')
        else:
            # Si ya es string, normalizar formato
            fecha_str = fecha_dataset
            
            # Intentar convertir si el formato es DD-MM-YYYY
            if len(fecha_str.split('-')[2]) == 4:  # Año con 4 dígitos
                fecha_dt = datetime.strptime(fecha_str, '%d-%m-%Y')
                fecha_str = fecha_dt.strftime('%d-%m-%y')
        
        # Nombre de archivo esperado (formato exacto)
        archivo_pedidos = f'Pedidos pendientes {fecha_str}'
        
        # Comprobar si existe con diferentes extensiones
        for ext in ['.csv', '.txt']:
            if os.path.exists(archivo_pedidos + ext):
                archivo_pedidos = archivo_pedidos + ext
                break
        
        # Si no encontramos el archivo, buscar alternativas
        if not os.path.exists(archivo_pedidos):
            # Intentar con otros formatos de fecha
            alternativas = []
            
            # Probar sin guiones
            fecha_sin_guiones = fecha_str.replace('-', '')
            alternativas.append(f'Pedidos pendientes {fecha_sin_guiones}')
            
            # Probar con diferentes separadores
            for sep in ['_', ' ']:
                alternativas.append(f'Pedidos{sep}pendientes{sep}{fecha_str}')
                alternativas.append(f'Pedidos{sep}pendientes{sep}{fecha_sin_guiones}')
            
            # Verificar cada alternativa
            for alt in alternativas:
                for ext in ['.csv', '.txt']:
                    if os.path.exists(alt + ext):
                        archivo_pedidos = alt + ext
                        break
                if os.path.exists(archivo_pedidos):
                    break
            
            # Si todavía no existe, buscar cualquier archivo de pedidos
            if not os.path.exists(archivo_pedidos):
                for archivo in os.listdir('.'):
                    if archivo.startswith('Pedidos pendientes') and (archivo.endswith('.csv') or archivo.endswith('.txt')):
                        archivo_pedidos = archivo
                        break
        
        # Verificar si se encontró un archivo
        if not os.path.exists(archivo_pedidos):
            logger.warning(f"No se encontró el archivo de pedidos pendientes para la fecha {fecha_str}")
            return None
            
        logger.info(f"Leyendo archivo de pedidos pendientes: {archivo_pedidos}")
        
        # Detectar qué formato tiene el archivo
        with open(archivo_pedidos, 'r', encoding='latin1') as f:
            primera_linea = f.readline().strip()
            
        # Determinar separador
        if '\t' in primera_linea:
            separador = '\t'
        elif ';' in primera_linea:
            separador = ';'
        else:
            separador = ','
            
        # Leer según el formato detectado
        try:
            df_pedidos = pd.read_csv(archivo_pedidos, sep=separador, encoding='latin1')
        except Exception as e:
            logger.warning(f"Error leyendo con separador '{separador}': {str(e)}")
            try:
                # Intentar con otro separador
                otro_separador = '\t' if separador != '\t' else ';'
                df_pedidos = pd.read_csv(archivo_pedidos, sep=otro_separador, encoding='latin1')
            except Exception as e2:
                logger.error(f"No se pudo leer el archivo con ningún separador estándar: {str(e2)}")
                # Leer como texto plano
                with open(archivo_pedidos, 'r', encoding='latin1') as f:
                    lineas = f.readlines()
                
                # Encontrar las columnas de fechas (formato DD/MM/YYYY o similar)
                headers = lineas[0].strip().split(separador)
                
                # Buscar columna de COD_ART o similar
                cod_art_col = None
                for i, h in enumerate(headers):
                    if 'COD' in h or 'ART' in h or 'codigo' in h.lower() or 'código' in h.lower():
                        cod_art_col = i
                        break
                
                if cod_art_col is None:
                    logger.error("No se pudo identificar la columna de código de artículo")
                    return None
                
                # Crear datos manualmente
                data = {'COD_ART': []}
                fecha_cols = []
                
                # Identificar columnas de fechas
                for i, h in enumerate(headers):
                    if i != cod_art_col and not h.startswith('Suma') and not h.startswith('Total'):
                        try:
                            # Intentar interpretar como fecha
                            datetime.strptime(h, '%d/%m/%Y')
                            fecha_cols.append(i)
                            data[h] = []
                        except:
                            try:
                                # Otro formato de fecha
                                datetime.strptime(h, '%d/%m/%y')
                                fecha_cols.append(i)
                                data[h] = []
                            except:
                                # No es fecha, pero podría ser una columna importante
                                if h.strip():
                                    data[h] = []
                
                # Procesar datos
                for linea in lineas[1:]:
                    if not linea.strip():
                        continue
                    
                    campos = linea.strip().split(separador)
                    if len(campos) <= cod_art_col:
                        continue
                    
                    # Añadir código de artículo
                    data['COD_ART'].append(campos[cod_art_col])
                    
                    # Añadir valores de fechas (cantidades)
                    for i, col in enumerate(headers):
                        if i in fecha_cols and col in data:
                            if i < len(campos):
                                try:
                                    # Convertir a número negativo (pedidos son negativos)
                                    valor = float(campos[i])
                                    data[col].append(valor)
                                except:
                                    data[col].append(0)
                            else:
                                data[col].append(0)
                
                # Crear DataFrame
                df_pedidos = pd.DataFrame(data)
        
        # Verificar columnas y ajustar si es necesario
        if 'COD_ART' not in df_pedidos.columns:
            # Buscar columna alternativa
            for col in df_pedidos.columns:
                if 'cod' in col.lower() or 'art' in col.lower() or 'código' in col.lower() or 'codigo' in col.lower():
                    df_pedidos.rename(columns={col: 'COD_ART'}, inplace=True)
                    break
            else:
                # Si no se encuentra, usar la primera columna no numérica
                for col in df_pedidos.columns:
                    if df_pedidos[col].dtype == 'object':
                        df_pedidos.rename(columns={col: 'COD_ART'}, inplace=True)
                        break
        
        # Convertir valores de fechas a números negativos (pedidos)
        for col in df_pedidos.columns:
            if col != 'COD_ART' and col != 'NOM_ART':
                try:
                    df_pedidos[col] = pd.to_numeric(df_pedidos[col], errors='coerce').fillna(0)
                except:
                    pass
        
        # Asegurar que COD_ART sea string
        df_pedidos['COD_ART'] = df_pedidos['COD_ART'].astype(str)
        
        # Mostrar información del resultado
        logger.info(f"Pedidos pendientes cargados: {len(df_pedidos)} productos")
        logger.info(f"Columnas de fechas: {[col for col in df_pedidos.columns if col not in ['COD_ART', 'NOM_ART']]}")
        
        return df_pedidos
        
    except Exception as e:
        logger.error(f"Error leyendo pedidos pendientes: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None