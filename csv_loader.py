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
    Lee y procesa el archivo de pedidos pendientes con el formato específico.
    
    Args:
        fecha_dataset: Fecha del dataset (datetime o string en formato DD-MM-YYYY)
        
    Returns:
        DataFrame con los pedidos pendientes procesados, o None si hay un error
    """
    try:
        import os
        import pandas as pd
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Formatear la fecha para el nombre de archivo
        if isinstance(fecha_dataset, datetime):
            fecha_str = fecha_dataset.strftime('%d-%m-%y')
        else:
            # Si ya es string, normalizar formato
            fecha_str = fecha_dataset
            
            # Si el formato es DD-MM-YYYY, convertir a DD-MM-YY
            if len(fecha_str.split('-')[2]) == 4:
                fecha_dt = datetime.strptime(fecha_str, '%d-%m-%Y')
                fecha_str = fecha_dt.strftime('%d-%m-%y')
        
        # Nombre del archivo
        archivo_pedidos = os.path.join('Pedidos', f'Pedidos pendientes {fecha_str}.csv')
       
        # Verificar si existe el archivo
        if not os.path.exists(archivo_pedidos):
            logger.warning(f"No se encontró el archivo: {archivo_pedidos}")
            return None
            
        logger.info(f"Leyendo archivo de pedidos pendientes: {archivo_pedidos}")
        
        # Para este formato específico, leer saltando la primera fila
        try:
            # Leer el archivo con pandas empezando desde la segunda fila
            df_pedidos = pd.read_csv(archivo_pedidos, sep=';', encoding='latin1', skiprows=1)
            
            # Asegurarse de que 'COD_ART' está en las columnas
            if 'COD_ART' not in df_pedidos.columns and df_pedidos.shape[1] > 0:
                # Renombrar la primera columna a COD_ART
                df_pedidos = df_pedidos.rename(columns={df_pedidos.columns[0]: 'COD_ART'})
                logger.info(f"Renombrando primera columna a 'COD_ART'")
            
            # Si 'NOM_ART' no está, renombrar la segunda columna
            if 'NOM_ART' not in df_pedidos.columns and df_pedidos.shape[1] > 1:
                df_pedidos = df_pedidos.rename(columns={df_pedidos.columns[1]: 'NOM_ART'})
                logger.info(f"Renombrando segunda columna a 'NOM_ART'")
            
            # Identificar columnas de fechas (columnas que no son COD_ART ni NOM_ART)
            columnas_fechas = [col for col in df_pedidos.columns 
                              if col not in ['COD_ART', 'NOM_ART'] and '/' in col]
            
            # Convertir columnas de fechas a numéricas
            for col in columnas_fechas:
                df_pedidos[col] = pd.to_numeric(df_pedidos[col], errors='coerce').fillna(0)
            
            # Asegurar que COD_ART es string
            df_pedidos['COD_ART'] = df_pedidos['COD_ART'].astype(str)
            
            logger.info(f"Pedidos pendientes procesados: {len(df_pedidos)} productos")
            logger.info(f"Columnas de fechas identificadas: {len(columnas_fechas)}")
            
            return df_pedidos
            
        except Exception as e:
            logger.error(f"Error procesando archivo con formato estándar: {str(e)}")
            
            # Intento alternativo: leer manualmente
            try:
                with open(archivo_pedidos, 'r', encoding='latin1') as file:
                    lineas = file.readlines()
                
                # Ignorar la primera línea
                encabezados = lineas[1].strip().split(';')
                
                # Buscar índices de columnas importantes
                idx_cod_art = 0  # Primera columna
                idx_nom_art = 1  # Segunda columna
                
                # Crear diccionario para almacenar datos
                data = {
                    'COD_ART': [],
                    'NOM_ART': []
                }
                
                # Añadir columnas de fechas (a partir de la tercera columna)
                fechas = encabezados[2:]
                for fecha in fechas:
                    if fecha.strip():  # Solo si no está vacía
                        data[fecha] = []
                
                # Procesar líneas de datos (a partir de la tercera línea)
                for i in range(2, len(lineas)):
                    if not lineas[i].strip():
                        continue
                        
                    campos = lineas[i].strip().split(';')
                    
                    # Añadir código y nombre
                    if len(campos) > idx_cod_art:
                        data['COD_ART'].append(campos[idx_cod_art])
                    else:
                        continue  # Saltar línea si no hay código
                        
                    if len(campos) > idx_nom_art:
                        data['NOM_ART'].append(campos[idx_nom_art])
                    else:
                        data['NOM_ART'].append("")
                    
                    # Procesar fechas
                    for j, fecha in enumerate(fechas):
                        if not fecha.strip():
                            continue
                            
                        idx = j + 2  # Offset para las columnas de fechas
                        if idx < len(campos) and campos[idx].strip():
                            try:
                                # Convertir a número (los pedidos son valores negativos)
                                valor = float(campos[idx].replace(',', '.'))
                                data[fecha].append(valor)
                            except:
                                data[fecha].append(0)
                        else:
                            data[fecha].append(0)
                
                # Crear DataFrame
                df_pedidos = pd.DataFrame(data)
                
                # Asegurar que COD_ART es string
                df_pedidos['COD_ART'] = df_pedidos['COD_ART'].astype(str)
                
                logger.info(f"Pedidos pendientes procesados manualmente: {len(df_pedidos)} productos")
                logger.info(f"Columnas identificadas: {list(data.keys())}")
                
                return df_pedidos
                
            except Exception as e:
                logger.error(f"Error en procesamiento manual: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
    
    except Exception as e:
        logger.error(f"Error general en leer_pedidos_pendientes: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None