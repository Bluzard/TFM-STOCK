import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import linprog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Producto:
    def __init__(self, cod_art, nom_art, cod_gru, cajas_hora, disponible, calidad, 
                 stock_externo, pedido, primera_of, of, vta_60, vta_15, m_vta_15, 
                 vta_15_aa, m_vta_15_aa, vta_15_mas_aa, m_vta_15_mas_aa):
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
        
        # Campos calculados (inicialmente 0)
        self.demanda_media = 0
        self.stock_inicial = 0
        self.cobertura_inicial = 0
        self.stock_seguridad = 0
        self.cajas_a_producir = 0
        self.horas_necesarias = 0

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
        productos = []
        with open(nombre_archivo, 'r', encoding='latin1') as file:
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
        productos_omitir = set()
        with open('Indicaciones articulos.csv', 'r', encoding='latin1') as file:
            header = file.readline().strip().split(';')
            try:
                idx_info = header.index('Info extra')
                idx_cod = header.index('COD_ART')
            except ValueError:
                logger.error("No se encontraron las columnas requeridas en el archivo de indicaciones")
                return set()
            
            for linea in file:
                if not linea.strip():
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) > max(idx_info, idx_cod):
                    info_extra = campos[idx_info].strip()
                    cod_art = campos[idx_cod].strip()
                    
                    if info_extra in ['DESCATALOGADO', 'PEDIDO']:
                        productos_omitir.add(cod_art)
        
        logger.info(f"Productos a omitir cargados: {len(productos_omitir)}")
        return productos_omitir
    except FileNotFoundError:
        logger.error("No se encontró el archivo 'Indicaciones articulos.csv'")
        return set()
    except Exception as e:
        logger.error(f"Error leyendo indicaciones de artículos: {str(e)}")
        return set()

def calcular_formulas(productos, fecha_inicio, fecha_dataset, dias_planificacion, dias_no_habiles, horas_mantenimiento):
    """Calcula todas las fórmulas para cada producto y aplica filtros"""
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        productos_omitir = leer_indicaciones_articulos()
        productos_validos = []
        
        for producto in productos:
            # 2. Cálculo de demanda media
            if producto.m_vta_15_aa > 0:
                variacion_aa = abs(1 - (producto.vta_15_mas_aa / producto.vta_15_aa))
                if variacion_aa > 0.20 and variacion_aa < 1:
                    producto.demanda_media = producto.m_vta_15 * (producto.vta_15_mas_aa / producto.vta_15_aa)
                else:
                    producto.demanda_media = producto.m_vta_15
            else:
                producto.demanda_media = producto.m_vta_15

            # 3. Demanda provisoria
            dias_diff = (fecha_inicio - fecha_dataset).days
            producto.demanda_provisoria = producto.demanda_media * dias_diff

            # 4. Actualizar Disponible
            if producto.primera_of != '(en blanco)':
                of_date = datetime.strptime(producto.primera_of, '%d/%m/%Y')
                if of_date >= fecha_dataset and of_date < fecha_inicio:
                    producto.disponible = producto.disponible + producto.of
            
            # 5. Stock Inicial
            producto.stock_inicial = producto.disponible + producto.calidad + producto.stock_externo - producto.demanda_provisoria
            
            ## ----------------- ALERTA STOCK INICIAL NEGATIVO ----------------- ##    
            ## ----------------- CORREGIR PLANIFICACION (ADELANTAR PLANIFICACION)----------------- ##
            
            if producto.stock_inicial < 0:
                producto.stock_inicial = 0
                logger.warning(f"Producto {producto.cod_art}: Stock Inicial negativo. Se ajustó a 0.")

            # 6. Cobertura Inicial  
            if producto.demanda_media > 0:
                producto.cobertura_inicial = producto.stock_inicial / producto.demanda_media
            else:
                producto.cobertura_inicial = 'NO VALIDO'
            
            # 7. Demanda Periodo
            producto.demanda_periodo = producto.demanda_media * dias_planificacion
            
            # 8. Stock de Seguridad (3 días)
            producto.stock_seguridad = producto.demanda_media * 3       

            # 9. Cobertura Final Estimada
            if producto.demanda_media > 0:
                 producto.cobertura_final_est = (producto.stock_inicial - producto.demanda_periodo) / producto.demanda_media
            else:
                producto.cobertura_final_est = 'NO VALIDO'

            # Aplicar filtros
            if (producto.cod_art not in productos_omitir and
                producto.vta_60 > 0 and 
                producto.cajas_hora > 0 and 
                producto.demanda_media > 0 and 
                producto.cobertura_inicial != 'NO VALIDO' and 
                producto.cobertura_final_est != 'NO VALIDO'):
                productos_validos.append(producto)
                    
        logger.info(f"Productos válidos tras filtros: {len(productos_validos)} de {len(productos)}")
        return productos_validos, horas_disponibles
        
    except Exception as e:
        logger.error(f"Error en cálculos: {str(e)}")
        return None, None

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion, dias_cobertura_base):
    """Aplica el método Simplex para optimizar la producción"""
    try:
        n_productos = len(productos_validos)
        cobertura_minima = dias_cobertura_base + dias_planificacion

        # Función objetivo
        coeficientes = []
        for producto in productos_validos:
            if producto.demanda_media > 0:
                prioridad = max(0, 1/producto.cobertura_inicial)
            else:
                prioridad = 0
            coeficientes.append(-prioridad)

        # Restricciones
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1 / producto.cajas_hora for producto in productos_validos]
        b_eq = [horas_disponibles]

        A_ub = []
        b_ub = []
        for i, producto in enumerate(productos_validos):
            if producto.demanda_media > 0:
                row = [0] * n_productos
                row[i] = -1
                A_ub.append(row)
                stock_min = (producto.demanda_media * cobertura_minima) - producto.stock_inicial
                b_ub.append(-stock_min)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Bounds
        bounds = []
        for producto in productos_validos:
            if producto.demanda_media > 0 and producto.cobertura_inicial < 30:
                min_cajas = 2 * producto.cajas_hora
                max_cajas = min(
                    horas_disponibles * producto.cajas_hora,
                    producto.demanda_media * 60 - producto.stock_inicial
                )
                max_cajas = max(min_cajas, max_cajas)
            else:
                min_cajas = 0
                max_cajas = 0
            bounds.append((min_cajas, max_cajas))

        # Optimización
        result = linprog(
            c=coeficientes,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            horas_producidas = 0
            
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora
                horas_producidas += producto.horas_necesarias
                
                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
            
            logger.info(f"Optimización exitosa - Horas planificadas: {horas_producidas:.2f}/{horas_disponibles:.2f}")
            return productos_validos
        else:
            logger.error(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Error en Simplex: {str(e)}")
        return None

def exportar_resultados(productos_optimizados, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    try:
        datos = []
        for producto in productos_optimizados:
            if producto.horas_necesarias > 0:
                cobertura_minima = dias_cobertura_base + dias_planificacion
                datos.append({
                    'COD_ART': producto.cod_art,
                    'NOM_ART': producto.nom_art,
                    'Demanda_Media': round(producto.demanda_media, 2),
                    'Stock_Inicial': round(producto.stock_inicial, 2),
                    'Cajas_a_Producir': producto.cajas_a_producir,
                    'Horas_Necesarias': round(producto.horas_necesarias, 2),
                    'Cobertura_Inicial': round(producto.cobertura_inicial, 2),
                    'Cobertura_Final': round(producto.cobertura_final_plan, 2),
                    'Cobertura_Final_Est': round(producto.cobertura_final_est, 2),
                    'Desviacion': round(producto.cobertura_final_plan - cobertura_minima, 2)
                })
        
        df = pd.DataFrame(datos)
        nombre_archivo = f"planificacion_{fecha_planificacion.strftime('%d-%m-%Y')}.csv"
        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")

def main():
    try:
        logger.info("Iniciando planificación de producción...")
        
        # Fechas
        fecha_dataset = datetime.strptime('16-01-2025', '%d-%m-%Y')
        fecha_planificacion = datetime.strptime('20-01-2025', '%d-%m-%Y')
        nombre_dataset = 'Dataset 16-01-25.csv'
        
        # Parámetros de planificación
        dias_planificacion = 6
        dias_no_habiles = 1.6667
        horas_mantenimiento = 9
        dias_cobertura_base = 5
        
        # 1. Leer dataset y calcular fórmulas
        productos = leer_dataset(nombre_dataset)
        if not productos:
            raise ValueError("Error al leer el dataset")
        
        productos_validos, horas_disponibles = calcular_formulas(
            productos=productos,
            fecha_inicio=fecha_planificacion,
            fecha_dataset=fecha_dataset,
            dias_planificacion=dias_planificacion,
            dias_no_habiles=dias_no_habiles,
            horas_mantenimiento=horas_mantenimiento
        )
        
        if not productos_validos:
            raise ValueError("Error en los cálculos")
        
        # 2. Aplicar Simplex
        productos_optimizados = aplicar_simplex(
            productos_validos=productos_validos,
            horas_disponibles=horas_disponibles,
            dias_planificacion=dias_planificacion,
            dias_cobertura_base=dias_cobertura_base
        )
        
        if not productos_optimizados:
            raise ValueError("Error en la optimización")
        
        # 3. Exportar resultados
        exportar_resultados(
            productos_optimizados=productos_optimizados,
            fecha_planificacion=fecha_planificacion,
            dias_planificacion=dias_planificacion,
            dias_cobertura_base=dias_cobertura_base
        )
        
        logger.info("Planificación completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")
        return None

if __name__ == "__main__":
    main()