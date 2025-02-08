import logging
from datetime import datetime
import numpy as np
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
        """Convierte valores a float, maneja valores vacíos y casos especiales"""
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
    """Lee el archivo CSV y crea objetos Producto"""
    try:
        productos = []
        
        with open(nombre_archivo, 'r', encoding='latin1') as file:
            # Saltar las primeras 5 líneas (encabezados)
            for _ in range(5):
                next(file)
            
            # Leer líneas de datos
            for linea in file:
                if not linea.strip() or linea.startswith('Total general'):
                    continue
                    
                campos = linea.strip().split(';')
                if len(campos) >= 15:  # Asegurar que hay suficientes campos
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
        print(f"Error leyendo dataset: {str(e)}")
        return None

def calcular_formulas(productos, dias_planificacion=7, dias_no_habiles=1.6667, horas_mantenimiento=9):
    """Calcula todas las fórmulas para cada producto"""
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
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

            # 3. Demanda provisoria *USAR FECHA_DATASET y FECHA_INICIO
            producto.demanda_provisoria = producto.demanda_media * (4)  #(fecha_inicio - fecha_dataset)

            # 4. Actualizar Disponible *USAR FECHA_DATASET y FECHA_INICIO
            if producto.primera_of != '(en blanco)':
                
                # Convertir las fechas usando los formatos correctos
                of_date = datetime.strptime(producto.primera_of, '%d/%m/%Y')  # Formato dd/mm/yyyy
                date_inicio = datetime.strptime('13-01-25', '%d-%m-%y')      # Formato dd-mm-yy
                date_dataset = datetime.strptime('09-01-25', '%d-%m-%y')     # Formato dd-mm-yy

                if of_date >= date_dataset and of_date < date_inicio:
                    producto.disponible = producto.disponible + producto.of
            else:
                producto.disponible = producto.disponible
            
            # 5. Stock Inicial *USAR FECHA_DATASET y FECHA_INICIO
            producto.stock_inicial = producto.disponible + producto.calidad + producto.stock_externo - producto.demanda_provisoria

            ## ----------------- ALERTA STOCK INICIAL NEGATIVO ----------------- ##    
            ## ----------------- CORREGIR PLANIFICACION (ADELANTAR PLANIFICACION)----------------- ##
            if producto.stock_inicial < 0:
                producto.stock_inicial = 0
                print(f"Producto {producto.cod_art} - {producto.nom_art}: Stock Inicial negativo. Se ajustó a 0.")

            # 6. Cobertura Inicial  
            if producto.demanda_media > 0:
                producto.cobertura_inicial = producto.stock_inicial / producto.demanda_media
            else:
                producto.cobertura_inicial = 'NO VALIDO'
            
            # 7. Demanda Periodo, demanda durante el periodo de planificación
            producto.demanda_periodo = producto.demanda_media * dias_planificacion
            
            # 8. Stock de Seguridad (3 días)
            producto.stock_seguridad = producto.demanda_media * 3       

            # 9. Cobertura Final Estimada (SIN PLANIFICACION)
            if producto.demanda_media > 0:
                 producto.cobertura_final_est = (producto.stock_inicial - producto.demanda_periodo) / producto.demanda_media
            else:
                producto.cobertura_final_est = 'NO VALIDO'

            ## FILTRO POR ARTICULOS EN DOCUMENTO (INDICACIONES ARTICULOS.CSV) DESCATALOGADO Y PEDIDO            


            # Filtros
            if producto.vta_60 > 0 and producto.cajas_hora > 0 and producto.demanda_media > 0 and producto.cobertura_inicial != 'NO VALIDO' and producto.cobertura_final_est != 'NO VALIDO':
                productos_validos.append(producto)
                    
        return productos_validos, horas_disponibles
        
    except Exception as e:
        print(f"Error en cálculos: {str(e)}")
        return None, None

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion):
    """Aplica el método Simplex para optimizar la producción considerando restricciones"""
    try:
        n_productos = len(productos_validos)
        cobertura_minima = 5 + dias_planificacion  # Nueva cobertura mínima

        # Imprimir productos válidos iniciales
        print("\n--- PRODUCTOS VALIDOS PARA PLANIFICAR ---")
        print(f"Total de productos validos: {len(productos_validos)}")

        # Función objetivo: priorizar productos con menor cobertura
        coeficientes = []
        for producto in productos_validos:
            if producto.demanda_media > 0:
                prioridad = max(0, 1/producto.cobertura_inicial)
            else:
                prioridad = 0
            coeficientes.append(-prioridad)

        # Restricción de igualdad para horas totales
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1 / producto.cajas_hora for producto in productos_validos]
        b_eq = [horas_disponibles]

        # Nueva restricción para cobertura mínima
        A_ub = []
        b_ub = []
        for i, producto in enumerate(productos_validos):
            if producto.demanda_media > 0:
                row = [0] * n_productos
                row[i] = -1  # Coeficiente para el producto actual
                A_ub.append(row)
                # Stock mínimo = días cobertura * demanda - stock actual
                stock_min = (producto.demanda_media * cobertura_minima) - producto.stock_inicial
                b_ub.append(-stock_min)  # Negativo porque la restricción es >=

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Bounds: límites mínimos y máximos para cada producto
        bounds = []
        for producto in productos_validos:
            if producto.demanda_media > 0 and producto.cobertura_inicial < 30:
                min_cajas = 2 * producto.cajas_hora  # Mínimo 2 horas
                max_cajas = min(
                    horas_disponibles * producto.cajas_hora,  # Límite por horas
                    producto.demanda_media * 60 - producto.stock_inicial  # Límite por cobertura
                )
                max_cajas = max(min_cajas, max_cajas)  # Asegurar que max >= min
            else:
                min_cajas = 0
                max_cajas = 0
            bounds.append((min_cajas, max_cajas))

        # Resolver optimización
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
            print("\n=== RESULTADOS DE PLANIFICACIÓN ===")
            print(f"{'Producto':<15} {'Cob.Inicial':>10} {'Cob.Final':>10} {'Desv.Objetivo':>12} {'Horas':>8}")
            print("-" * 60)
            
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora
                horas_producidas += producto.horas_necesarias
                
                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
                    # Calcular desviación respecto al objetivo
                    desviacion = producto.cobertura_final_plan - cobertura_minima
                    
                    print(f"{producto.cod_art:<15} {producto.cobertura_inicial:>10.1f} "
                          f"{producto.cobertura_final_plan:>10.1f} {desviacion:>12.1f} "
                          f"{producto.horas_necesarias:>8.1f}")
                else:
                    producto.cobertura_final_plan = float('inf')

          
            return productos_validos
        else:
            print(f"Error en optimización: {result.message}")
            # Imprimir más información de diagnóstico
            print(f"Dimensiones de matrices:")
            print(f"Coeficientes: {len(coeficientes)}")
            print(f"A_eq: {A_eq.shape}")
            print(f"A_ub: {A_ub.shape}")
            print(f"Bounds: {len(bounds)}")
            return None

    except Exception as e:
        print(f"Error en Simplex: {str(e)}")
        print(f"Traza completa:", exc_info=True)
        return None

def main():
    try:
        # 1. Leer dataset
        logger.info("Iniciando lectura del dataset...")
        productos = leer_dataset('Dataset 09-01-25.csv')
        if not productos:
            raise ValueError("Error al leer el dataset")
        
        # 2. Filtrar y calcular fórmulas
        logger.info("Realizando cálculos...")
        productos_validos, horas_disponibles = calcular_formulas(productos)
        
        if not productos_validos:
            raise ValueError("Error en los cálculos")
        
        # 3. Aplicar Simplex
        logger.info("Aplicando optimización Simplex...")
        productos_optimizados = aplicar_simplex(productos_validos, horas_disponibles, 6)
        
        # 4. Imprimir resumen de productos tras Simplex
        if productos_optimizados:
            print("\n--- RESUMEN DE PRODUCTOS TRAS OPTIMIZACIÓN ---")
            print (f"Horas disponibles: {horas_disponibles:.2f}")
            print(f"Total de productos optimizados: {len(productos_optimizados)}")
            print("\nDetalle de productos:")
            for producto in productos_optimizados:
                if producto.horas_necesarias > 0:
                    print(f"\nProducto {producto.cod_art} - {producto.nom_art}")
                    print(f"Demanda media: {producto.demanda_media:.2f}")
                    print(f"Stock inicial: {producto.stock_inicial:.2f}")
                    print(f"Cajas a producir: {producto.cajas_a_producir}")
                    print(f"Horas necesarias: {producto.horas_necesarias:.2f}")
                    print(f"Cobertura inicial: {producto.cobertura_inicial:.2f}")
                    print(f"Cobertura final: {producto.cobertura_final_plan:.2f}")
                    print(f"Cobertura Final Est: {producto.cobertura_final_est:.2f}")
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()