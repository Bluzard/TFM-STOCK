import logging
from datetime import datetime
import numpy as np
from scipy.optimize import linprog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Producto:
    def __init__(self, cod_art, nom_art, cod_gru, cajas_hora, disponible, calidad, 
                 stock_externo, pedido, primera_of, vta_60, vta_15, m_vta_15, 
                 vta_15_aa, m_vta_15_aa, vta_15_mas_aa, m_vta_15_mas_aa):
        # Datos básicos
        self.cod_art = cod_art
        self.nom_art = nom_art
        self.cod_gru = cod_gru
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
        self.cobertura_actual = 0
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

def calcular_formulas(productos, dias_planificacion=6, dias_no_habiles=0.6667, horas_mantenimiento=9):
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

            # 3. Demanda provisoria
            producto.demanda_provisoria = producto.demanda_media * (4)  #(fecha_inicio - fecha_dataset)
            
            # 4. Stock Inicial
            producto.stock_inicial = producto.disponible + producto.calidad + producto.stock_externo - producto.demanda_provisoria
            
            # 5. Cobertura Actual
            if producto.demanda_media > 0:
                producto.cobertura_actual = producto.stock_inicial / producto.demanda_media
            else:
                producto.cobertura_actual = 0
            
            # 6. Stock de Seguridad (3 días)
            producto.stock_seguridad = producto.demanda_media * 3
            
            # Filtros
            if producto.vta_60 > 0 and producto.cajas_hora > 0:
                productos_validos.append(producto)
        
        return productos_validos, horas_disponibles
        
    except Exception as e:
        print(f"Error en cálculos: {str(e)}")
        return None, None

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion):
    """Aplica el método Simplex para optimizar la producción considerando restricciones"""
    try:
        n_productos = len(productos_validos)

        # Función objetivo: minimizar déficit 
        coeficientes = [
            abs(producto.demanda_media * dias_planificacion - producto.stock_inicial) 
            for producto in productos_validos
        ]

        # Matriz de restricciones
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1 / producto.cajas_hora for producto in productos_validos]
        b_eq = [horas_disponibles]

        # Restricciones de producción mínima (al menos 2 horas)
        A_ub = np.zeros((n_productos + 1, n_productos))
        for i in range(n_productos):
            A_ub[i, i] = -1 / productos_validos[i].cajas_hora  # Evita valores negativos
        b_ub = [-2] * n_productos

        # Restricción de cobertura máxima (no más de 60 días de stock)
        A_ub[-1] = [producto.demanda_media * 60 - producto.stock_inicial for producto in productos_validos]
        b_ub.append(0)

        result = linprog(
            c=coeficientes,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            method='highs'
        )

        if result.success:
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora

            return productos_validos

        else:
            print(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        print(f"Error en Simplex: {str(e)}")
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
            print(f"Total de productos optimizados: {len(productos_optimizados)}")
            print("\nDetalle de productos:")
            for producto in productos_optimizados:
                print(f"\nProducto {producto.cod_art} - {producto.nom_art}")
                print(f"Demanda media: {producto.demanda_media:.2f}")
                print(f"Stock total inicial: {producto.stock_inicial:.2f}")
                print(f"Cajas a producir: {producto.cajas_a_producir}")
                print(f"Horas necesarias: {producto.horas_necesarias:.2f}")
                print(f"Cobertura actual: {producto.cobertura_actual:.2f}")
                print(f"Stock de seguridad: {producto.stock_seguridad:.2f}")
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()