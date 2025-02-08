import logging
from datetime import datetime
from scipy.optimize import linprog
import numpy as np

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
        self.vta_60 = self._convertir_float(vta_60)  # Nuevo campo
        self.vta_15 = self._convertir_float(vta_15)
        self.m_vta_15 = self._convertir_float(m_vta_15)
        self.vta_15_aa = self._convertir_float(vta_15_aa)
        self.m_vta_15_aa = self._convertir_float(m_vta_15_aa)
        self.vta_15_mas_aa = self._convertir_float(vta_15_mas_aa)
        self.m_vta_15_mas_aa = self._convertir_float(m_vta_15_mas_aa)
        
        # Campos calculados (inicialmente 0)
        self.demanda_media = 0
        self.stock_total = 0
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

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion):
    """Aplica el método Simplex para optimizar la producción"""
    try:
        print("\nIniciando optimización Simplex...")
        n_productos = len(productos_validos)
        
        # Función objetivo: Z = Σ(demanda_media * dias - stock_total - x)
        c = np.ones(n_productos)  # Coeficiente de x_ij es 1 
        
        # 1. Restricción de horas totales (igualdad)
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1/producto.cajas_hora for producto in productos_validos]
        b_eq = [horas_disponibles]
        
        # 2. Restricción de producción mínima (≥ 2 horas)
        A_ub = np.zeros((n_productos, n_productos))
        np.fill_diagonal(A_ub, [-1/producto.cajas_hora for producto in productos_validos])
        b_ub = [-2 for _ in range(n_productos)]
        
        print("\nResolviendo sistema...")
        print(f"Productos a optimizar: {n_productos}")
        print(f"Horas a distribuir: {horas_disponibles:.2f}")
        
        # Resolver Simplex
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            method='highs'
        )
        
        if result.success:
            print("\nOptimización exitosa!")
            print(f"Valor función objetivo: {result.fun:.2f}")
            
            # Asignar resultados
            print("\nPlan de producción:")
            total_horas = 0
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora
                total_horas += producto.horas_necesarias
                
                if producto.horas_necesarias > 0:
                    print(f"\nProducto {producto.cod_art}:")
                    print(f"- Cajas a producir: {producto.cajas_a_producir}")
                    print(f"- Horas necesarias: {producto.horas_necesarias:.2f}")
                    deficit = (producto.demanda_media * dias_planificacion - 
                             producto.stock_total - producto.cajas_a_producir)
                    print(f"- Déficit final: {deficit:.2f}")
            
            print(f"\nTotal horas planificadas: {total_horas:.2f}")
            return productos_validos
            
        else:
            print(f"Error en optimización: {result.message}")
            return None
            
    except Exception as e:
        print(f"Error en Simplex: {str(e)}")
        return None
    
def calcular_formulas(productos, dias_planificacion=6, dias_no_habiles=0.6667, horas_mantenimiento=9):
    """Calcula todas las fórmulas para cada producto"""
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        productos_validos = []
        
        for producto in productos:
            # 2. Cálculo de Demanda Media
            if producto.m_vta_15_aa > 0:
                variacion_aa = abs(1 - (producto.vta_15_mas_aa / producto.vta_15_aa))
                if variacion_aa > 0.20:
                    producto.demanda_media = producto.m_vta_15 * (producto.m_vta_15/ producto.m_vta_15_aa) / 15
                else:
                    producto.demanda_media = producto.m_vta_15 / 15
            else:
                producto.demanda_media = 0
            
            # 3. Stock Total
            producto.stock_total = (producto.disponible + 
                                  producto.calidad + 
                                  producto.stock_externo)
            
            # 4. Cobertura Actual
            if producto.demanda_media > 0:
                producto.cobertura_actual = producto.stock_total / producto.demanda_media
            else:
                producto.cobertura_actual = 0
            
            # 5. Stock de Seguridad (3 días)
            producto.stock_seguridad = producto.demanda_media * 3
            
            # Filtros
            if producto.vta_60 > 0 and producto.cajas_hora > 0:
                productos_validos.append(producto)
        
        # 6. Aplicar Simplex a productos válidos
        productos_optimizados = aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion)
        if productos_optimizados is None:
            return None, None
                
        return productos_optimizados, horas_disponibles
        
    except Exception as e:
        print(f"Error en cálculos: {str(e)}")
        return None, None

def mostrar_producto(producto, indice):
    """Muestra los datos de un producto"""
    print(f"\nProducto {indice}:")
    print(f"Código: {producto.cod_art}")
    print(f"Nombre: {producto.nom_art}")
    print(f"Grupo: {producto.cod_gru}")
    print(f"Cajas/Hora: {producto.cajas_hora}")
    print(f"Stock Total: {producto.stock_total if producto.stock_total is not None else 'No calculado'}")
    print(f"Demanda Media: {producto.demanda_media if producto.demanda_media is not None else 'No calculado'}")
    print(f"Cobertura Actual: {producto.cobertura_actual if producto.cobertura_actual is not None else 'No calculado'}")
    print(f"Stock Seguridad: {producto.stock_seguridad if producto.stock_seguridad is not None else 'No calculado'}")

def main():
    try:
        # 1. Leer dataset
        logger.info("Iniciando lectura del dataset...")
        productos = leer_dataset('Dataset 09-01-25.csv')
        if not productos:
            raise ValueError("Error al leer el dataset")
        
        # 3. Realizar cálculos
        logger.info("Realizando cálculos...")
        productos, horas_disponibles = calcular_formulas(productos)
        if not productos:
            raise ValueError("Error en los cálculos")
            
        print(f"\nHoras disponibles totales: {horas_disponibles:.2f}")
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()