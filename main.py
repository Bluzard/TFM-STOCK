import logging
from datetime import datetime
import numpy as np
import pandas as pd
import os
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
        
        carpeta="Dataset"
                # Encontrar archivo correspondiente
        archivo = next(
            (f for f in os.listdir(carpeta) 
                if f.endswith('.csv') and nombre_archivo in f),
            None
        )
        
        if not archivo:
            raise FileNotFoundError(
                f"No se encontró archivo para fecha: {nombre_archivo}"
            )
        
        ruta_completa = os.path.join(carpeta, archivo)
             
               
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

        # Convertir fechas usando el formato correcto
        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%d-%m-%Y')
            # Convertir fecha_dataset a formato completo YYYY
            if len(fecha_dataset.split('-')[2]) == 2:  # Si el año tiene 2 dígitos
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%y')
            else:  # Si el año tiene 4 dígitos
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%Y')
            
            logger.info(f"Fecha inicio: {fecha_inicio_dt}, Fecha dataset: {fecha_dataset_dt}")
        except ValueError as e:
            logger.error(f"Error en formato de fechas: {str(e)}")
            return None, None
        
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
            dias_diff = (fecha_inicio_dt - fecha_dataset_dt).days
            producto.demanda_provisoria = producto.demanda_media * dias_diff
            # 4. Actualizar Disponible
            if producto.primera_of != '(en blanco)':
                of_date = datetime.strptime(producto.primera_of, '%d/%m/%Y')
                if of_date >= fecha_dataset_dt and of_date < fecha_inicio_dt:
                        producto.disponible = producto.disponible + producto.of
            
            # 5. Stock Inicial
            producto.stock_inicial = producto.disponible + producto.calidad + producto.stock_externo - producto.demanda_provisoria

            ## ----------------- ALERTA STOCK INICIAL NEGATIVO ----------------- ##    
            # Verificar si el stock inicial es negativo
            if producto.stock_inicial < 0:
                print("\n⚠️  ALERTA: STOCK INICIAL NEGATIVO ⚠️")
                print("El stock inicial del producto es menor a 0.")
                print("🔹 Se recomienda adelantar la planificación para evitar problemas.\n")
                
                # Preguntar al usuario si desea continuar
                respuesta = input("¿Desea continuar de todos modos? (s/n): ").strip().lower()

                if respuesta != 's':
                    print("⛔ Proceso interrumpido por el usuario.")
                    exit()  # Detiene la ejecución del programa

                # El código continúa normalmente si el usuario elige 's'
                print("✅ Continuando con la ejecución...")

            
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

        # Función objetivo - Ajustada para manejar coberturas de 0
        coeficientes = []
        for producto in productos_validos:
            if producto.demanda_media > 0:
                # Si la cobertura es 0, asignamos la máxima prioridad
                if producto.cobertura_inicial == 0:
                    prioridad = 1000  # Valor alto para máxima prioridad
                else:
                    prioridad = max(0, 1/producto.cobertura_inicial)
            else:
                prioridad = 0
            coeficientes.append(-prioridad)  # Negativo porque minimizamos

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

        # Bounds - Ajustados para considerar stock inicial 0
        bounds = []
        for producto in productos_validos:
            if producto.demanda_media > 0 and producto.cobertura_inicial < 30:
                min_cajas = 2 * producto.cajas_hora
                # Si el stock inicial es 0, priorizamos producción mínima
                if producto.stock_inicial == 0:
                    min_cajas = max(min_cajas, producto.demanda_media * 3)  # Mínimo 3 días de cobertura
                
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

def exportar_resultados(productos_optimizados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    try:
        datos = []
        productos_omitir = leer_indicaciones_articulos()
        
        # Obtener todos los productos activos
        for producto in productos:
            if producto.cod_art not in productos_omitir:
                estado = "No válido"  # Por defecto
                
                # Verificar si el producto está en productos_optimizados
                producto_opt = next((p for p in productos_optimizados if p.cod_art == producto.cod_art), None)
                
                if producto_opt:
                    if producto_opt.horas_necesarias > 0:
                        estado = "Planificado"
                    else:
                        estado = "Válido sin producción"
                        
                    # Usar valores del producto optimizado si existe
                    producto_final = producto_opt
                else:
                    producto_final = producto
                
                # Calcular coberturas y valores finales
                cobertura_final = producto_final.cobertura_final_plan if hasattr(producto_final, 'cobertura_final_plan') else producto_final.cobertura_final_est
                cajas_producir = producto_final.cajas_a_producir if hasattr(producto_final, 'cajas_a_producir') else 0
                horas_necesarias = producto_final.horas_necesarias if hasattr(producto_final, 'horas_necesarias') else 0
                
                datos.append({
                    'COD_ART': producto_final.cod_art,
                    'NOM_ART': producto_final.nom_art,
                    'Estado': estado,
                    'Demanda_Media': round(producto_final.demanda_media, 2) if producto_final.demanda_media != 'NO VALIDO' else 0,
                    'Stock_Inicial': round(producto_final.stock_inicial, 2),
                    'Cajas_a_Producir': cajas_producir,
                    'Horas_Necesarias': round(horas_necesarias, 2),
                    'Cobertura_Inicial': round(producto_final.cobertura_inicial, 2) if producto_final.cobertura_inicial != 'NO VALIDO' else 0,
                    'Cobertura_Final': round(cobertura_final, 2) if cobertura_final != 'NO VALIDO' else 0,
                    'Cobertura_Final_Est': round(producto_final.cobertura_final_est, 2) if producto_final.cobertura_final_est != 'NO VALIDO' else 0
                })
        
        df = pd.DataFrame(datos)
        
        # Ordenar el DataFrame por Estado y Cobertura_Inicial
        df['orden_estado'] = df['Estado'].map({'Planificado': 0, 'Válido sin producción': 1, 'No válido': 2})
        df = df.sort_values(['orden_estado', 'Cobertura_Inicial'])
        df = df.drop('orden_estado', axis=1)
        
        # Convertir fecha_planificacion a datetime si es string
        if isinstance(fecha_planificacion, str):
            fecha_planificacion = datetime.strptime(fecha_planificacion, '%d-%m-%Y')
            
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"

        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
        logger.error("Tipo de fecha_planificacion: %s", type(fecha_planificacion))
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")

def validar_fecha(fecha_str, nombre_campo):
    """Valida el formato de fecha y la convierte a objeto datetime"""
    try:
        if not fecha_str:
            raise ValueError(f"La {nombre_campo} no puede estar vacía")
        
        # Normalizar formato
        fecha_str = fecha_str.replace("/", "-")
        
        # Intentar diferentes formatos
        try:
            fecha_dt = datetime.strptime(fecha_str, '%d-%m-%Y')
        except ValueError:
            try:
                fecha_dt = datetime.strptime(fecha_str, '%d-%m-%y')
                # Convertir a formato completo
                fecha_str = fecha_dt.strftime('%d-%m-%Y')
                fecha_dt = datetime.strptime(fecha_str, '%d-%m-%Y')
            except ValueError:
                raise ValueError(f"Formato de {nombre_campo} inválido. Use DD-MM-YYYY o DD-MM-YY")
        
        return fecha_dt, fecha_str
    except Exception as e:
        raise ValueError(f"Error validando {nombre_campo}: {str(e)}")

def verificar_dataset_existe(nombre_archivo):
    """Verifica si existe el archivo de dataset"""
    return True
    """
    try:
        if not os.path.exists(nombre_archivo):
            print(f"\n⚠️  ADVERTENCIA: No se encuentra el archivo '{nombre_archivo}'")
            print("Verifique que el archivo existe en el directorio actual con ese nombre exacto.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error verificando dataset: {str(e)}")
        return False
    """


def solicitar_parametros():
    """Solicita y valida todos los parámetros de entrada"""
    while True:
        try:
            # Fecha dataset
            fecha_dataset_str = input("\nIngrese fecha de dataset (DD-MM-YYYY): ").strip()
            fecha_dataset_dt, fecha_dataset_str = validar_fecha(fecha_dataset_str, "fecha de dataset")
            
            # Verificar existencia del dataset
            if not verificar_dataset_existe(fecha_dataset_dt):
                print("\n🔄 Por favor, ingrese una nueva fecha.")
                continue
            
            # Fecha planificación
            while True:
                fecha_planif_str = input("\nIngrese fecha inicio planificación (DD-MM-YYYY): ").strip()
                fecha_planif_dt, fecha_planif_str = validar_fecha(fecha_planif_str, "fecha de planificación")
                
                if fecha_planif_dt < fecha_dataset_dt:
                    print("❌ La fecha de planificación debe ser posterior a la fecha del dataset")
                else:
                    break
            
            # Días de planificación
            while True:
                try:
                    dias_planif = int(input("\nIngrese días de planificación: "))
                    if dias_planif <= 0:
                        print("❌ Los días de planificación deben ser positivos")
                        continue
                    break
                except ValueError:
                    print("❌ Por favor ingrese un número entero válido")
            
            # Días no hábiles
            while True:
                try:
                    dias_no_habiles = float(input("\nIngrese días no hábiles en el periodo: "))
                    if dias_no_habiles < 0 or dias_no_habiles >= dias_planif:
                        print("❌ Los días no hábiles deben ser un número entre 0 y los días de planificación")
                        continue
                    break
                except ValueError:
                    print("❌ Por favor ingrese un número válido")
            
            # Horas mantenimiento
            while True:
                try:
                    horas_mant = int(input("\nIngrese horas de mantenimiento: "))
                    if horas_mant < 0:
                        print("❌ Las horas de mantenimiento no pueden ser negativas")
                        continue
                    break
                except ValueError:
                    print("❌ Por favor ingrese un número entero válido")
            
            return {
                'fecha_dataset': fecha_dataset_str,
                'fecha_dataset_dt': fecha_dataset_dt,
                'fecha_planificacion': fecha_planif_str,
                'fecha_planificacion_dt': fecha_planif_dt,
                'dias_planificacion': dias_planif,
                'dias_no_habiles': dias_no_habiles,
                'horas_mantenimiento': horas_mant
            }
                
        except ValueError as ve:
            print(f"\n❌ Error: {str(ve)}")
            continuar = input("\n¿Desea intentar nuevamente? (s/n): ").strip().lower()
            if continuar != 's':
                print("⛔ Proceso interrumpido por el usuario.")
                return None
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            return None
def main():
    try:
        logger.info("Iniciando planificación de producción...")
        
        while True:
            # Fechas
            fecha_dataset = input("Ingrese fecha de dataset DD-MM-YYYY: ").strip()
            
            # Formatear la fecha para el nombre del archivo
            try:
                nombre_dataset = datetime.strptime(
                    fecha_dataset.replace("/", "-"), 
                    "%d-%m-%Y"
                ).strftime("%d-%m-%y") + '.csv'
                
                if verificar_dataset_existe(nombre_dataset):
                    break
                print("\n🔄 Por favor, ingrese una nueva fecha.")
            except ValueError:
                print("\n❌ Formato de fecha inválido. Use DD-MM-YYYY")
                continue
        
        fecha_planificacion = input("Ingrese fecha inicio planificación DD-MM-YYYY: ").strip()
        
        try:
            fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%Y')
            fecha_planificacion_dt = datetime.strptime(fecha_planificacion, '%d-%m-%Y')
            
            if fecha_planificacion_dt < fecha_dataset_dt:
                raise ValueError("La fecha de planificación debe ser posterior a la fecha del dataset")
        except ValueError as e:
            logger.error(f"Error en fechas: {str(e)}")
            return None
            
        # Parámetros de planificación
        try:
            dias_planificacion = int(input("Ingrese días de planificación: "))
            if dias_planificacion <= 0:
                raise ValueError("Los días de planificación deben ser positivos")
                
            dias_no_habiles = float(input("Ingrese días no hábiles en el periodo: "))
            if dias_no_habiles < 0 or dias_no_habiles >= dias_planificacion:
                raise ValueError("Días no hábiles inválidos")
                
            horas_mantenimiento = int(input("Ingrese horas de mantenimiento: "))
            if horas_mantenimiento < 0:
                raise ValueError("Las horas de mantenimiento no pueden ser negativas")
            
            dias_cobertura_base = int(input("Ingrese días de cobertura tras planificación: "))
            if dias_cobertura_base <= 0:
                raise ValueError("Los días de cobertura deben ser positivos")
        except ValueError as e:
            logger.error(f"Error en parámetros: {str(e)}")
            return None
            
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
            productos=productos,  # Agregamos todos los productos originales
            fecha_dataset=fecha_dataset_dt,
            fecha_planificacion=fecha_planificacion_dt,
            dias_planificacion=dias_planificacion,
            dias_cobertura_base=dias_cobertura_base
        )
        
        logger.info("Planificación completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")
        return None

if __name__ == "__main__":
    main()