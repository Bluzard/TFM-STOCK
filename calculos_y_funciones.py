# CALCULOS INTERMEDIOS, SIMPLEX Y METODOS/FUNCIONES DE LA IMPLEMENTACION

import os
import sys
import logging
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import math
from scipy.optimize import linprog
from csv_loader import leer_indicaciones_articulos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calcular_formulas(productos, fecha_inicio, fecha_dataset, dias_planificacion, dias_no_habiles, horas_mantenimiento):
    """Calcula todas las fórmulas para cada producto y aplica filtros"""
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        productos_info, productos_omitir = leer_indicaciones_articulos()        
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
                        producto.disponible = producto.disponible + producto.of_reales
            
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

            # Aplicar filtros y asignar orden de planificación
            if (producto.cod_art not in productos_omitir and
                producto.vta_60 > 0 and 
                producto.cajas_hora > 0 and 
                producto.demanda_media > 0 and 
                producto.cobertura_inicial != 'NO VALIDO' and 
                producto.cobertura_final_est != 'NO VALIDO'):
                
                # Asignar orden de planificación
                if producto.cod_art in productos_info:
                    producto.orden_planificacion = productos_info[producto.cod_art]['orden_planificacion']
                productos_validos.append(producto)

        logger.info(f"Productos válidos tras filtros: {len(productos_validos)} de {len(productos)}")
        return productos_validos, horas_disponibles
        
    except Exception as e:
        logger.error(f"Error en cálculos: {str(e)}")
        return None, None

def calcular_cobertura_maxima(m_vta_15):
    """Calcula la cobertura máxima basada en m_vta_15."""
    if m_vta_15 is None:
        logger.warning("m_vta_15 es None. Se asume cobertura máxima infinita.")
        return float('inf')  # Sin límite de cobertura máxima
    
    logger.info(f"Calculando cobertura máxima para m_vta_15={m_vta_15}")  # <-- Log adicional

    if m_vta_15 >= 150:
        return 14.00
    elif 100 <= m_vta_15 < 150:
        return 18.00
    elif 50 <= m_vta_15 < 100:
        return 20.00
    elif 25 <= m_vta_15 < 50:
        return 30.00
    elif 10 <= m_vta_15 < 25:
        return 60.00
    else:
        # logger.info("m_vta_15 < 10, cobertura máxima infinita")  # <-- Log adicional
        return 120.00  # <-- Asegurar el return con 120 días para no rellenar excesivamente la producción y no entrar en conflicto con la restricción 2h min

def redondear_media_hora_al_alza(horas):   # Para adeptar a la realidad del proceso productivo redondemos las horas planificadas por el modelo a divisibles 0.5 horas
    """
    Redondea las horas al múltiplo de 0.5 más cercano, siempre hacia arriba.
    
    Ejemplos:
    - 3.23 -> 3.5
    - 6.71 -> 7.0
    - 4.0 -> 4.0
    - 5.5 -> 5.5
    """
    return math.ceil(horas * 2) / 2


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
        A_eq[0] = [1 / producto.cajas_hora_reales for producto in productos_validos]
        b_eq = [horas_disponibles]

        # <-- CAMBIO: Verificar valores en A_eq y b_eq
        logger.info(f"A_eq: {A_eq}")  # <-- CAMBIO
        logger.info(f"b_eq: {b_eq}")  # <-- CAMBIO

        A_ub = []
        b_ub = []
        bounds = []

        for i, producto in enumerate(productos_validos):
            # Calcular min_cajas y max_cajas
            if producto.demanda_media > 0 and producto.cobertura_inicial < 60:
                cobertura_maxima = calcular_cobertura_maxima(producto.m_vta_15)
                min_cajas = 2 * producto.cajas_hora_reales
                max_cajas = min(
                    horas_disponibles * producto.cajas_hora_reales,
                    producto.demanda_media * cobertura_maxima - producto.stock_inicial
                )
                max_cajas = max(min_cajas, max_cajas)  # Asegurar que max_cajas >= min_cajas
            else:
                min_cajas = 0
                max_cajas = 0

            # Agregar bounds
            bounds.append((min_cajas, max_cajas))

            # Verificar si es necesario agregar una restricción de desigualdad
            if producto.demanda_media > 0:
                stock_min = max((producto.demanda_media * cobertura_minima) - producto.stock_inicial, 0)
                if stock_min > 0:  # Solo agregar la restricción si es necesaria
                    if max_cajas < stock_min:
                        logger.warning(f"Producto {producto.cod_art}: max_cajas ({max_cajas}) < stock_min ({stock_min})")
                    else:
                        row = [0] * n_productos
                        row[i] = -1  # Restricción: -xᵢ ≤ b_ub[i]
                        A_ub.append(row)
                        b_ub.append(-stock_min)

            # Log para verificar valores
            logger.info(f"Producto {producto.cod_art}: min_cajas={min_cajas}, max_cajas={max_cajas}, stock_min={stock_min}")

        # Convertir a arrays de numpy
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Verificar valores en A_ub y b_ub
        logger.info(f"A_ub: {A_ub}")
        logger.info(f"b_ub: {b_ub}")

        # Verificar horas_disponibles * cajas_hora_reales para todos los productos
        for producto in productos_validos:
            horas_cajas = horas_disponibles * producto.cajas_hora_reales
            logger.info(f"Producto {producto.cod_art}: horas_disponibles * cajas_hora_reales = {horas_cajas}")

        # Verificar valores en bounds
        logger.info(f"Bounds: {bounds}")

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
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora_reales
                """horas_producidas += producto.horas_necesarias

                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media"""
                 # Redondear las horas necesarias a tramos de media hora al alza
                producto.horas_necesarias = redondear_media_hora_al_alza(producto.horas_necesarias)

                # Recalcular las cajas a producir basadas en las horas redondeadas
                producto.cajas_a_producir = producto.horas_necesarias * producto.cajas_hora_reales

                # Acumular las horas redondeadas
                horas_producidas += producto.horas_necesarias

                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media

            # Verificar que no se superen las horas disponibles
            if horas_producidas > horas_disponibles:
                logger.warning(f"Horas planificadas ({horas_producidas:.2f}) superan las horas disponibles ({horas_disponibles:.2f}).")
                
                # >>> CAMBIO: Ordenamos los productos por horas asignadas (mayor primero)
                productos_validos.sort(key=lambda p: p.horas_necesarias, reverse=True)

                # >>> CAMBIO: Tomamos el producto con más horas asignadas
                producto_ajustar = productos_validos[0]

                # >>> CAMBIO: Reducir solo el producto con más horas para cuadrar exactamente a horas_disponibles
                diferencia_horas = horas_producidas - horas_disponibles
                producto_ajustar.horas_necesarias = max(producto_ajustar.horas_necesarias - diferencia_horas, 0)

                # Recalcular su producción de cajas después del ajuste
                producto_ajustar.cajas_a_producir = producto_ajustar.horas_necesarias * producto_ajustar.cajas_hora_reales

                # >>> CAMBIO: Volver a calcular horas_producidas después del ajuste
                horas_producidas = sum(producto.horas_necesarias for producto in productos_validos)

                # >>> VERIFICACIÓN FINAL: Asegurar que las horas ajustadas no sobrepasan el límite
                if horas_producidas > horas_disponibles:
                    logger.error(f"Error: No se pudo ajustar correctamente. Horas finales: {horas_producidas:.2f}/{horas_disponibles:.2f}")
                else:
                    logger.info(f"Optimización corregida - Horas finales: {horas_producidas:.2f}/{horas_disponibles:.2f}")


            logger.info(f"Optimización exitosa - Horas planificadas: {horas_producidas:.2f}/{horas_disponibles:.2f}")
            return productos_validos
        else:
            logger.error(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Error en Simplex: {str(e)}")
        return None
            

"""def optimizar_orden_grupos(productos):"""
"""
    Optimiza el orden de los productos minimizando el tiempo perdido en cambios
    entre grupos MEC y VIME."""
"""
    if not productos:
        return []
        
    # Separar productos por grupo
    mec_products = [p for p in productos if p.cod_gru == 'MEC']
    vime_products = [p for p in productos if p.cod_gru == 'VIME']
    
    # Si solo hay productos de un grupo, mantener ese orden
    if not mec_products or not vime_products:
        return productos
        
    # Ordenar cada grupo por cobertura
    mec_products.sort(key=lambda p: p.cobertura_inicial if isinstance(p.cobertura_inicial, (int, float)) else float('inf'))
    vime_products.sort(key=lambda p: p.cobertura_inicial if isinstance(p.cobertura_inicial, (int, float)) else float('inf'))
    
    # Decidir qué grupo va primero basado en los tiempos de setup
    # VIME -> MEC = 8 min
    # MEC -> VIME = 10 min
    # Por lo tanto, es mejor empezar con VIME si hay productos de ambos grupos
    ordered_products = vime_products + mec_products
    
    return ordered_products"""

def clasificar_grupos(productos_optimizados, archivo_indicaciones):  # Clasificamos los artículos por requerimientos de archivo_indicaciones
    """
    Clasifica los productos en tres grupos según el archivo de indicaciones:
    - GRUPO 1: "INICIO"        # Producto que debe fabricarse prioritariamente a inicios de semana (evitar contaminaciones: ecológico, línea producción activa...).
    - GRUPO 2: "VACÍO" o irrelevante # Producto donde es indiferente la localización en el día de la semana.
    - GRUPO 3: "FINAL"         # Producto con relevancia para fabricarse a final de semana (alérgenos, semillas, limpieza de linea...).
    
    :param productos_optimizados: Lista de productos optimizados.
    :param archivo_indicaciones: Ruta al archivo CSV con las indicaciones.
    :return: Tres listas (grupo_inicio, grupo_intermedio, grupo_final).
    """
    try:
        # Leer el archivo de indicaciones
        df_indicaciones = pd.read_csv(archivo_indicaciones, sep=';', encoding='latin1')
        
        # Inicializar los grupos
        grupo_inicio = []
        grupo_intermedio = []
        grupo_final = []
        
        # Clasificar los productos
        for producto in productos_optimizados:
            # Buscar la indicación para este producto
            indicacion = df_indicaciones.loc[
                df_indicaciones['COD_ART'] == producto.cod_art, 'Orden de planificación'
            ].values[0].strip().upper() if producto.cod_art in df_indicaciones['COD_ART'].values else ""
            
            # Clasificar según la indicación
            if indicacion == "INICIO":
                grupo_inicio.append(producto)
            elif indicacion == "FINAL":
                grupo_final.append(producto)
            else:  # Vacío o irrelevante
                grupo_intermedio.append(producto)
        
        logger.info(f"Productos clasificados: INICIO={len(grupo_inicio)}, INTERMEDIO={len(grupo_intermedio)}, FINAL={len(grupo_final)}")
        return grupo_inicio, grupo_intermedio, grupo_final
    
    except Exception as e:
        logger.error(f"Error clasificando grupos: {str(e)}")
        return [], [], []
    
def ordenar_planificacion(grupo_inicio, grupo_intermedio, grupo_final):   # Se ordena la planificación dentro del grupo clasificado por menor cobertura inicial
    """
    Ordena los productos según los grupos clasificados y asigna un número de orden secuencial
    que no se reinicia entre grupos. Además, dentro de cada grupo, los productos se ordenan
    por cobertura inicial de menor a mayor.
    
    :param grupo_inicio: Lista de productos del GRUPO 1.
    :param grupo_intermedio: Lista de productos del GRUPO 2.
    :param grupo_final: Lista de productos del GRUPO 3.
    :return: Lista de productos ordenados con su número de orden secuencial.
    """
    try:
        # Función para ordenar un grupo por cobertura inicial
        def ordenar_por_cobertura(grupo):
            return sorted(grupo, key=lambda x: x.cobertura_inicial if x.cobertura_inicial != 'NO VALIDO' else float('inf'))
        
        # Ordenar cada grupo por cobertura inicial
        grupo_inicio_ordenado = ordenar_por_cobertura(grupo_inicio)
        grupo_intermedio_ordenado = ordenar_por_cobertura(grupo_intermedio)
        grupo_final_ordenado = ordenar_por_cobertura(grupo_final)
        
        # Asignar número de orden secuencial (no se reinicia entre grupos)
        contador_orden = 1
        
        for producto in grupo_inicio_ordenado:
            producto.orden_planificacion = contador_orden
            contador_orden += 1
        
        for producto in grupo_intermedio_ordenado:
            producto.orden_planificacion = contador_orden
            contador_orden += 1
        
        for producto in grupo_final_ordenado:
            producto.orden_planificacion = contador_orden
            contador_orden += 1
        
        # Concatenar los grupos en el orden deseado
        productos_ordenados = grupo_inicio_ordenado + grupo_intermedio_ordenado + grupo_final_ordenado
        
        logger.info(f"Planificación ordenada: {len(productos_ordenados)} productos.")
        return productos_ordenados
    
    except Exception as e:
        logger.error(f"Error ordenando la planificación: {str(e)}")
        return []
    

"""def ordenar_productos(productos):
    
    Ordena los productos según los tres criterios:
    1. Prioridad (INICIO -> sin valor -> FINAL)
    2. Grupo (optimizando cambios entre MEC y VIME)
    3. Cobertura (menor a mayor)
    
    # 1. Separar por prioridad
    inicio = [p for p in productos if p.orden_planificacion == 'INICIO']
    medio = [p for p in productos if not p.orden_planificacion]
    final = [p for p in productos if p.orden_planificacion == 'FINAL']
    
    # 2. Optimizar cada grupo por separado
    inicio_ordenado = optimizar_orden_grupos(inicio)
    medio_ordenado = optimizar_orden_grupos(medio)
    final_ordenado = optimizar_orden_grupos(final)
    
    # 3. Combinar los grupos en el orden correcto
    return inicio_ordenado + medio_ordenado + final_ordenado"""

def verificar_pedidos(productos, df_pedidos, fecha_dataset, dias_planificacion):
    """
    Verifica si los pedidos confirmados provocan rotura de stock
    """
    try:
        if isinstance(fecha_dataset, date):
            fecha_dataset = datetime.combine(fecha_dataset, datetime.min.time())
        # Lista de productos que deben planificarse adicionalmente
        productos_a_planificar = []

        # Recorrer cada producto
        for producto in productos:
            if producto.demanda_media <= 0:
                continue  # No se considera si no hay demanda media

            # Obtener los pedidos para este producto
            pedidos_producto = df_pedidos[df_pedidos['COD_ART'] == producto.cod_art]
            if pedidos_producto.empty:
                continue  # No hay pedidos para este producto

            # Inicializar el stock previsto
            stock_previsto = producto.stock_inicial
            stock_seguridad = producto.demanda_media * 3

            # Recorrer día a día
            for i in range(dias_planificacion):
                fecha_actual = fecha_dataset + timedelta(days=i)
                fecha_str = fecha_actual.strftime('%d/%m/%Y')

                # Restar la demanda media
                stock_previsto -= producto.demanda_media

                # Sumar la OF si la fecha es igual o superior a la del dataset
                if producto.primera_of != '(en blanco)':
                    of_date = datetime.strptime(producto.primera_of, '%d/%m/%Y')
                    if of_date <= fecha_actual:
                        stock_previsto += producto.of_reales

                # Restar los pedidos del día
                if fecha_str in df_pedidos.columns:
                    pedido_dia = pedidos_producto[fecha_str].values[0]
                    if pd.notna(pedido_dia):
                        stock_previsto -= abs(pedido_dia)  # Los pedidos son negativos en el archivo

                # Verificar si el stock previsto es menor que el stock de seguridad
                if stock_previsto < stock_seguridad:
                    logger.warning(f"El día {fecha_str}, el stock de seguridad ha sido sobrepasado para el producto {producto.cod_art}.")

                    # Calcular la cantidad a fabricar
                    dias_faltantes = dias_planificacion - i
                    cantidad_a_fabricar = producto.demanda_media * min(dias_faltantes + 7, dias_planificacion) + abs(pedido_dia)

                    # Añadir el producto a la lista de productos a planificar
                    producto.cajas_a_producir = cantidad_a_fabricar
                    productos_a_planificar.append(producto)
                    break  # Solo necesitamos detectar la primera vez que se sobrepasa el stock de seguridad

        logger.info(f"Se han identificado {len(productos_a_planificar)} productos adicionales para planificar debido a pedidos.")
        return productos_a_planificar

    except Exception as e:
        logger.error(f"Error verificando pedidos: {str(e)}")
        return []
    
def calcular_ocupacion_almacen(productos, productos_info):
    """
    Calcula métricas de ocupación de almacén
    
    :param productos: Lista de productos producidos
    :param productos_info: Diccionario con información adicional de productos
    :return: Diccionario con métricas de ocupación
    """
    total_palets = 0
    total_stock = 0
    
    for producto in productos:
        # Obtener cajas por palet, usar 40 como valor por defecto
        cajas_palet = productos_info.get(producto.cod_art, {}).get('cajas_palet', 40)
        
        # Calcular stock total (inicial + producido)
        stock_total = producto.stock_inicial + producto.cajas_a_producir
        
        # Calcular número de palets
        palets_producto = stock_total / cajas_palet
        
        total_palets += palets_producto
        total_stock += stock_total
    
    # Métricas de penalización según documento original
    penalizacion = 0
    if total_palets > 1200:
        penalizacion = -100
    elif total_palets > 1000:
        penalizacion = -50
    elif total_palets > 800:
        penalizacion = -10
    
    return {
        'total_palets': round(total_palets, 2),
        'total_stock': round(total_stock, 2),
        'penalizacion_espacio': penalizacion
    }

def exportar_resultados(productos_ordenados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    """Exporta los resultados a un archivo CSV"""
    try:
        datos = []
        productos_info, productos_omitir = leer_indicaciones_articulos()

        # Obtener información de productos
        productos_info, _ = leer_indicaciones_articulos()
        # Calcular ocupación de almacén
        ocupacion = calcular_ocupacion_almacen(productos_ordenados, productos_info)

        # Leer el archivo de indicaciones para obtener el grupo de cada producto, para verificar el orden despues quitar
        df_indicaciones = pd.read_csv('Indicaciones articulos.csv', sep=';', encoding='latin1')
        grupos = {}
        for _, row in df_indicaciones.iterrows():
            cod_art = row['COD_ART']
            grupo = row['ORDEN PLANIFICACION'].strip().upper() if pd.notna(row['ORDEN PLANIFICACION']) else " "
            grupos[cod_art] = grupo

        # Agrupar productos por estado
        productos_planificados = []
        productos_validos = []
        productos_no_validos = []
        
        # Clasificar productos
        for producto in productos:
            if producto.cod_art not in productos_omitir:
                # Verificar si el producto está en productos_ordenados
                producto_opt = next((p for p in productos_ordenados if p.cod_art == producto.cod_art), None)
                
                if producto_opt:
                    if producto_opt.horas_necesarias > 0:
                        productos_planificados.append(producto_opt)
                    else:
                        productos_validos.append(producto_opt)
                else:
                    productos_no_validos.append(producto)
        
        """ # Ordenar cada grupo
        productos_planificados = ordenar_planificacion(grupo_inicio, grupo_intermedio, grupo_final)
        # productos_validos = ordenar_planificacion(productos_validos)
        # productos_no_validos = ordenar_planificacion(productos_no_validos)"""
        
        # Procesar productos en orden
        for producto in productos_planificados + productos_validos + productos_no_validos:
            estado = "Planificado" if producto in productos_planificados else \
                    "Válido sin producción" if producto in productos_validos else \
                    "No válido"
                    
            cobertura_final = producto.cobertura_final_plan if hasattr(producto, 'cobertura_final_plan') else producto.cobertura_final_est
            cajas_producir = producto.cajas_a_producir if hasattr(producto, 'cajas_a_producir') else 0
            horas_necesarias = producto.horas_necesarias if hasattr(producto, 'horas_necesarias') else 0

            # Si el producto no está planificado, el orden de planificación es 0
            orden_planificacion = producto.orden_planificacion if hasattr(producto, 'orden_planificacion') else 0
            
            datos.append({
                'COD_ART': producto.cod_art,
                'NOM_ART': producto.nom_art,
                'COD_GRU': producto.cod_gru,
                'Estado': estado,
                'Grupo': grupo,  # Nueva columna para el grupo
                'Orden_Planificacion': producto.orden_planificacion,   # Número de orden secuencial (0 si no está planificado)
                'Demanda_Media': round(producto.demanda_media, 2) if producto.demanda_media != 'NO VALIDO' else 0,
                'Stock_Inicial': round(producto.stock_inicial, 2),
                'Cajas_a_Producir': cajas_producir,
                'Horas_Necesarias': round(horas_necesarias, 2),
                'Cobertura_Inicial': round(producto.cobertura_inicial, 2) if producto.cobertura_inicial != 'NO VALIDO' else 0,
                'Cobertura_Final': round(cobertura_final, 2) if cobertura_final != 'NO VALIDO' else 0,
                'Cobertura_Final_Est': round(producto.cobertura_final_est, 2) if producto.cobertura_final_est != 'NO VALIDO' else 0,
                'Tipo': 'RESUMEN',
                'Total_Palets': ocupacion['total_palets'],
                'Total_Stock': ocupacion['total_stock'],
                'Penalizacion_Espacio': ocupacion['penalizacion_espacio']
            })
        
        df = pd.DataFrame(datos)
            
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"

        # Verificar si el archivo ya existe y eliminarlo si es necesario
        # Verificar si el archivo ya existe y eliminarlo si es necesario
        if os.path.exists(nombre_archivo):
            try:
                os.remove(nombre_archivo)  # Intentar eliminar el archivo
            except PermissionError:
                # Si el archivo está abierto, mostrar un mensaje claro al usuario
                print(f"Error: El archivo '{nombre_archivo}' está abierto en otro programa.")
                print("Por favor, cierre el archivo y vuelva a intentarlo.")
                sys.exit(1)  # Salir del programa con un código de error

        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
        logger.error("Tipo de fecha_planificacion: %s", type(fecha_planificacion))
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
