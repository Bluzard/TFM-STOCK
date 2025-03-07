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
    try:
        # Consolidar productos con el mismo código antes de la optimización
        productos_consolidados = {}
        for producto in productos_validos:
            if producto.cod_art not in productos_consolidados:
                productos_consolidados[producto.cod_art] = producto
            else:
                # Combinar productos con el mismo código
                existente = productos_consolidados[producto.cod_art]
                existente.cajas_hora = (existente.cajas_hora + producto.cajas_hora) / 2
                existente.cajas_hora_reales = (existente.cajas_hora_reales + producto.cajas_hora_reales) / 2
                existente.m_vta_15 = max(existente.m_vta_15, producto.m_vta_15)
                existente.stock_inicial = max(existente.stock_inicial, producto.stock_inicial)

        # Convertir a lista de productos válidos sin restricciones infactibles
        productos_validos = [
            p for p in productos_consolidados.values() 
            if p.demanda_media > 0 and p.cajas_hora_reales > 0
        ]

        n_productos = len(productos_validos)
        if n_productos == 0:
            logger.error("No hay productos válidos para optimizar")
            return None

        cobertura_minima = dias_cobertura_base + dias_planificacion

        # Función objetivo
        coeficientes = []
        for producto in productos_validos:
            if producto.demanda_media > 0:
                # Priorizar productos con menor cobertura
                prioridad = max(0, 1 / (producto.cobertura_inicial + 0.01))
            else:
                prioridad = 0
            coeficientes.append(-prioridad)

        # Restricciones
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1 / producto.cajas_hora_reales for producto in productos_validos]
        b_eq = [horas_disponibles]

        A_ub = []
        b_ub = []
        bounds = []

        for i, producto in enumerate(productos_validos):
            # Calcular límites de producción más flexibles
            cobertura_maxima = calcular_cobertura_maxima(producto.m_vta_15)
            
            # Calcular cajas mínimas y máximas
            min_cajas = max(2 * producto.cajas_hora_reales, 0)
            max_cajas = min(
                horas_disponibles * producto.cajas_hora_reales,
                max(producto.demanda_media * cobertura_maxima - producto.stock_inicial, min_cajas)
            )

            bounds.append((min_cajas, max_cajas))

            # Restricción de stock mínimo más flexible
            stock_min = max((producto.demanda_media * cobertura_minima) - producto.stock_inicial, 0)
            if stock_min > 0:
                row = [0] * n_productos
                row[i] = -1
                A_ub.append(row)
                b_ub.append(-stock_min)

        # Convertir a arrays de numpy
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        # Optimización con manejo de restricciones opcionales
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
            horas_redondeadas = 0
            
            for i, producto in enumerate(productos_validos):
                # Cálculo de cajas y horas con redondeo consistente
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora_reales

                # Redondear horas al múltiplo de 0.5 más cercano
                producto.horas_necesarias = redondear_media_hora_al_alza(producto.horas_necesarias)

                # Recalcular cajas basadas en horas redondeadas
                producto.cajas_a_producir = round(producto.horas_necesarias * producto.cajas_hora_reales)

                # Acumular horas
                horas_redondeadas += producto.horas_necesarias

                # Calcular cobertura final
                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
            
            logger.info(f"Optimización exitosa - Horas planificadas: {horas_redondeadas:.2f}/{horas_disponibles:.2f}")
            return productos_validos
        else:
            logger.error(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Error en Simplex: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return None
def optimizar_orden_grupos(productos):
    """
    Optimiza el orden de los productos minimizando el tiempo perdido en cambios
    entre grupos MEC y VIME."""

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
    
    return ordered_products

def ordenar_productos(df):
    """
    Ordena el DataFrame de productos según los criterios especificados
    """
    # Definir orden de prioridad para Estado
    df['orden_estado'] = df['Estado'].map({
        'Planificado': 0,
        'Válido sin producción': 1,
        'No válido': 2
    })
    
    # Definir orden de prioridad para Orden_Planificacion
    df['orden_planificacion'] = df['Orden_Planificacion'].map({
        'INICIO': 0,
        '': 1,
        'FINAL': 2
    }).fillna(1)
    
    # Definir orden para COD_GRU
    df['orden_grupo'] = df['COD_GRU'].map({
        'VIME': 0,
        'MEC': 1
    })
    
    # Asegurar que Cobertura_Inicial sea numérica
    df['Cobertura_Sort'] = pd.to_numeric(df['Cobertura_Inicial'], errors='coerce').fillna(float('inf'))
    
    # Dar prioridad extra a productos con cobertura final estimada negativa
    mask_negativa = df['Cobertura_Final_Est'] < 0
    df.loc[mask_negativa, 'Cobertura_Sort'] = -1
    
    # Ordenar el DataFrame
    df_ordenado = df.sort_values([
        'orden_estado',
        'orden_planificacion',
        'orden_grupo',
        'Cobertura_Sort'
    ])
    
    # Eliminar columnas temporales de ordenamiento
    df_ordenado = df_ordenado.drop(['orden_estado', 'orden_planificacion', 'orden_grupo', 'Cobertura_Sort'], axis=1)
    
    return df_ordenado

def verificar_pedidos(productos, df_pedidos, fecha_dataset, dias_planificacion):
    """
    Verifica si los pedidos confirmados provocan rotura de stock
    """
    try:
        if isinstance(fecha_dataset, date):
            fecha_dataset = datetime.combine(fecha_dataset, datetime.min.time())
            
        # CORRECCIÓN: No intentar obtener basename de un DataFrame
        logger.info(f"Procesando datos de pedidos pendientes")
        logger.info(f"Pedidos pendientes cargados: {len(df_pedidos)} productos")
        
        # Identificar columnas de fechas
        fecha_cols = [col for col in df_pedidos.columns if col not in ['COD_ART', 'NOM_ART']]
        logger.info(f"Columnas de fechas: {fecha_cols}")
        
        # Lista de productos que deben planificarse adicionalmente
        productos_a_planificar = []

        # Recorrer cada producto
        for producto in productos:
            if producto.demanda_media <= 0:
                continue  # No se considera si no hay demanda media

            # Obtener los pedidos para este producto
            pedidos_producto = df_pedidos[df_pedidos['COD_ART'] == str(producto.cod_art)]
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

                # Verificar si el stock previsto es menor que el stock de seguridad
                if stock_previsto < stock_seguridad:
                    logger.warning(f"El día {fecha_str}, el stock de seguridad ha sido sobrepasado para el producto {producto.cod_art}.")

                    # Calcular la cantidad a fabricar
                    dias_faltantes = dias_planificacion - i
                    cantidad_a_fabricar = producto.demanda_media * min(dias_faltantes + 7, dias_planificacion)
                    
                    # Para los pedidos específicos, buscar en todas las columnas de fecha
                    pedido_dia = 0
                    for col in fecha_cols:
                        try:
                            # Intentar obtener el valor del pedido para esta fecha
                            if col in pedidos_producto.columns and not pedidos_producto[col].empty:
                                valor = pedidos_producto[col].values[0]
                                if pd.notna(valor) and valor != 0:
                                    pedido_dia += abs(valor)  # Los pedidos son negativos en el archivo
                        except Exception as e:
                            logger.warning(f"Error al procesar pedido en columna {col}: {str(e)}")

                    # Añadir pedidos específicos si existen
                    if pedido_dia > 0:
                        cantidad_a_fabricar += pedido_dia
                        logger.info(f"Añadiendo pedido específico de {pedido_dia} cajas para producto {producto.cod_art}")

                    # Añadir el producto a la lista de productos a planificar
                    producto.cajas_a_producir = cantidad_a_fabricar
                    productos_a_planificar.append(producto)
                    break  # Solo necesitamos detectar la primera vez que se sobrepasa el stock de seguridad

        logger.info(f"Se han identificado {len(productos_a_planificar)} productos adicionales para planificar debido a pedidos.")
        return productos_a_planificar

    except Exception as e:
        logger.error(f"Error verificando pedidos: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return []
    
def calcular_ocupacion_almacen(productos, fecha_tag, productos_info=None):
    """
    Calcula la ocupación del almacén en ubicaciones para un momento específico.
    
    Args:
        productos: Lista de objetos Producto
        fecha_tag: Etiqueta para identificar qué stock usar ('dataset', 'inicio', 'fin')
        productos_info: Diccionario con información adicional de productos (cajas/palet)
        
    Returns:
        dict: Diccionario con información de ocupación
    """
    if productos_info is None:
        # Si no se proporciona info de productos, intentar cargarla
        productos_info, _ = leer_indicaciones_articulos()
        
    total_ubicaciones = 0
    total_cajas = 0
    productos_procesados = 0
    
    for producto in productos:
        # Omitir productos descatalogados o sin información
        if not hasattr(producto, 'cod_art') or producto.cod_art not in productos_info:
            continue
            
        # Obtener cajas por palet (valor por defecto: 40)
        cajas_palet = productos_info[producto.cod_art].get('cajas_palet', 40)
        if cajas_palet <= 0:
            cajas_palet = 40  # Evitar división por cero
            
        # Determinar qué stock usar según la etiqueta
        stock = 0
        if fecha_tag == 'dataset':
            # Stock al momento del dataset
            stock = producto.disponible + producto.calidad + producto.stock_externo
        elif fecha_tag == 'inicio':
            # Stock al inicio de la planificación (ya ajustado por demanda entre dataset e inicio)
            if hasattr(producto, 'stock_inicial'):
                stock = producto.stock_inicial
        elif fecha_tag == 'fin':
            # Stock al final de la planificación (inicial + producido)
            if hasattr(producto, 'stock_inicial'):
                stock_planeado = producto.cajas_a_producir if hasattr(producto, 'cajas_a_producir') else 0
                stock = producto.stock_inicial + stock_planeado - (producto.demanda_media * producto.dias_planificacion if hasattr(producto, 'demanda_media') and hasattr(producto, 'dias_planificacion') else 0)
        
        # Calcular ubicaciones ocupadas por este producto
        ubicaciones = stock / cajas_palet
        
        # Solo contar si hay stock positivo
        if stock > 0:
            total_ubicaciones += ubicaciones
            total_cajas += stock
            productos_procesados += 1
    
    return {
        'fecha_tag': fecha_tag,
        'total_ubicaciones': round(total_ubicaciones, 2),
        'total_cajas': round(total_cajas, 2),
        'productos_procesados': productos_procesados
    }
def mostrar_comparativa_ocupacion(productos, dias_planificacion, productos_info=None):
    """
    Calcula y muestra la comparativa de ocupación del almacén en los tres momentos clave.
    
    Args:
        productos: Lista de objetos Producto
        dias_planificacion: Número de días de la planificación
        productos_info: Diccionario con información adicional de productos
        
    Returns:
        dict: Diccionario con los resultados comparativos
    """
    # Asegurarse de que los productos tengan el atributo dias_planificacion
    for producto in productos:
        producto.dias_planificacion = dias_planificacion
    
    # Calcular ocupación en los tres momentos
    ocupacion_dataset = calcular_ocupacion_almacen(productos, 'dataset', productos_info)
    ocupacion_inicio = calcular_ocupacion_almacen(productos, 'inicio', productos_info)
    ocupacion_fin = calcular_ocupacion_almacen(productos, 'fin', productos_info)
    
    # Calcular porcentaje de cambio
    if ocupacion_inicio['total_ubicaciones'] > 0:
        porcentaje_cambio = ((ocupacion_fin['total_ubicaciones'] - ocupacion_inicio['total_ubicaciones']) 
                            / ocupacion_inicio['total_ubicaciones']) * 100
    else:
        porcentaje_cambio = 0
    
    # Preparar resultado
    resultado = {
        'ocupacion_dataset': ocupacion_dataset,
        'ocupacion_inicio': ocupacion_inicio,
        'ocupacion_fin': ocupacion_fin,
        'porcentaje_cambio': round(porcentaje_cambio, 2)
    }
    
    # Crear mensaje para mostrar
    mensaje = f"""
    OCUPACIÓN DEL ALMACÉN:
    ---------------------
    Día dataset:    {ocupacion_dataset['total_ubicaciones']} ubicaciones ({ocupacion_dataset['total_cajas']} cajas)
    Inicio planif.: {ocupacion_inicio['total_ubicaciones']} ubicaciones ({ocupacion_inicio['total_cajas']} cajas)
    Fin planif.:    {ocupacion_fin['total_ubicaciones']} ubicaciones ({ocupacion_fin['total_cajas']} cajas)
    
    La ocupación al final de la planificación es un {'+' if porcentaje_cambio >= 0 else ''}{porcentaje_cambio}% 
    respecto al inicio de la planificación.
    """
    
    # Mostrar mensaje en consola
    print(mensaje)
    
    return resultado

def exportar_resultados(productos_optimizados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    try:
        datos = []
        productos_info, productos_omitir = leer_indicaciones_articulos()
        
        # CAMBIO: Variable para sumar el total de horas planificadas
        total_horas_planificadas = 0.0
        
        # Procesar productos y calcular totales
        total_palets = 0
        total_stock = 0
        
        for producto in productos:
            if producto.cod_art not in productos_omitir:
                estado = "No válido"  # Por defecto
                
                # Verificar si el producto está en productos_optimizados
                producto_opt = next((p for p in productos_optimizados if p.cod_art == producto.cod_art), None)
                
                if producto_opt:
                    if producto_opt.horas_necesarias > 0:
                        estado = "Planificado"
                        # CAMBIO: Acumular horas planificadas para productos con estado "Planificado"
                        total_horas_planificadas += producto_opt.horas_necesarias
                    else:
                        estado = "Válido sin producción"
                    producto_final = producto_opt
                else:
                    producto_final = producto
                
                # Obtener información adicional del producto
                info_producto = productos_info.get(producto.cod_art, {})
                cajas_palet = info_producto.get('cajas_palet', 40)  # Valor por defecto: 40
                
                # Calcular valores individuales
                stock_total = producto_final.stock_inicial + (
                    producto_final.cajas_a_producir if hasattr(producto_final, 'cajas_a_producir') else 0
                )
                palets = stock_total / cajas_palet if cajas_palet > 0 else 0
                
                # Actualizar totales
                total_palets += palets
                total_stock += stock_total
                
                datos.append({
                    'COD_ART': producto_final.cod_art,
                    'NOM_ART': producto_final.nom_art,
                    'COD_GRU': producto_final.cod_gru,
                    'Estado': estado,
                    'Orden_Planificacion': info_producto.get('orden_planificacion', ''),
                    'Demanda_Media': round(producto_final.demanda_media, 2) if producto_final.demanda_media != 'NO VALIDO' else 0,
                    'Stock_Inicial': round(producto_final.stock_inicial, 2),
                    'Cajas_a_Producir': round(producto_final.cajas_a_producir, 2) if hasattr(producto_final, 'cajas_a_producir') else 0,
                    'Horas_Necesarias': round(producto_final.horas_necesarias, 2) if hasattr(producto_final, 'horas_necesarias') else 0,
                    'Cobertura_Inicial': round(producto_final.cobertura_inicial, 2) if producto_final.cobertura_inicial != 'NO VALIDO' else 0,
                    'Cobertura_Final': round(producto_final.cobertura_final_plan, 2) if hasattr(producto_final, 'cobertura_final_plan') else round(producto_final.cobertura_final_est, 2) if producto_final.cobertura_final_est != 'NO VALIDO' else 0,
                    'Cobertura_Final_Est': round(producto_final.cobertura_final_est, 2) if producto_final.cobertura_final_est != 'NO VALIDO' else 0,
                    'Total_Palets': round(palets, 2),
                    'Total_Stock': round(stock_total, 2),
                    'Penalizacion_Espacio': calcular_penalizacion_espacio(total_palets)
                })
        
        # NUEVO: Añadir logging detallado de horas
        logger.info(f"Total de horas planificadas: {total_horas_planificadas:.2f}")
        print("\n==== DETALLE DE HORAS POR PRODUCTO ====")
        for producto_opt in productos_optimizados:
            if hasattr(producto_opt, 'horas_necesarias') and producto_opt.horas_necesarias > 0:
                print(f"Producto {producto_opt.cod_art}: {producto_opt.horas_necesarias:.2f} horas")
        print("==== FIN DETALLE DE HORAS ====\n")
        
        # NUEVO: Calcular y mostrar la ocupación del almacén
        for p in productos:
            p.dias_planificacion = dias_planificacion
        resultado_ocupacion = mostrar_comparativa_ocupacion(productos, dias_planificacion, productos_info)
        
        # Convertir a DataFrame y ordenar
        df = pd.DataFrame(datos)
        df_ordenado = ordenar_productos(df)
        
        # Generar nombre de archivo y exportar
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"
        df_ordenado.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
        # Generar calendario de producción
        productos_planificados = [p for p in productos_optimizados if hasattr(p, 'horas_necesarias') and p.horas_necesarias > 0]
        
        # NUEVO: Logging de productos planificados antes de generar calendario
        print("\n==== PRODUCTOS PARA CALENDARIO ====")
        for p in productos_planificados:
            print(f"Producto {p.cod_art}: {p.horas_necesarias:.2f} horas, {p.cajas_a_producir:.2f} cajas")
        print("==== FIN PRODUCTOS PARA CALENDARIO ====\n")
        
        calendario = generar_calendario_produccion(productos_planificados)
        
        # NUEVO: Validación de horas del calendario
        total_horas_calendario = sum(
            sum(producto['horas'] for producto in dia) 
            for dia in calendario.values()
        )
        
        print("\n==== DETALLE DE HORAS POR DÍA ====")
        for dia, productos in calendario.items():
            horas_dia = sum(p['horas'] for p in productos)
            print(f"Día {dia}: {horas_dia:.2f} horas")
        print(f"Total horas en calendario: {total_horas_calendario:.2f}")
        print("==== FIN DETALLE DE HORAS POR DÍA ====\n")
        
        # Comparar horas de optimización con calendario
        if abs(total_horas_planificadas - total_horas_calendario) > 0.01:
            logger.warning(f"INCONSISTENCIA DE HORAS: Optimización ({total_horas_planificadas:.2f}) vs Calendario ({total_horas_calendario:.2f})")
        
        # Exportar calendario
        nombre_calendario = f"calendario_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}.csv"
        exportar_calendario(calendario, fecha_planificacion, nombre_calendario)
        
        return resultado_ocupacion
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return None
def generar_calendario_produccion(productos_planificados, horas_por_dia=24):
    try:
        # Consolidar productos por código de artículo
        productos_consolidados = {}
        for producto in productos_planificados:
            if not hasattr(producto, 'horas_necesarias') or producto.horas_necesarias <= 0:
                continue
            
            if producto.cod_art not in productos_consolidados:
                productos_consolidados[producto.cod_art] = producto
            else:
                # Sumar horas y cajas de productos con el mismo código
                existente = productos_consolidados[producto.cod_art]
                existente.horas_necesarias += producto.horas_necesarias
                existente.cajas_a_producir += producto.cajas_a_producir

        # Convertir a lista de productos únicos
        productos_planificados = list(productos_consolidados.values())
        
        # Definir orden de planificación
        orden_plan = {"INICIO": 0, "": 1, "FINAL": 2}
        orden_grupo = {"VIME": 0, "MEC": 1}
        
        # Ordenar productos
        sorted_productos = sorted(
            productos_planificados, 
            key=lambda p: (
                orden_plan.get(p.orden_planificacion, 1),
                orden_grupo.get(p.cod_gru, 1),
                p.cobertura_inicial if isinstance(p.cobertura_inicial, (int, float)) else float('inf')
            )
        )
        
        # Inicializar calendario
        calendario = {}
        dia_actual = 1
        horas_disponibles_dia = horas_por_dia
        total_horas_planificadas = 0
        
        for producto in sorted_productos:
            # Redondear horas totales
            horas_pendientes = redondear_media_hora_al_alza(producto.horas_necesarias)
            horas_totales = horas_pendientes
            
            while horas_pendientes > 0:
                # Cambiar de día si no hay horas disponibles
                if horas_disponibles_dia < 2:  # Cambio: requiere al menos 2 horas
                    dia_actual += 1
                    horas_disponibles_dia = horas_por_dia
                
                # Determinar horas a asignar
                horas_a_asignar = min(horas_pendientes, horas_disponibles_dia)
                
                # Asegurar lote mínimo de 2 horas
                if horas_a_asignar < 2 and horas_pendientes > horas_a_asignar:
                    dia_actual += 1
                    horas_disponibles_dia = horas_por_dia
                    continue
                
                # Redondear horas 
                horas_a_asignar = redondear_media_hora_al_alza(horas_a_asignar)
                
                # Calcular proporción de cajas
                proporcion = horas_a_asignar / horas_totales
                cajas_asignadas = round(proporcion * producto.cajas_a_producir)
                
                # Inicializar día si no existe
                if dia_actual not in calendario:
                    calendario[dia_actual] = []
                
                # Añadir al calendario
                calendario[dia_actual].append({
                    'cod_art': producto.cod_art,
                    'nom_art': producto.nom_art,
                    'cod_gru': producto.cod_gru,
                    'horas': horas_a_asignar,
                    'cajas': cajas_asignadas
                })
                
                # Actualizar seguimiento
                horas_pendientes -= horas_a_asignar
                horas_disponibles_dia -= horas_a_asignar
                total_horas_planificadas += horas_a_asignar
        
        # Verificar la asignación total para cada día
        for dia, productos in calendario.items():
            horas_dia = sum(p['horas'] for p in productos)
            logger.info(f"Día {dia}: Total horas asignadas: {horas_dia:.1f}/{horas_por_dia}")
        
        logger.info(f"Calendario generado: {len(calendario)} días de producción, {total_horas_planificadas:.1f} horas totales asignadas")
        return calendario
    
    except Exception as e:
        logger.error(f"Error generando calendario: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return {}

def exportar_calendario(calendario, fecha_inicio, nombre_archivo):
    """
    Exporta el calendario de producción a un archivo CSV
    
    Args:
        calendario (dict): Calendario de producción por días
        fecha_inicio (datetime): Fecha de inicio de la producción
        nombre_archivo (str): Nombre del archivo a generar
    """
    try:
        filas = []
        
        # Convertir fecha_inicio si es string
        if isinstance(fecha_inicio, str):
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%d-%m-%Y')
        elif isinstance(fecha_inicio, date):
            fecha_inicio_dt = datetime.combine(fecha_inicio, datetime.min.time())
        else:
            fecha_inicio_dt = fecha_inicio
        
        # Procesar cada día del calendario
        for dia, productos in calendario.items():
            fecha_actual = fecha_inicio_dt + timedelta(days=dia-1)
            fecha_str = fecha_actual.strftime('%d-%m-%Y')
            
            # Calcular el total de horas del día
            total_horas_dia = sum(producto['horas'] for producto in productos)
            
            # Verificar que el total sea un múltiplo de 0.5
            total_horas_dia = redondear_media_hora_al_alza(total_horas_dia)
            
            for producto in productos:
                # Asegurar que las horas son múltiplos de 0.5
                horas = redondear_media_hora_al_alza(producto['horas'])
                
                filas.append({
                    'Fecha': fecha_str,
                    'Dia': dia,
                    'COD_ART': producto['cod_art'],
                    'NOM_ART': producto['nom_art'],
                    'COD_GRU': producto['cod_gru'],
                    'Horas': round(horas, 1),  # Redondear a 1 decimal (0.5)
                    'Cajas': round(producto['cajas'], 2),
                    'Total_Horas_Dia': round(total_horas_dia, 1)  # Redondear a 1 decimal (0.5)
                })
        
        # Exportar a CSV
        df = pd.DataFrame(filas)
        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Calendario exportado a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando calendario: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        
def calcular_penalizacion_espacio(palets):
    """Calcula la penalización por espacio ocupado"""
    if palets > 1200:
        return -100
    elif palets > 1000:
        return -50
    elif palets > 800:
        return -10
    return 0