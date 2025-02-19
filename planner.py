import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from csv_loader import leer_indicaciones_articulos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calcular_tiempo_cambio(producto_actual, producto_siguiente):
    """
    Calcula el tiempo de cambio entre productos considerando:
    - Cambio entre grupos
    - Limpieza por alérgenos
    - Tiempo base de cambio
    """
    if not producto_actual or not producto_siguiente:
        return 0
        
    tiempo_total = 0
    
    # Tiempo base de cambio
    tiempo_total += producto_siguiente.configuracion.tiempo_cambio
    
    # Cambio entre grupos diferentes
    if producto_actual.cod_gru != producto_siguiente.cod_gru:
        tiempo_total += 30  # 30 minutos adicionales por cambio de grupo
    
    # Limpieza por alérgenos
    if producto_actual.configuracion.grupo_alergenicos:
        tiempo_total += 45  # 45 minutos adicionales por limpieza de alérgenos
    
    return tiempo_total

def ordenar_productos_produccion(productos):
    """
    Ordena los productos para producción considerando:
    1. Orden de planificación (INICIO, intermedio, FINAL)
    2. Productos alergénicos
    3. Prioridad semanal
    4. Cobertura inicial
    """
    def clave_ordenamiento(producto):
        orden_valor = {
            'INICIO': 0,
            '': 1,
            'FINAL': 2
        }.get(producto.configuracion.orden_planificacion, 1)
        
        return (
            orden_valor,
            0 if producto.configuracion.grupo_alergenicos else 1,
            producto.configuracion.prioridad_semanal,
            -producto.cobertura_inicial if isinstance(producto.cobertura_inicial, (int, float)) else 0
        )
    
    return sorted(productos, key=clave_ordenamiento)

def es_dia_valido(producto, dia):
    """
    Verifica si un producto debe producirse en un día específico
    
    Args:
    - producto: Objeto producto
    - dia: Número de día en la planificación
    
    Returns:
    - Boolean indicando si es válido
    """
    orden = producto.configuracion.orden_planificacion
    
    if not orden or orden == '':
        return True
    
    if orden == 'INICIO':
        return dia < 2  # Lunes y martes
    
    if orden == 'FINAL':
        return dia >= 3  # Jueves y viernes
    
    return True

def planificar_dia(productos_validos, horas_disponibles_dia, dia):
    """
    Planifica la producción para un día específico
    
    Args:
    - productos_validos: Lista de productos válidos
    - horas_disponibles_dia: Horas disponibles en el día
    - dia: Número de día en la planificación
    
    Returns:
    - Lista de productos planificados para el día
    """
    tiempo_disponible = horas_disponibles_dia * 60  # Convertir a minutos
    productos_ordenados = ordenar_productos_produccion(productos_validos)
    
    planificacion_dia = []
    producto_actual = None
    tiempo_acumulado = 0
    
    for producto in productos_ordenados:
        # Verificar si es un día válido para este producto
        if not es_dia_valido(producto, dia):
            continue
        
        # Calcular tiempo de cambio
        tiempo_cambio = calcular_tiempo_cambio(producto_actual, producto)
        
        # Tiempo mínimo de producción (2 horas)
        tiempo_minimo_produccion = 120  # 2 horas en minutos
        
        # Verificar tiempo disponible
        tiempo_restante = tiempo_disponible - tiempo_acumulado
        
        if tiempo_restante >= (tiempo_cambio + tiempo_minimo_produccion):
            # Calcular tiempo de producción
            tiempo_produccion = min(
                tiempo_restante - tiempo_cambio,
                (producto.demanda_media * 60 - producto.stock_inicial) / producto.cajas_hora * 60
            )
            
            # Verificar si cumple con el mínimo de producción
            if tiempo_produccion >= tiempo_minimo_produccion:
                cajas_producir = (tiempo_produccion / 60) * producto.cajas_hora
                
                # Verificar capacidad de almacenamiento solo si hay valor válido de cajas_palet
                if producto.configuracion.cajas_palet > 0:
                    ocupacion_nueva = cajas_producir / producto.configuracion.cajas_palet
                    if ocupacion_nueva > producto.configuracion.capacidad_almacen:
                        continue  # Saltamos este producto si excede capacidad
                else:
                    # Si no hay valor de cajas_palet, asumimos que no hay restricción de almacenamiento
                    ocupacion_nueva = 0
                
                # Añadir a planificación
                entrada_planificacion = {
                    'producto': producto,
                    'dia': dia,
                    'cajas': cajas_producir,
                    'tiempo_cambio': tiempo_cambio / 60,  # Convertir a horas
                    'tiempo_produccion': tiempo_produccion / 60,  # Convertir a horas
                    'hora_inicio': tiempo_acumulado / 60,  # Hora de inicio en el día
                    'ocupacion_almacen': ocupacion_nueva
                }
                
                planificacion_dia.append(entrada_planificacion)
                
                # Actualizar tiempos
                tiempo_acumulado += tiempo_cambio + tiempo_produccion
                producto_actual = producto
    
    return planificacion_dia

def optimizar_planificacion_multidia(productos_validos, horas_disponibles, dias_planificacion, dias_cobertura_base):
    """
    Optimiza la planificación considerando múltiples días
    
    Args:
    - productos_validos: Lista de productos válidos
    - horas_disponibles: Horas totales disponibles
    - dias_planificacion: Número de días a planificar
    - dias_cobertura_base: Días mínimos de cobertura requeridos
    
    Returns:
    - Lista de productos planificados
    """
    # Distribuir horas diarias
    horas_dia = horas_disponibles / dias_planificacion
    
    # Planificación global
    planificacion_global = []
    
    # Planificar por día
    for dia in range(dias_planificacion):
        # Crear copia de productos para este día
        productos_dia = [p for p in productos_validos]
        
        # Planificar día
        planificacion_dia = planificar_dia(
            productos_dia, 
            horas_dia, 
            dia
        )
        
        # Actualizar productos y añadir a planificación global
        planificacion_global.extend(planificacion_dia)
        
        # Actualizar stock y demanda
        for entrada in planificacion_dia:
            producto = entrada['producto']
            cajas_producir = entrada['cajas']
            
            # Actualizar stock
            producto.stock_inicial += cajas_producir
            
            # Ajustar demanda y cobertura
            if producto.demanda_media > 0:
                producto.demanda_media -= cajas_producir / dias_planificacion
                producto.cobertura_inicial = producto.stock_inicial / producto.demanda_media
    
    # Procesar productos finales
    for producto in productos_validos:
        # Calcular cobertura final
        if producto.demanda_media > 0:
            producto.cobertura_final_plan = (
                producto.stock_inicial + 
                sum(p['cajas'] for p in planificacion_global if p['producto'].cod_art == producto.cod_art)
            ) / producto.demanda_media
        
        # Calcular cajas producidas
        producto.cajas_a_producir = sum(
            p['cajas'] for p in planificacion_global if p['producto'].cod_art == producto.cod_art
        )
        
        # Calcular horas necesarias
        producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora if producto.cajas_hora > 0 else 0
        
        # Guardar día y hora de planificación
        entradas_producto = [p for p in planificacion_global if p['producto'].cod_art == producto.cod_art]
        if entradas_producto:
            producto.dia_planificado = entradas_producto[0]['dia']
            producto.hora_inicio = entradas_producto[0]['hora_inicio']
    
    return productos_validos

def calcular_formulas(productos, fecha_inicio, fecha_dataset, dias_planificacion, dias_no_habiles, horas_mantenimiento):
    """
    Calcula todas las fórmulas para cada producto y aplica filtros
    
    Args:
    - productos: Lista de productos
    - fecha_inicio: Fecha de inicio de la planificación
    - fecha_dataset: Fecha del dataset
    - dias_planificacion: Número de días a planificar
    - dias_no_habiles: Días no laborables en el periodo
    - horas_mantenimiento: Horas de mantenimiento programadas
    
    Returns:
    - Tuple (productos_validos, horas_disponibles) o (None, None) si hay error
    """
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        configuraciones = leer_indicaciones_articulos()        
        productos_validos = []

        # Convertir fechas
        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%d-%m-%Y')
            # Verificar formato de año en fecha_dataset
            if len(fecha_dataset.split('-')[2]) == 2:
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%y')
            else:
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%Y')
            
            logger.info(f"Fecha inicio: {fecha_inicio_dt}, Fecha dataset: {fecha_dataset_dt}")
        except ValueError as e:
            logger.error(f"Error en formato de fechas: {str(e)}")
            return None, None
        
        for producto in productos:
            # Obtener configuración
            config = configuraciones.get(producto.cod_art)
            if config:
                producto.configuracion = config
            else:
                continue
            
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
                try:
                    of_date = datetime.strptime(producto.primera_of, '%d/%m/%Y')
                    if of_date >= fecha_dataset_dt and of_date < fecha_inicio_dt:
                        producto.disponible = producto.disponible + producto.of
                except ValueError:
                    logger.warning(f"Formato de fecha inválido en primera_of para {producto.cod_art}")
            
            # 5. Stock Inicial
            producto.stock_inicial = (
                producto.disponible + 
                producto.calidad + 
                producto.stock_externo - 
                producto.demanda_provisoria
            )

            if producto.stock_inicial < 0:
                logger.warning(f"Stock inicial negativo para {producto.cod_art}. Ajustando a 0.")
                producto.stock_inicial = 0

            # 6. Cobertura Inicial
            if producto.demanda_media > 0:
                producto.cobertura_inicial = producto.stock_inicial / producto.demanda_media
            else:
                producto.cobertura_inicial = 'NO VALIDO'
            
            # 7. Demanda Periodo
            producto.demanda_periodo = producto.demanda_media * dias_planificacion
            
            # 8. Stock de Seguridad
            producto.stock_seguridad = producto.demanda_media * 3
            
            # 9. Cobertura Final Estimada
            if producto.demanda_media > 0:
                producto.cobertura_final_est = (
                    producto.stock_inicial - producto.demanda_periodo
                ) / producto.demanda_media
            else:
                producto.cobertura_final_est = 'NO VALIDO'

            # Aplicar filtros
            if (producto.configuracion.info_extra not in ['DESCATALOGADO', 'PEDIDO'] and
                producto.vta_60 > 0 and 
                producto.cajas_hora > 0 and 
                producto.demanda_media > 0 and 
                producto.cobertura_inicial != 'NO VALIDO' and 
                producto.cobertura_final_est != 'NO VALIDO'):
                productos_validos.append(producto)

        # Ordenar productos
        productos_validos.sort(key=lambda p: (
            {'INICIO': 0, '': 1, 'FINAL': 2}[p.configuracion.orden_planificacion],
            p.configuracion.prioridad_semanal,
            p.cod_gru,
            -1/p.cobertura_inicial if isinstance(p.cobertura_inicial, (int, float)) else 0
        ))

        logger.info(f"Productos válidos tras filtros: {len(productos_validos)} de {len(productos)}")
        return productos_validos, horas_disponibles
        
    except Exception as e:
        logger.error(f"Error en cálculos: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion, dias_cobertura_base):
    """
    Aplica el método Simplex para optimizar la producción
    """
    try:
        n_productos = len(productos_validos)
        cobertura_minima = dias_cobertura_base + dias_planificacion

        # Función objetivo: priorizar productos con menor cobertura
        coeficientes = []
        for producto in productos_validos:
            if producto.demanda_media > 0:
                prioridad = max(0, 1/producto.cobertura_inicial)
            else:
                prioridad = 0
            coeficientes.append(-prioridad)  # Negativo porque minimizamos

        # Restricción de horas totales disponibles
        A_eq = np.zeros((1, n_productos))
        A_eq[0] = [1 / producto.cajas_hora for producto in productos_validos]
        b_eq = [horas_disponibles]

        # Restricciones de stock mínimo
        A_ub = []
        b_ub = []
        for i, producto in enumerate(productos_validos):
            if producto.demanda_media > 0:
                row = [0] * n_productos
                row[i] = -1  # Coeficiente para el producto actual
                A_ub.append(row)
                stock_min = (producto.demanda_media * cobertura_minima) - producto.stock_inicial
                b_ub.append(-stock_min)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Límites de producción por producto
        bounds = []
        for producto in productos_validos:
            if producto.demanda_media > 0 and producto.cobertura_inicial < 30:
                min_cajas = 2 * producto.cajas_hora  # Mínimo 2 horas de producción
                max_cajas = min(
                    horas_disponibles * producto.cajas_hora,  # No exceder horas disponibles
                    producto.demanda_media * 60 - producto.stock_inicial  # No exceder 60 días
                )
                max_cajas = max(min_cajas, max_cajas)
            else:
                min_cajas = 0
                max_cajas = 0
            bounds.append((min_cajas, max_cajas))

        # Optimización usando método highs
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
            # Asignar resultados a productos
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora
                
                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
            
            # Distribuir en días
            return optimizar_planificacion_multidia(
                productos_validos,
                horas_disponibles,
                dias_planificacion,
                dias_cobertura_base
            )
        else:
            logger.error(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Error en Simplex: {str(e)}")
        return None

def exportar_resultados(productos_optimizados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    """
    Exporta los resultados de la planificación a CSV
    """
    try:
        datos = []
        configuraciones = leer_indicaciones_articulos()
        
        # Procesar todos los productos activos
        for producto in productos:
            if hasattr(producto, 'configuracion') and producto.configuracion.info_extra not in ['DESCATALOGADO', 'PEDIDO']:
                estado = "No válido"  # Estado por defecto
                
                # Buscar producto en optimizados
                producto_opt = next(
                    (p for p in productos_optimizados if p.cod_art == producto.cod_art),
                    None
                )
                
                if producto_opt:
                    estado = "Planificado" if producto_opt.horas_necesarias > 0 else "Válido sin producción"
                    producto_final = producto_opt
                else:
                    producto_final = producto
                
                # Calcular valores seguros
                demanda_media = producto_final.demanda_media if hasattr(producto_final, 'demanda_media') else 0
                stock_inicial = producto_final.stock_inicial if hasattr(producto_final, 'stock_inicial') else 0
                cajas_producir = producto_final.cajas_a_producir if hasattr(producto_final, 'cajas_a_producir') else 0
                horas_necesarias = producto_final.horas_necesarias if hasattr(producto_final, 'horas_necesarias') else 0
                
                # Cálculo seguro de coberturas
                cobertura_inicial = 0
                cobertura_final = 0
                if demanda_media > 0:
                    cobertura_inicial = stock_inicial / demanda_media
                    cobertura_final = (stock_inicial + cajas_producir) / demanda_media
                
                datos.append({
                    'COD_ART': producto_final.cod_art,
                    'NOM_ART': producto_final.nom_art,
                    'Grupo': producto_final.cod_gru,
                    'Estado': estado,
                    'Demanda_Media': round(demanda_media, 2),
                    'Stock_Inicial': round(stock_inicial, 2),
                    'Cajas_a_Producir': round(cajas_producir, 2),
                    'Horas_Necesarias': round(horas_necesarias, 2),
                    'Cobertura_Inicial': round(cobertura_inicial, 2),
                    'Cobertura_Final': round(cobertura_final, 2),
                    'Dia_Planificado': getattr(producto_final, 'dia_planificado', ''),
                    'Hora_Inicio': round(getattr(producto_final, 'hora_inicio', 0), 2)
                })
        
        if not datos:
            raise ValueError("No hay datos para exportar")
        
        # Crear DataFrame y ordenar
        df = pd.DataFrame(datos)
        df['orden_estado'] = df['Estado'].map({
            'Planificado': 0,
            'Válido sin producción': 1,
            'No válido': 2
        })
        df = df.sort_values(['orden_estado', 'Grupo', 'Cobertura_Inicial'])
        df = df.drop('orden_estado', axis=1)
        
        # Generar nombre de archivo y exportar
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"
        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise