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
            # Calcular cajas_hora_reales (85% de las cajas por hora teóricas)
            producto.cajas_hora_reales = producto.cajas_hora * 0.85
            
            # Calcular of_reales (85% de las órdenes de fabricación)
            producto.of_reales = producto.of * 0.85
            
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
                print(f"El stock inicial del producto {producto.cod_art} ({producto.nom_art}) es menor a 0.")
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
                producto.cobertura_inicial = float('inf')  # Valor infinito para productos sin demanda
            
            # 7. Demanda Periodo
            producto.demanda_periodo = producto.demanda_media * dias_planificacion
            
            # 8. Stock de Seguridad (3 días)
            producto.stock_seguridad = producto.demanda_media * 3       

            # 9. Cobertura Final Estimada
            if producto.demanda_media > 0:
                producto.cobertura_final_est = (producto.stock_inicial - producto.demanda_periodo) / producto.demanda_media
            else:
                producto.cobertura_final_est = float('inf')  # Valor infinito para productos sin demanda

            # Asignar orden de planificación desde la información adicional
            if producto.cod_art in productos_info:
                producto.orden_planificacion = productos_info[producto.cod_art].get('orden_planificacion', '')
            else:
                producto.orden_planificacion = ''

            # Aplicar filtros
            if (producto.cod_art not in productos_omitir and
                producto.vta_60 > 0 and 
                producto.cajas_hora > 0 and 
                producto.demanda_media > 0 and 
                producto.cobertura_inicial != float('inf') and 
                producto.cobertura_final_est != float('inf')):
                productos_validos.append(producto)

        logger.info(f"Productos válidos tras filtros: {len(productos_validos)} de {len(productos)}")
        
        # Ordenar productos según criterios de prioridad
        productos_validos = ordenar_productos_para_planificacion(productos_validos)
        
        return productos_validos, horas_disponibles
        
    except Exception as e:
        logger.error(f"Error en cálculos: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def calcular_cobertura_maxima(m_vta_15):
    """
    Calcula la cobertura máxima basada en m_vta_15, siguiendo la tabla de restricciones.
    
    La restricción superior vendrá dada por:
    M_Vta -15        Cobertura_maxima
    >=150            14
    100<=M_Vta -15<150  18
    50<=M_Vta -15<100   20
    25<=M_Vta -15<50    30
    10<=M_Vta -15<25    60
    <10              120
    """
    if m_vta_15 is None or m_vta_15 <= 0:
        return 120.0  # Valor por defecto para casos extremos
    
    if m_vta_15 >= 150:
        return 14.0
    elif 100 <= m_vta_15 < 150:
        return 18.0
    elif 50 <= m_vta_15 < 100:
        return 20.0
    elif 25 <= m_vta_15 < 50:
        return 30.0
    elif 10 <= m_vta_15 < 25:
        return 60.0
    else:  # < 10
        return 120.0

def redondear_media_hora_al_alza(horas):
    """
    Redondea las horas al múltiplo de 0.5 más cercano, siempre hacia arriba.
    
    Ejemplos:
    - 3.23 -> 3.5
    - 6.71 -> 7.0
    - 4.0 -> 4.0
    - 5.5 -> 5.5
    """
    return math.ceil(horas * 2) / 2

def ordenar_productos_para_planificacion(productos):
    """
    Ordena los productos para planificación según estos criterios:
    1. Orden de planificación (INICIO > vacío > FINAL)
    2. Grupo (respetando el orden definido)
    3. Cobertura inicial (ascendente, priorizando cobertura negativa)
    """
    # Definir función para ordenamiento multicriterio
    def clave_orden(p):
        # Orden de planificación: INICIO (0), vacío (1), FINAL (2)
        if p.orden_planificacion == 'INICIO':
            prioridad_orden = 0
        elif not p.orden_planificacion or p.orden_planificacion.strip() == '':
            prioridad_orden = 1
        else:  # FINAL u otros valores
            prioridad_orden = 2
            
        # Prioridad especial a productos con cobertura negativa
        if p.cobertura_final_est < 0:
            prioridad_cobertura = -1  # Alta prioridad para cobertura negativa
        else:
            prioridad_cobertura = p.cobertura_inicial
            
        return (prioridad_orden, p.cod_gru, prioridad_cobertura)
    
    # Ordenar usando la función de clave personalizada
    productos_ordenados = sorted(productos, key=clave_orden)
    
    # Mostrar información de depuración
    logger.info("Orden de productos para planificación:")
    for i, p in enumerate(productos_ordenados[:10]):  # Mostrar solo los primeros 10
        logger.info(f"{i+1}. {p.cod_art} - {p.nom_art} - Orden: {p.orden_planificacion} - Grupo: {p.cod_gru} - Cobertura: {p.cobertura_inicial:.2f} - Final Est: {p.cobertura_final_est:.2f}")
    
    return productos_ordenados

import logging
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import math
from scipy.optimize import linprog
from csv_loader import leer_indicaciones_articulos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aplicar_simplex(productos_validos, horas_disponibles, dias_planificacion, dias_cobertura_base):
    """Aplica el método Simplex para optimizar la producción con mayor tolerancia a problemas de factibilidad"""
    try:
        # Verificar si hay productos válidos
        if not productos_validos:
            logger.warning("No hay productos válidos para optimizar")
            return []
            
        n_productos = len(productos_validos)
        cobertura_minima = dias_cobertura_base + dias_planificacion

        # Identificar productos que realmente necesitan producción
        productos_a_planificar = []
        for i, producto in enumerate(productos_validos):
            # Calcular cobertura máxima basada en la demanda media
            cobertura_maxima = calcular_cobertura_maxima(producto.m_vta_15)
            
            # Calcular mínimo y máximo de cajas a producir
            min_cajas = 2 * producto.cajas_hora_reales  # Mínimo 2 horas de producción
            
            # Máximo de cajas basado en la cobertura máxima
            max_cajas_cobertura = (producto.demanda_media * cobertura_maxima) - producto.stock_inicial
            
            # Si el máximo es negativo (ya tenemos más stock que la cobertura máxima), no planificar
            if max_cajas_cobertura <= 0 and producto.cobertura_final_est >= 0:
                continue  # No necesita producción
                
            # El máximo no puede exceder la capacidad disponible
            max_cajas_capacidad = horas_disponibles * producto.cajas_hora_reales
            
            # Tomar el mínimo entre la cobertura máxima y la capacidad disponible
            max_cajas = min(max_cajas_cobertura, max_cajas_capacidad)
            
            # Si el máximo es menor que el mínimo, considerar ajustes especiales
            if max_cajas < min_cajas:
                if producto.cobertura_final_est < 0:
                    # Prioridad absoluta a productos con cobertura negativa
                    max_cajas = min_cajas
                    productos_a_planificar.append((i, producto.cod_art, min_cajas, max_cajas, True))  # True indica prioridad
                elif producto.cobertura_inicial < 3:
                    # Prioridad a productos con baja cobertura
                    max_cajas = min_cajas
                    productos_a_planificar.append((i, producto.cod_art, min_cajas, max_cajas, False))
            elif max_cajas > 0:
                # Producto normal que necesita producción
                productos_a_planificar.append((i, producto.cod_art, min_cajas, max_cajas, False))
            
            # Log para verificar valores
            logger.info(f"Producto {producto.cod_art}: m_vta_15={producto.m_vta_15:.2f}, cobertura_maxima={cobertura_maxima:.2f}")
            logger.info(f"Producto {producto.cod_art}: min_cajas={min_cajas:.2f}, max_cajas={max_cajas:.2f}")

        # Verificar si hay productos a planificar
        if not productos_a_planificar:
            logger.warning("No hay productos que requieran planificación")
            return productos_validos
            
        logger.info(f"Productos a planificar: {len(productos_a_planificar)}")
        for idx, cod_art, min_c, max_c, es_prioritario in productos_a_planificar:
            prioridad = "PRIORITARIO" if es_prioritario else ""
            logger.info(f"  {idx}: {cod_art} - min: {min_c:.2f}, max: {max_c:.2f} {prioridad}")

        # Verificar si hay demasiadas restricciones conflictivas
        total_min_horas = sum(min_c/productos_validos[idx].cajas_hora_reales 
                            for idx, _, min_c, _, _ in productos_a_planificar)
        
        logger.info(f"Total horas mínimas requeridas: {total_min_horas:.2f}")
        logger.info(f"Horas disponibles: {horas_disponibles:.2f}")
        
        if total_min_horas > horas_disponibles * 1.05:  # Permitir un pequeño margen
            logger.warning(f"Restricciones conflictivas: Total horas mínimas ({total_min_horas:.2f}) > Horas disponibles ({horas_disponibles:.2f})")
            logger.info("Realizando asignación manual basada en prioridades")
            
            # Ordenar productos por prioridad
            productos_a_planificar.sort(key=lambda p: (
                0 if p[4] else 1,  # Primero los prioritarios
                productos_validos[p[0]].cobertura_inicial  # Luego por cobertura
            ))
            
            # Asignar horas manualmente
            horas_restantes = horas_disponibles
            for idx, cod_art, min_cajas, max_cajas, _ in productos_a_planificar:
                producto = productos_validos[idx]
                
                # Si es un producto prioritario o tenemos suficientes horas
                min_horas = min_cajas / producto.cajas_hora_reales
                
                if min_horas <= horas_restantes:
                    # Podemos asignar al menos el mínimo
                    horas_asignadas = min_horas
                    horas_restantes -= horas_asignadas
                else:
                    # No podemos cumplir ni el mínimo, asignar lo que queda
                    horas_asignadas = 0 if horas_restantes < 1 else horas_restantes
                    horas_restantes = 0
                
                # Redondear a media hora
                horas_asignadas = redondear_media_hora_al_alza(horas_asignadas)
                
                # Calcular cajas y actualizar producto
                producto.horas_necesarias = horas_asignadas
                producto.cajas_a_producir = round(horas_asignadas * producto.cajas_hora_reales)
                
                if producto.demanda_media > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
                
                logger.info(f"Asignación manual: {producto.cod_art} - {horas_asignadas:.2f} horas - {producto.cajas_a_producir} cajas")
                
                if horas_restantes <= 0:
                    break
            
            logger.info(f"Asignación manual completada. Horas restantes: {horas_restantes:.2f}")
            return productos_validos
            
        # Si llegamos aquí, intentamos la optimización lineal
        # Preparar los parámetros para el método simplex
        coeficientes = np.zeros(n_productos)
        bounds = [(0, 0) for _ in range(n_productos)]
        
        # Llenar solo para productos a planificar
        for i, cod_art, min_cajas, max_cajas, es_prioritario in productos_a_planificar:
            if es_prioritario:
                # Dar mucho peso a productos prioritarios
                coeficientes[i] = -100
            elif productos_validos[i].cobertura_final_est < 0:
                # Prioridad a productos con cobertura negativa
                coeficientes[i] = -10
            else:
                # Prioridad inversamente proporcional a la cobertura
                coeficientes[i] = -1 / max(1, productos_validos[i].cobertura_inicial)
            
            bounds[i] = (min_cajas, max_cajas)
        
        # Definir restricción de horas totales como desigualdad en lugar de igualdad
        # Esto da más flexibilidad al solver
        A_ub = np.zeros((1, n_productos))
        A_ub[0] = [1 / producto.cajas_hora_reales for producto in productos_validos]
        b_ub = [horas_disponibles]
        
        try:
            # Intentar optimización con restricción de horas como desigualdad
            result = linprog(
                c=coeficientes,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs'
            )
            
            if not result.success:
                # Si falla, intentar sin ninguna restricción excepto las bounds
                logger.warning(f"Primera optimización falló: {result.message}. Intentando sin restricciones globales.")
                result = linprog(
                    c=coeficientes,
                    bounds=bounds,
                    method='highs'
                )
        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            logger.warning("Implementando asignación manual simplificada")
            
            # Asignar solo los mínimos para productos prioritarios
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = 0
                producto.horas_necesarias = 0
            
            # Encontrar productos prioritarios
            productos_prioritarios = [p for p in productos_a_planificar if p[4]]
            if not productos_prioritarios:
                # Si no hay prioritarios, usar los de menor cobertura
                productos_a_planificar.sort(key=lambda p: productos_validos[p[0]].cobertura_inicial)
                productos_prioritarios = productos_a_planificar[:min(5, len(productos_a_planificar))]
            
            # Asignar horas
            horas_restantes = horas_disponibles
            for idx, cod_art, min_cajas, _, _ in productos_prioritarios:
                producto = productos_validos[idx]
                min_horas = min_cajas / producto.cajas_hora_reales
                
                if min_horas <= horas_restantes:
                    producto.horas_necesarias = redondear_media_hora_al_alza(min_horas)
                    producto.cajas_a_producir = round(producto.horas_necesarias * producto.cajas_hora_reales)
                    horas_restantes -= producto.horas_necesarias
                else:
                    break
            
            # Crear un resultado ficticio
            class DummyResult:
                def __init__(self):
                    self.success = True
                    self.x = np.zeros(n_productos)
            
            result = DummyResult()
            for i, producto in enumerate(productos_validos):
                result.x[i] = producto.cajas_a_producir
            
        if result.success:
            # Procesar resultados
            horas_producidas = 0
            
            for i, producto in enumerate(productos_validos):
                producto.cajas_a_producir = max(0, round(result.x[i]))
                producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora_reales
                
                # Redondear las horas necesarias a tramos de media hora al alza
                if producto.horas_necesarias > 0:
                    producto.horas_necesarias = redondear_media_hora_al_alza(producto.horas_necesarias)
                    # Recalcular las cajas a producir basadas en las horas redondeadas
                    producto.cajas_a_producir = round(producto.horas_necesarias * producto.cajas_hora_reales)
                
                horas_producidas += producto.horas_necesarias
                
                # Calcular cobertura final
                if producto.demanda_media > 0 and producto.cajas_a_producir > 0:
                    producto.cobertura_final_plan = (
                        producto.stock_inicial + producto.cajas_a_producir
                    ) / producto.demanda_media
            
            # Verificar que no se superen las horas disponibles
            if horas_producidas > horas_disponibles * 1.05:  # Permitir un pequeño margen
                logger.warning(f"Horas planificadas ({horas_producidas:.2f}) superan las horas disponibles ({horas_disponibles:.2f}).")
                
                # Ajustar el plan para no exceder las horas disponibles
                exceso_horas = horas_producidas - horas_disponibles
                
                # Ordenar productos por prioridad (menor cobertura = mayor prioridad)
                productos_ordenados = sorted(
                    [(i, p) for i, p in enumerate(productos_validos) if p.horas_necesarias > 0], 
                    key=lambda x: (
                        0 if x[1].cobertura_final_est < 0 else 1,  # Primero negativos
                        x[1].cobertura_inicial  # Luego por cobertura inicial
                    )
                )
                
                # Calcular cuánto reducir de cada producto (proporcionalmente)
                total_a_reducir = exceso_horas
                for i, producto in reversed(productos_ordenados):
                    # Saltamos productos prioritarios (cobertura negativa)
                    if producto.cobertura_final_est < 0:
                        continue
                        
                    # Cálculo proporcional
                    factor_reduccion = producto.horas_necesarias / horas_producidas
                    reduccion_propuesta = exceso_horas * factor_reduccion
                    
                    # Asegurar que no quede por debajo del mínimo de 2 horas
                    reduccion_maxima = max(0, producto.horas_necesarias - 2.0)
                    reduccion = min(reduccion_propuesta, reduccion_maxima)
                    
                    # Aplicar reducción
                    if reduccion > 0:
                        producto.horas_necesarias -= reduccion
                        producto.horas_necesarias = redondear_media_hora_al_alza(producto.horas_necesarias)
                        producto.cajas_a_producir = round(producto.horas_necesarias * producto.cajas_hora_reales)
                        total_a_reducir -= reduccion
                        
                        if producto.demanda_media > 0:
                            producto.cobertura_final_plan = (
                                producto.stock_inicial + producto.cajas_a_producir
                            ) / producto.demanda_media
                    
                    if total_a_reducir <= 0:
                        break
                
                # Recalcular horas totales
                horas_producidas = sum(p.horas_necesarias for p in productos_validos)
                
                logger.info(f"Plan ajustado: {horas_producidas:.2f} horas planificadas.")
            
            # Resumen final
            horas_por_grupo = {}
            for p in productos_validos:
                if p.horas_necesarias > 0:
                    horas_por_grupo[p.cod_gru] = horas_por_grupo.get(p.cod_gru, 0) + p.horas_necesarias
            
            for grupo, horas in horas_por_grupo.items():
                logger.info(f"Grupo {grupo}: {horas:.2f} horas")
                
            logger.info(f"Optimización exitosa - Horas planificadas: {horas_producidas:.2f}/{horas_disponibles:.2f}")
            return productos_validos
        else:
            logger.error(f"Error en optimización: {result.message}")
            return None

    except Exception as e:
        logger.error(f"Error en Simplex: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
def calcular_cobertura_maxima(m_vta_15):
    """
    Calcula la cobertura máxima basada en m_vta_15, siguiendo la tabla de restricciones.
    
    La restricción superior vendrá dada por:
    M_Vta -15        Cobertura_maxima
    >=150            14
    100<=M_Vta -15<150  18
    50<=M_Vta -15<100   20
    25<=M_Vta -15<50    30
    10<=M_Vta -15<25    60
    <10              120
    """
    if m_vta_15 is None or m_vta_15 <= 0:
        return 120.0  # Valor por defecto para casos extremos
    
    if m_vta_15 >= 150:
        return 14.0
    elif 100 <= m_vta_15 < 150:
        return 18.0
    elif 50 <= m_vta_15 < 100:
        return 20.0
    elif 25 <= m_vta_15 < 50:
        return 30.0
    elif 10 <= m_vta_15 < 25:
        return 60.0
    else:  # < 10
        return 120.0

def redondear_media_hora_al_alza(horas):
    """
    Redondea las horas al múltiplo de 0.5 más cercano, siempre hacia arriba.
    
    Ejemplos:
    - 3.23 -> 3.5
    - 6.71 -> 7.0
    - 4.0 -> 4.0
    - 5.5 -> 5.5
    """
    return math.ceil(horas * 2) / 2

def optimizar_orden_grupos(productos):
    """
    Optimiza el orden de los productos minimizando el tiempo perdido en cambios
    entre grupos MEC y VIME.
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
    df['orden_planificacion'] = df['Orden_Planificacion'].fillna('').map({
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        stock_total = producto.stock_inicial + getattr(producto, 'cajas_a_producir', 0)
        
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

def calcular_penalizacion_espacio(palets):
    """Calcula la penalización por espacio ocupado"""
    if palets > 1200:
        return -100
    elif palets > 1000:
        return -50
    elif palets > 800:
        return -10
    return 0

def exportar_resultados(productos_optimizados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    try:
        datos = []
        productos_info, productos_omitir = leer_indicaciones_articulos()
        
        # Procesar productos y calcular totales
        total_palets = 0
        total_stock = 0
        
        for producto in productos:
            if producto.cod_art not in productos_omitir:
                estado = "No válido"  # Por defecto
                
                # Verificar si el producto está en productos_optimizados
                producto_opt = next((p for p in productos_optimizados if p.cod_art == producto.cod_art), None)
                
                if producto_opt:
                    if hasattr(producto_opt, 'horas_necesarias') and producto_opt.horas_necesarias > 0:
                        estado = "Planificado"
                    else:
                        estado = "Válido sin producción"
                    producto_final = producto_opt
                else:
                    producto_final = producto
                
                # Obtener información adicional del producto
                info_producto = productos_info.get(producto.cod_art, {})
                cajas_palet = info_producto.get('cajas_palet', 40)  # Valor por defecto: 40
                orden_planificacion = info_producto.get('orden_planificacion', '')
                
                # Calcular valores individuales
                cajas_a_producir = getattr(producto_final, 'cajas_a_producir', 0)
                horas_necesarias = getattr(producto_final, 'horas_necesarias', 0)
                
                stock_total = producto_final.stock_inicial + cajas_a_producir
                palets = stock_total / cajas_palet if cajas_palet > 0 else 0
                
                # Actualizar totales
                total_palets += palets
                total_stock += stock_total
                
                # Preparar cobertura final
                if hasattr(producto_final, 'cobertura_final_plan'):
                    cobertura_final = producto_final.cobertura_final_plan
                elif hasattr(producto_final, 'cobertura_final_est'):
                    cobertura_final = producto_final.cobertura_final_est
                else:
                    cobertura_final = 0
                
                # Para la cobertura final estimada, usar el valor calculado o un valor por defecto
                cobertura_final_est = getattr(producto_final, 'cobertura_final_est', 0)
                if cobertura_final_est == float('inf'):
                    cobertura_final_est = 0
                
                datos.append({
                    'COD_ART': producto_final.cod_art,
                    'NOM_ART': producto_final.nom_art,
                    'COD_GRU': producto_final.cod_gru,
                    'Estado': estado,
                    'Orden_Planificacion': orden_planificacion,
                    'Demanda_Media': round(producto_final.demanda_media, 2) if producto_final.demanda_media != float('inf') else 0,
                    'Stock_Inicial': round(producto_final.stock_inicial, 2),
                    'Cajas_a_Producir': cajas_a_producir,
                    'Horas_Necesarias': round(horas_necesarias, 2),
                    'Cobertura_Inicial': round(producto_final.cobertura_inicial, 2) if producto_final.cobertura_inicial != float('inf') else 0,
                    'Cobertura_Final': round(cobertura_final, 2) if cobertura_final != float('inf') else 0,
                    'Cobertura_Final_Est': round(cobertura_final_est, 2),
                    'Total_Palets': round(palets, 2),
                    'Total_Stock': round(stock_total, 2),
                    'Penalizacion_Espacio': calcular_penalizacion_espacio(total_palets)
                })
    except Exception as e:
        logger.error(f"Error al exportar resultados: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None