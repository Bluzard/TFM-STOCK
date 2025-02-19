import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import linprog
from csv_loader import leer_indicaciones_articulos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ordenar_productos_planificacion(productos_validos):
    """
    Ordena los productos según la lógica de priorización:
    1. Orden de planificación (INICIO, intermedio, FINAL)
    2. Productos alergénicos (primero)
    3. Grupo (COD_GRU)
    4. Prioridad semanal
    5. Cobertura (menor primero)
    """
    def clave_ordenamiento(producto):
        # Orden de planificación
        orden_valor = {
            'INICIO': 0,
            '': 1,
            'FINAL': 2
        }.get(producto.configuracion.orden_planificacion, 1)
        
        # Productos alergénicos primero en su grupo
        alergenicos_valor = 0 if producto.configuracion.grupo_alergenicos else 1
        
        return (
            orden_valor,
            alergenicos_valor,
            producto.cod_gru,
            producto.configuracion.prioridad_semanal,
            producto.cobertura_inicial if isinstance(producto.cobertura_inicial, (int, float)) else float('inf')
        )
    
    return sorted(productos_validos, key=clave_ordenamiento)

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

def calcular_formulas(productos, fecha_inicio, fecha_dataset, dias_planificacion, dias_no_habiles, horas_mantenimiento):
    """Calcula todas las fórmulas para cada producto y aplica filtros"""
    try:
        # 1. Cálculo de Horas Disponibles
        horas_disponibles = 24 * (dias_planificacion - dias_no_habiles) - horas_mantenimiento
        
        configuraciones = leer_indicaciones_articulos()        
        productos_validos = []

        # Convertir fechas usando el formato correcto
        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%d-%m-%Y')
            if len(fecha_dataset.split('-')[2]) == 2:  # Si el año tiene 2 dígitos
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%y')
            else:  # Si el año tiene 4 dígitos
                fecha_dataset_dt = datetime.strptime(fecha_dataset, '%d-%m-%Y')
            
            logger.info(f"Fecha inicio: {fecha_inicio_dt}, Fecha dataset: {fecha_dataset_dt}")
        except ValueError as e:
            logger.error(f"Error en formato de fechas: {str(e)}")
            return None, None
        
        for producto in productos:
            # Obtener configuración del producto
            config = configuraciones.get(producto.cod_art)
            if config:
                producto.configuracion = config
            else:
                continue  # Saltamos productos sin configuración
            
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

            # Alerta stock inicial negativo
            if producto.stock_inicial < 0:
                print("\n⚠️  ALERTA: STOCK INICIAL NEGATIVO ⚠️")
                print(f"Producto: {producto.cod_art} - {producto.nom_art}")
                print("El stock inicial del producto es menor a 0.")
                print("🔹 Se recomienda adelantar la planificación para evitar problemas.\n")
                
                respuesta = input("¿Desea continuar de todos modos? (s/n): ").strip().lower()
                if respuesta != 's':
                    print("⛔ Proceso interrumpido por el usuario.")
                    exit()
                print("✅ Continuando con la ejecución...")

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
            if (producto.configuracion.info_extra not in ['DESCATALOGADO', 'PEDIDO'] and
                producto.vta_60 > 0 and 
                producto.cajas_hora > 0 and 
                producto.demanda_media > 0 and 
                producto.cobertura_inicial != 'NO VALIDO' and 
                producto.cobertura_final_est != 'NO VALIDO'):
                productos_validos.append(producto)

        # Ordenar productos según prioridad
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
        return None, None


def planificar_produccion_diaria(productos_validos, horas_disponibles_dia):
    """Planifica la producción de un día considerando prioridades y tiempos de cambio"""
    tiempo_disponible = horas_disponibles_dia * 60  # Convertir a minutos
    productos_ordenados = ordenar_productos_planificacion(productos_validos)
    
    planificacion = []
    producto_actual = None
    tiempo_acumulado = 0
    
    for producto in productos_ordenados:
        if producto.demanda_media <= 0:
            continue
            
        # Calcular tiempo de cambio
        tiempo_cambio = calcular_tiempo_cambio(producto_actual, producto)
        
        # Verificar tiempo mínimo necesario (cambio + 2 horas producción)
        tiempo_minimo = tiempo_cambio + (2 * 60)
        tiempo_restante = tiempo_disponible - tiempo_acumulado
        
        if tiempo_restante >= tiempo_minimo:
            # Calcular tiempo de producción disponible
            tiempo_produccion = min(
                tiempo_restante - tiempo_cambio,
                (producto.demanda_media * 60 - producto.stock_inicial) / producto.cajas_hora * 60
            )
            
            # Asegurar mínimo 2 horas de producción
            if tiempo_produccion >= 120:  # 120 minutos = 2 horas
                cajas_producir = (tiempo_produccion / 60) * producto.cajas_hora
                
                # Verificar capacidad de almacenamiento
                ocupacion_nueva = cajas_producir / producto.configuracion.cajas_palet
                if ocupacion_nueva <= producto.configuracion.capacidad_almacen:
                    planificacion.append({
                        'producto': producto,
                        'cajas': cajas_producir,
                        'tiempo_cambio': tiempo_cambio,
                        'tiempo_produccion': tiempo_produccion,
                        'inicio': tiempo_acumulado + tiempo_cambio,
                        'ocupacion': ocupacion_nueva
                    })
                    
                    tiempo_acumulado += tiempo_cambio + tiempo_produccion
                    producto_actual = producto
    
    return planificacion

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
                
                if producto.cajas_hora > 0:
                    producto.horas_necesarias = producto.cajas_a_producir / producto.cajas_hora
                else:
                    producto.horas_necesarias = 0
                
                horas_producidas += producto.horas_necesarias
                               
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def es_dia_valido(producto, fecha):
    """Verifica si el producto debe planificarse en la fecha dada"""
    orden = producto.configuracion.orden_planificacion
    dia_semana = fecha.weekday()
    
    if orden == 'INICIO':
        return dia_semana < 2  # Lunes y martes
    elif orden == 'FINAL':
        return dia_semana >= 3  # Jueves y viernes
    return True  # Productos sin orden específico
def exportar_resultados(productos_optimizados, productos, fecha_dataset, fecha_planificacion, dias_planificacion, dias_cobertura_base):
    """Exporta los resultados de la planificación a un archivo CSV"""
    try:
        logger.info("Iniciando exportación de resultados")
        datos = []
        configuraciones = leer_indicaciones_articulos()
        
        # Obtener todos los productos activos
        for producto in productos:
            try:
                logger.debug(f"Procesando producto para exportación: {producto.cod_art}")
                
                if hasattr(producto, 'configuracion') and producto.configuracion.info_extra not in ['DESCATALOGADO', 'PEDIDO']:
                    estado = "No válido"
                    
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
                        
                    logger.debug(f"Estado del producto: {estado}")
                    
                    # Valores seguros con logging
                    demanda_media = 0
                    if hasattr(producto_final, 'demanda_media'):
                        demanda_media = float(producto_final.demanda_media) if isinstance(producto_final.demanda_media, (int, float)) else 0
                    logger.debug(f"Demanda media: {demanda_media}")
                    
                    stock_inicial = getattr(producto_final, 'stock_inicial', 0)
                    logger.debug(f"Stock inicial: {stock_inicial}")
                    
                    cajas_producir = getattr(producto_final, 'cajas_a_producir', 0)
                    logger.debug(f"Cajas a producir: {cajas_producir}")
                    
                    # Cálculo seguro de coberturas con logging
                    cobertura_inicial = 0
                    cobertura_final = 0
                    try:
                        if demanda_media > 0:
                            cobertura_inicial = stock_inicial / demanda_media
                            cobertura_final = (stock_inicial + cajas_producir) / demanda_media
                            logger.debug(f"Coberturas calculadas - Inicial: {cobertura_inicial}, Final: {cobertura_final}")
                        else:
                            logger.warning(f"No se puede calcular cobertura para {producto_final.cod_art} (demanda_media = 0)")
                    except Exception as e:
                        logger.error(f"Error calculando coberturas para {producto_final.cod_art}: {str(e)}")
                    
                    datos.append({
                        'COD_ART': producto_final.cod_art,
                        'NOM_ART': producto_final.nom_art,
                        'Estado': estado,
                        'Demanda_Media': round(demanda_media, 2),
                        'Stock_Inicial': round(stock_inicial, 2),
                        'Cajas_a_Producir': round(cajas_producir, 2),
                        'Cobertura_Inicial': round(cobertura_inicial, 2),
                        'Cobertura_Final': round(cobertura_final, 2)
                    })
                    
            except Exception as e:
                logger.error(f"Error procesando producto {producto.cod_art}: {str(e)}")
                continue
        
        logger.info(f"Productos procesados: {len(datos)}")
        
        if not datos:
            raise ValueError("No hay datos para exportar")
            
        df = pd.DataFrame(datos)
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"
        
        df.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def calcular_metricas_planificacion(planificacion):
    """Calcula métricas globales de la planificación"""
    total_tiempo_cambio = sum(plan['tiempo_cambio'] for plan in planificacion)
    total_tiempo_produccion = sum(plan['tiempo_produccion'] for plan in planificacion)
    total_cajas = sum(plan['cajas'] for plan in planificacion)
    
    logger.info(f"""
    Métricas de planificación:
    - Tiempo total de cambios: {total_tiempo_cambio/60:.2f} horas
    - Tiempo total de producción: {total_tiempo_produccion/60:.2f} horas
    - Total cajas planificadas: {total_cajas:.0f}
    - Eficiencia: {(total_tiempo_produccion/(total_tiempo_produccion+total_tiempo_cambio))*100:.2f}%
    """)
