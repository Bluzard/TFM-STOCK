import logging
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
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
                        stock_previsto += producto.of

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
                    if producto_opt.horas_necesarias > 0:
                        estado = "Planificado"
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
                    'Cajas_a_Producir': producto_final.cajas_a_producir if hasattr(producto_final, 'cajas_a_producir') else 0,
                    'Horas_Necesarias': round(producto_final.horas_necesarias, 2) if hasattr(producto_final, 'horas_necesarias') else 0,
                    'Cobertura_Inicial': round(producto_final.cobertura_inicial, 2) if producto_final.cobertura_inicial != 'NO VALIDO' else 0,
                    'Cobertura_Final': round(producto_final.cobertura_final_plan, 2) if hasattr(producto_final, 'cobertura_final_plan') else producto_final.cobertura_final_est,
                    'Cobertura_Final_Est': round(producto_final.cobertura_final_est, 2) if producto_final.cobertura_final_est != 'NO VALIDO' else 0,
                    'Total_Palets': round(palets, 2),
                    'Total_Stock': round(stock_total, 2),
                    'Penalizacion_Espacio': calcular_penalizacion_espacio(total_palets)
                })
        
        # Convertir a DataFrame y ordenar
        df = pd.DataFrame(datos)
        df_ordenado = ordenar_productos(df)
        
        # Generar nombre de archivo y exportar
        nombre_archivo = f"planificacion_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}_dp{dias_planificacion}_cmin{dias_cobertura_base}.csv"
        df_ordenado.to_csv(nombre_archivo, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Resultados exportados a {nombre_archivo}")
        
    except Exception as e:
        logger.error(f"Error exportando resultados: {str(e)}")
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