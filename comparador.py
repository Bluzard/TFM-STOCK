import os
import logging
import pandas as pd
import json
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

def procesar_horas_string(valor_str):
    """
    Procesa un string de horas con formato posiblemente irregular y lo convierte a float.
    
    Args:
        valor_str: String con valor de horas
        
    Returns:
        float: Valor de horas convertido a float
    """
    try:
        # Si es nulo o vacío
        if pd.isna(valor_str) or str(valor_str).strip() == '':
            return 0.0
            
        # Convertir a string si no lo es
        valor_str = str(valor_str)
            
        # Reemplazar comas por puntos
        valor_str = valor_str.replace(',', '.')
        
        # Si hay múltiples valores separados por espacios, sumarlos
        if ' ' in valor_str:
            return sum(float(v) for v in valor_str.split(' ') if v.strip())
            
        # Si hay caracteres que no son números, puntos, o signos +/-
        valor_limpio = ''.join(c for c in valor_str if c.isdigit() or c in ['.', '+', '-'])
        
        # Convertir a float
        return float(valor_limpio) if valor_limpio else 0.0
    except Exception as e:
        logger.warning(f"Error procesando valor de horas '{valor_str}': {str(e)}")
        return 0.0

def cargar_planning_propuesto(fecha_inicio):
    """
    Carga el planning propuesto para una fecha específica.
    
    Args:
        fecha_inicio: Fecha de inicio del planning en formato datetime
        
    Returns:
        DataFrame con el planning propuesto o None si no existe
    """
    try:
        # Formatear la fecha para buscar el archivo
        fecha_str = fecha_inicio.strftime('%d-%m-%y')
        nombre_archivo = f"Planning propuesto {fecha_str}.csv"
        
        # Verificar si existe el archivo
        if not os.path.exists(nombre_archivo):
            logger.info(f"No se encontró el archivo de planning propuesto: {nombre_archivo}")
            return None
            
        # Cargar el archivo CSV sin conversión automática de tipos
        df_propuesto = pd.read_csv(
            nombre_archivo, 
            sep=';', 
            encoding='latin1'
        )
        
        # Convertir columnas numéricas manualmente
        if 'Horas' in df_propuesto.columns:
            df_propuesto['Horas_float'] = df_propuesto['Horas'].apply(procesar_horas_string)
        
        if 'Cajas' in df_propuesto.columns:
            df_propuesto['Cajas_float'] = df_propuesto['Cajas'].apply(procesar_horas_string)
            
        logger.info(f"Planning propuesto cargado: {len(df_propuesto)} registros")
        
        return df_propuesto
    except Exception as e:
        logger.error(f"Error cargando planning propuesto: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def comparar_calendarios(df_propuesto, df_calendario, fecha_inicio, fecha_dataset):
    """
    Compara el planning propuesto con el calendario generado y muestra gráficas.
    
    Args:
        df_propuesto: DataFrame con el planning propuesto
        df_calendario: DataFrame con el calendario generado
        fecha_inicio: Fecha de inicio del planning
        fecha_dataset: Fecha del dataset
        
    Returns:
        dict: Diccionario con información de comparación o None si no se pudo realizar
    """
    try:
        if df_propuesto is None:
            logger.info("No hay planning propuesto para comparar")
            return None
            
        # Verificar que df_calendario tiene contenido
        if df_calendario is None or len(df_calendario) == 0:
            logger.info("No hay calendario generado para comparar")
            return None
        
        # Normalizar columnas para asegurar compatibilidad
        # En caso de que las columnas tengan nombres diferentes
        columnas_requeridas = ['Dia', 'COD_ART', 'NOM_ART', 'COD_GRU', 'Horas', 'Cajas']
        
        # Para df_propuesto
        # Usar Horas_float si existe
        if 'Horas_float' in df_propuesto.columns:
            df_propuesto['Horas'] = df_propuesto['Horas_float']
            
        # Usar Cajas_float si existe
        if 'Cajas_float' in df_propuesto.columns:
            df_propuesto['Cajas'] = df_propuesto['Cajas_float']
            
        # Verificar columnas requeridas
        for col in columnas_requeridas:
            if col not in df_propuesto.columns:
                # Buscar columnas similares
                col_similar = None
                for c in df_propuesto.columns:
                    if col.lower() in c.lower():
                        col_similar = c
                        break
                
                if col_similar:
                    df_propuesto = df_propuesto.rename(columns={col_similar: col})
                else:
                    # Si no existe una columna similar pero es Cajas, se puede calcular
                    if col == 'Cajas' and 'Horas' in df_propuesto.columns and 'COD_ART' in df_propuesto.columns:
                        # Asumir 10 cajas/hora como valor predeterminado si no existe
                        df_propuesto['Cajas'] = df_propuesto['Horas'] * 10
                    else:
                        logger.warning(f"Columna {col} no encontrada en planning propuesto")
                        
        # Para df_calendario
        for col in columnas_requeridas:
            if col not in df_calendario.columns:
                # Buscar columnas similares
                col_similar = None
                for c in df_calendario.columns:
                    if col.lower() in c.lower():
                        col_similar = c
                        break
                
                if col_similar:
                    df_calendario = df_calendario.rename(columns={col_similar: col})
                else:
                    logger.warning(f"Columna {col} no encontrada en calendario generado")
        
        # Asegurar que las columnas numéricas son realmente numéricas
        for df in [df_propuesto, df_calendario]:
            for col in ['Horas', 'Cajas']:
                if col in df.columns:
                    # Convertir a numérico si no lo es
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].apply(procesar_horas_string)
        
        # 1. Resumen de productos planificados
        productos_propuesto = df_propuesto['COD_ART'].nunique() if 'COD_ART' in df_propuesto.columns else 0
        productos_calendario = df_calendario['COD_ART'].nunique() if 'COD_ART' in df_calendario.columns else 0
        
        # 2. Resumen de horas totales
        horas_propuesto = float(df_propuesto['Horas'].sum()) if 'Horas' in df_propuesto.columns else 0.0
        horas_calendario = float(df_calendario['Horas'].sum()) if 'Horas' in df_calendario.columns else 0.0
        
        # 3. Resumen de distribución por días
        if 'Dia' in df_propuesto.columns and 'Horas' in df_propuesto.columns:
            horas_dia_propuesto = {str(k): float(v) for k, v in df_propuesto.groupby('Dia')['Horas'].sum().to_dict().items()}
        else:
            horas_dia_propuesto = {}
            
        if 'Dia' in df_calendario.columns and 'Horas' in df_calendario.columns:
            horas_dia_calendario = {str(k): float(v) for k, v in df_calendario.groupby('Dia')['Horas'].sum().to_dict().items()}
        else:
            horas_dia_calendario = {}
            
        # 4. Resumen de distribución por grupos
        if 'COD_GRU' in df_propuesto.columns and 'Horas' in df_propuesto.columns:
            horas_grupo_propuesto = {str(k): float(v) for k, v in df_propuesto.groupby('COD_GRU')['Horas'].sum().to_dict().items()}
        else:
            horas_grupo_propuesto = {}
            
        if 'COD_GRU' in df_calendario.columns and 'Horas' in df_calendario.columns:
            horas_grupo_calendario = {str(k): float(v) for k, v in df_calendario.groupby('COD_GRU')['Horas'].sum().to_dict().items()}
        else:
            horas_grupo_calendario = {}
        
        # 5. Productos en común y diferentes
        productos_propuesto_set = set(df_propuesto['COD_ART']) if 'COD_ART' in df_propuesto.columns else set()
        productos_calendario_set = set(df_calendario['COD_ART']) if 'COD_ART' in df_calendario.columns else set()
        
        productos_comunes = productos_propuesto_set.intersection(productos_calendario_set)
        productos_solo_propuesto = productos_propuesto_set - productos_calendario_set
        productos_solo_calendario = productos_calendario_set - productos_propuesto_set
        
        # 6. Horas por producto en común
        horas_productos_comunes = {}
        for producto in productos_comunes:
            try:
                # Cálculo de horas para cada producto
                horas_propuesto_producto = float(df_propuesto[df_propuesto['COD_ART'] == producto]['Horas'].sum()) if 'COD_ART' in df_propuesto.columns and 'Horas' in df_propuesto.columns else 0.0
                horas_calendario_producto = float(df_calendario[df_calendario['COD_ART'] == producto]['Horas'].sum()) if 'COD_ART' in df_calendario.columns and 'Horas' in df_calendario.columns else 0.0
                
                # Solo almacenar si hay diferencia
                if abs(horas_propuesto_producto - horas_calendario_producto) > 0.1:
                    horas_productos_comunes[producto] = {
                        'Propuesto': horas_propuesto_producto,
                        'Calendario': horas_calendario_producto,
                        'Diferencia': horas_calendario_producto - horas_propuesto_producto
                    }
            except Exception as e:
                logger.warning(f"Error calculando horas para producto {producto}: {str(e)}")
        
        # Agregar columna de identificación para las gráficas
        df_propuesto['Origen'] = 'Propuesto'
        df_calendario['Origen'] = 'Generado'
        
        # Combinar datos para las gráficas
        # Seleccionar solo las columnas comunes para evitar errores
        cols_comunes = list(set(df_propuesto.columns).intersection(set(df_calendario.columns)))
        df_combinado = pd.concat([df_propuesto[cols_comunes], df_calendario[cols_comunes]])
        
        # Generar gráficas
        generar_graficas_comparacion(df_combinado, fecha_inicio)
        
        # Crear resumen
        resumen = {
            'Fecha': fecha_inicio.strftime('%d/%m/%Y'),
            'Productos': {
                'Propuesto': productos_propuesto,
                'Calendario': productos_calendario,
                'Comunes': len(productos_comunes),
                'Solo Propuesto': len(productos_solo_propuesto),
                'Solo Calendario': len(productos_solo_calendario)
            },
            'Horas': {
                'Propuesto': horas_propuesto,
                'Calendario': horas_calendario,
                'Diferencia': horas_calendario - horas_propuesto
            },
            'Distribucion Dias': {
                'Propuesto': horas_dia_propuesto,
                'Calendario': horas_dia_calendario
            },
            'Distribucion Grupos': {
                'Propuesto': horas_grupo_propuesto,
                'Calendario': horas_grupo_calendario
            },
            'Diferencias Productos': horas_productos_comunes
        }
        
        # Guardar resumen
        with open(f"resumen_comparacion_{fecha_inicio.strftime('%d-%m-%Y')}.json", 'w') as f:
            json.dump(resumen, f, indent=4, default=str)  # default=str maneja valores no serializables
        
        return resumen
        
    except Exception as e:
        logger.error(f"Error comparando calendarios: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def generar_graficas_comparacion(df_combinado, fecha_inicio):
    """
    Genera gráficas comparativas basadas en el dataframe combinado.
    
    Args:
        df_combinado: DataFrame con datos de ambos calendarios
        fecha_inicio: Fecha de inicio de la planificación
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Para entornos sin interfaz gráfica
        import matplotlib.pyplot as plt
        
        # Asegurar que la columna Horas es numérica
        if 'Horas' in df_combinado.columns:
            if not pd.api.types.is_numeric_dtype(df_combinado['Horas']):
                df_combinado['Horas'] = df_combinado['Horas'].apply(procesar_horas_string)
        
        # 1. Gráfica de horas por día
        if 'Dia' in df_combinado.columns and 'Horas' in df_combinado.columns and 'Origen' in df_combinado.columns:
            # Agrupar por día y origen
            horas_por_dia = df_combinado.groupby(['Dia', 'Origen'])['Horas'].sum().reset_index()
            
            # Pivot para tener orígenes como columnas
            horas_por_dia_pivot = horas_por_dia.pivot(index='Dia', columns='Origen', values='Horas')
            
            # Crear gráfica
            plt.figure(figsize=(10, 6))
            horas_por_dia_pivot.plot(kind='bar', ax=plt.gca())
            plt.title(f'Comparación de Horas por Día - {fecha_inicio.strftime("%d/%m/%Y")}')
            plt.xlabel('Día')
            plt.ylabel('Horas')
            plt.tight_layout()
            plt.savefig(f'horas_por_dia_{fecha_inicio.strftime("%d-%m-%Y")}.png')
            plt.close()
            
            # Guardar datos
            horas_por_dia_pivot.to_csv(f'horas_por_dia_{fecha_inicio.strftime("%d-%m-%Y")}.csv', sep=';')
        
        # 2. Gráfica de horas por grupo
        if 'COD_GRU' in df_combinado.columns and 'Horas' in df_combinado.columns and 'Origen' in df_combinado.columns:
            # Agrupar por grupo y origen
            horas_por_grupo = df_combinado.groupby(['COD_GRU', 'Origen'])['Horas'].sum().reset_index()
            
            # Pivot para tener orígenes como columnas
            horas_por_grupo_pivot = horas_por_grupo.pivot(index='COD_GRU', columns='Origen', values='Horas')
            
            # Crear gráfica
            plt.figure(figsize=(10, 6))
            horas_por_grupo_pivot.plot(kind='bar', ax=plt.gca())
            plt.title(f'Comparación de Horas por Grupo - {fecha_inicio.strftime("%d/%m/%Y")}')
            plt.xlabel('Grupo')
            plt.ylabel('Horas')
            plt.tight_layout()
            plt.savefig(f'horas_por_grupo_{fecha_inicio.strftime("%d-%m-%Y")}.png')
            plt.close()
            
            # Guardar datos
            horas_por_grupo_pivot.to_csv(f'horas_por_grupo_{fecha_inicio.strftime("%d-%m-%Y")}.csv', sep=';')
        
        # 3. Gráfica de productos por origen
        if 'COD_ART' in df_combinado.columns and 'Origen' in df_combinado.columns:
            # Contar productos por origen
            productos_por_origen = df_combinado.groupby('Origen')['COD_ART'].nunique()
            
            # Crear gráfica
            plt.figure(figsize=(8, 6))
            productos_por_origen.plot(kind='bar', ax=plt.gca())
            plt.title(f'Productos Planificados - {fecha_inicio.strftime("%d/%m/%Y")}')
            plt.xlabel('Origen')
            plt.ylabel('Número de Productos')
            plt.tight_layout()
            plt.savefig(f'productos_por_origen_{fecha_inicio.strftime("%d-%m-%Y")}.png')
            plt.close()
            
            # Guardar datos
            productos_por_origen.to_csv(f'productos_por_origen_{fecha_inicio.strftime("%d-%m-%Y")}.csv', sep=';')
        
        logger.info("Gráficas generadas correctamente")
    except ImportError:
        logger.warning("No se pudo generar gráficas. Asegúrese de tener matplotlib instalado")
    except Exception as e:
        logger.error(f"Error generando gráficas: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def generar_mensaje_comparacion(resumen_comparacion):
    """
    Genera un mensaje formateado para mostrar la comparación.
    
    Args:
        resumen_comparacion: Diccionario con el resumen de la comparación
        
    Returns:
        str: Mensaje formateado para mostrar al usuario
    """
    if not resumen_comparacion:
        return "\n\nNo se pudo realizar la comparación con el planning propuesto."
        
    try:
        productos_comunes = resumen_comparacion['Productos']['Comunes']
        productos_solo_prop = resumen_comparacion['Productos']['Solo Propuesto']
        productos_solo_cal = resumen_comparacion['Productos']['Solo Calendario']
        diferencia_horas = resumen_comparacion['Horas']['Diferencia']
        
        mensaje = (f"\n\nCOMPARACIÓN CON PLANNING PROPUESTO:"
                  f"\n- Productos en planning propuesto: {resumen_comparacion['Productos']['Propuesto']}"
                  f"\n- Productos en planning generado: {resumen_comparacion['Productos']['Calendario']}"
                  f"\n- Productos en común: {productos_comunes}"
                  f"\n- Productos solo en propuesto: {productos_solo_prop}"
                  f"\n- Productos solo en generado: {productos_solo_cal}"
                  f"\n- Horas en planning propuesto: {resumen_comparacion['Horas']['Propuesto']:.1f}"
                  f"\n- Horas en planning generado: {resumen_comparacion['Horas']['Calendario']:.1f}"
                  f"\n- Diferencia de horas: {diferencia_horas:.1f}")
        
        # Agregar información sobre distribución por grupos si está disponible
        if (resumen_comparacion['Distribucion Grupos']['Propuesto'] and 
            resumen_comparacion['Distribucion Grupos']['Calendario']):
            mensaje += "\n\nDistribución por grupos:"
            
            # Combinar claves de ambos diccionarios
            grupos = set(resumen_comparacion['Distribucion Grupos']['Propuesto'].keys()).union(
                set(resumen_comparacion['Distribucion Grupos']['Calendario'].keys())
            )
            
            for grupo in sorted(grupos):
                horas_prop = float(resumen_comparacion['Distribucion Grupos']['Propuesto'].get(grupo, 0))
                horas_cal = float(resumen_comparacion['Distribucion Grupos']['Calendario'].get(grupo, 0))
                mensaje += f"\n  - {grupo}: Propuesto={horas_prop:.1f}h, Generado={horas_cal:.1f}h"
        
        # Agregar información sobre las diferencias principales
        if resumen_comparacion['Diferencias Productos']:
            mensaje += "\n\nPrincipales diferencias en productos comunes:"
            # Ordenar por magnitud de diferencia
            productos_ordenados = sorted(
                resumen_comparacion['Diferencias Productos'].items(),
                key=lambda x: abs(x[1]['Diferencia']),
                reverse=True
            )
            
            # Mostrar solo los 5 con mayores diferencias
            for producto, datos in productos_ordenados[:5]:
                mensaje += f"\n  - {producto}: Propuesto={datos['Propuesto']:.1f}h, Generado={datos['Calendario']:.1f}h (Dif: {datos['Diferencia']:.1f}h)"
        
        mensaje += f"\n\nSe han generado gráficas y un resumen detallado de la comparación."
        return mensaje
    except Exception as e:
        logger.error(f"Error generando mensaje de comparación: {str(e)}")
        return "\n\nSe realizó la comparación pero ocurrió un error al generar el mensaje detallado."