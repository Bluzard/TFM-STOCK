import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
import os
from csv_loader import leer_dataset, leer_pedidos_pendientes, verificar_dataset_existe
from planner import calcular_formulas, aplicar_simplex, exportar_resultados, verificar_pedidos
from comparador import cargar_planning_propuesto, comparar_calendarios, generar_mensaje_comparacion
from PIL import Image, ImageTk 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Opti-Planner")
        self.root.geometry("600x450")
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.dataset_date = tk.StringVar()
        self.dias_planificacion = tk.StringVar(value="7")  # Valor por defecto
        self.dias_no_habiles = tk.StringVar(value="2")  # Valor por defecto
        self.horas_mantenimiento = tk.StringVar(value="8")  # Valor por defecto
        self.dias_cobertura = tk.StringVar(value="7")  # Valor por defecto
        
        self.create_widgets()
        self.load_image("imagen_logo.png") 
        
    def create_widgets(self):
        # Frame principal con padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Dataset
        ttk.Label(main_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.dataset_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Buscar", command=self.browse_file).grid(row=0, column=2)
        
        # Fecha Dataset (solo mostrar)
        ttk.Label(main_frame, text="Fecha Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.dataset_date).grid(row=1, column=1, sticky=tk.W)
        
        # Fecha Inicio
        ttk.Label(main_frame, text="Fecha Inicio:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.fecha_inicio = DateEntry(main_frame, width=12, background='darkblue',
                                    foreground='white', borderwidth=2,
                                    date_pattern='dd/mm/yyyy')
        self.fecha_inicio.grid(row=2, column=1, sticky=tk.W)
        
        # Días planificación
        ttk.Label(main_frame, text="Días Planificación:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.combo_dias_planif = ttk.Combobox(main_frame, textvariable=self.dias_planificacion, 
                                           values=["1", "2", "3", "4", "5", "6", "7"], 
                                           width=10, state="readonly")
        self.combo_dias_planif.current(6)  # Seleccionar 7 por defecto
        self.combo_dias_planif.grid(row=3, column=1, sticky=tk.W)
        
        # Días no hábiles
        ttk.Label(main_frame, text="Días No Hábiles:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.dias_no_habiles, width=10).grid(row=4, column=1, sticky=tk.W)
        
        # Horas mantenimiento
        ttk.Label(main_frame, text="Horas Mantenimiento/Pruebas:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.combo_horas_mant = ttk.Combobox(main_frame, textvariable=self.horas_mantenimiento, 
                                          values=["0","1", "2", "3","4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"], 
                                          width=10, state="readonly")
        self.combo_horas_mant.current(8)  # Seleccionar 8 por defecto
        self.combo_horas_mant.grid(row=5, column=1, sticky=tk.W)
        
        # Días cobertura
        ttk.Label(main_frame, text="Días Cobertura:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.combo_dias_cobertura = ttk.Combobox(main_frame, textvariable=self.dias_cobertura, 
                                              values=["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], 
                                              width=10, state="readonly")
        self.combo_dias_cobertura.current(4)  # Seleccionar 3 por defecto
        self.combo_dias_cobertura.grid(row=6, column=1, sticky=tk.W)
        
        # Botón generar
        ttk.Button(main_frame, text="Generar Plan", command=self.generate_plan).grid(row=8, column=1, pady=20)

        # Botón comparar resultado
        ttk.Button(main_frame, text="Comparar Resultado", command=self.comparar_resultado).grid(row=9, column=1, pady=10)

        # Label para la imagen logo
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=10, column=0, columnspan=3, pady=10, padx=255)
   
    def load_image(self, image_path):
        """Carga y muestra una imagen en la interfaz."""
        try:
            image = Image.open(image_path)
            image = image.resize((100, 100), Image.LANCZOS)  # Redimensionar para ajustarse a la ventana
            self.img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.img_tk)
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")

        
    def browse_file(self):
        filetypes = (
            ('Archivos CSV', '*.csv'),
            ('Todos los archivos', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Seleccionar Dataset',
            initialdir='.',
            filetypes=filetypes
        )
        
        if filename:
            self.dataset_path.set(filename)
            self.extract_dataset_date(filename)
            
    def extract_dataset_date(self, filename):
        try:
            basename = os.path.basename(filename)
            if basename.startswith('Dataset '):
                date_str = basename[8:16]  # Formato DD-MM-YY
                date_obj = datetime.strptime(date_str, '%d-%m-%y')
                self.dataset_date.set(date_obj.strftime('%d/%m/%Y'))
                
                # Establecer fecha inicio un día después
                next_day = date_obj + timedelta(days=1)
                self.fecha_inicio.set_date(next_day)
            else:
                self.dataset_date.set("Formato de archivo no reconocido")
        except Exception as e:
            logger.error(f"Error extrayendo fecha: {str(e)}")
            self.dataset_date.set("Error en fecha")
            
    def validate_inputs(self):
        try:
            if not self.dataset_path.get():
                raise ValueError("Seleccione un archivo dataset")
                
            if not verificar_dataset_existe(self.dataset_path.get()):
                raise ValueError("Archivo dataset no encontrado")
                
            dias_planificacion = int(self.dias_planificacion.get())
            if dias_planificacion < 1 or dias_planificacion > 7:
                raise ValueError("Los días de planificación deben estar entre 1 y 7")
                
            try:
                dias_no_habiles = float(self.dias_no_habiles.get())
                if dias_no_habiles < 1 or dias_no_habiles > 4:
                    raise ValueError("Los días no hábiles deben estar entre 1 y 4")
                    
                # Verificar que días no hábiles sean menores que días planificación
                if dias_no_habiles >= dias_planificacion:
                    raise ValueError("Los días no hábiles deben ser menos que los días de planificación")
            except ValueError as e:
                if "could not convert string to float" in str(e):
                    raise ValueError("Formato incorrecto para días no hábiles. Ingrese un número entre 1 y 4.")
                else:
                    raise e
                
            horas_mantenimiento = int(self.horas_mantenimiento.get())
            if horas_mantenimiento < 0 or horas_mantenimiento > 16:
                raise ValueError("Las horas de mantenimiento deben estar entre 0 y 16")
                
            dias_cobertura = int(self.dias_cobertura.get())
            if dias_cobertura < 3 or dias_cobertura > 12:
                raise ValueError("Los días de cobertura deben estar entre 3 y 12")
                
            # Validar fechas
            fecha_dataset = datetime.strptime(self.dataset_date.get(), '%d/%m/%Y').date()
            fecha_inicio = self.fecha_inicio.get_date()
            
            if fecha_inicio < fecha_dataset:
                raise ValueError("La fecha de inicio debe ser posterior a la fecha del dataset")
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Error de Validación", str(e))
            return False
        except Exception as e:
            logger.error(f"Error en validación: {str(e)}")
            messagebox.showerror("Error", "Por favor verifique todos los campos")
            return False
            
    def mostrar_alerta_stock_negativo(self, productos_stock_negativo, fecha_dataset_dt, fecha_sugerida=None):
        """
        Muestra una alerta visual con productos de stock negativo y pide confirmación
        
        Args:
            productos_stock_negativo: Lista de productos con stock negativo
            fecha_dataset_dt: Fecha del dataset
            fecha_sugerida: Fecha sugerida para planificación (opcional)
            
        Returns:
            bool: True si el usuario decide continuar con la fecha actual
                False si el usuario decide cancelar
                "fecha_actualizada" (string) si el usuario selecciona actualizar la fecha
        """
        if not productos_stock_negativo:
            return True
                
        # Crear un mensaje detallado para el diálogo
        mensaje = "⚠️ ALERTA: STOCK INICIAL NEGATIVO ⚠️\n\n"
        mensaje += "Se detectaron productos con stock inicial negativo.\n"
        mensaje += "Esto indica que podría haberse producido una rotura de stock antes del inicio de la planificación.\n\n"
        
        # Añadir información de los productos (limitado a los primeros 5 para no saturar)
        mensaje += "Productos afectados (hasta 5 mostrados):\n\n"
        mensaje += f"{'CÓDIGO':<10} {'NOMBRE':<30} {'STOCK INICIAL':<15} {'DÍAS ANTES':<12} {'ROTURA EN':<12}\n"
        
        for i, p in enumerate(productos_stock_negativo[:5]):
            mensaje += f"{p['cod_art']:<10} {p['nom_art'][:28]:<30} {p['stock_inicial']:<15} {p['dias_cobertura']:<12} {p['dia_rotura']:<12}\n"
                
        if len(productos_stock_negativo) > 5:
            mensaje += f"\n... y {len(productos_stock_negativo) - 5} productos más."
        
        if fecha_sugerida:
            mensaje += f"\n\n🔹 Se recomienda planificar desde la fecha: {fecha_sugerida.strftime('%d/%m/%Y')}"
            mensaje += "\n\n¿Qué desea hacer?\n"
            mensaje += "- Actualizar fecha: Actualiza la fecha de inicio a la recomendada y cancela el proceso actual\n"
            mensaje += "- Continuar: Sigue con el proceso usando la fecha actual\n"
            mensaje += "- Cancelar: Detiene el proceso sin cambios"
            
            # Crear un diálogo personalizado con tres botones
            dialog = tk.Toplevel(self.root)
            dialog.title("Alerta de Stock Negativo")
            dialog.geometry("600x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Hacer que el diálogo sea modal
            dialog.focus_set()
            
            # Variable para almacenar el resultado
            result = tk.StringVar()
            
            # Crear un widget de texto para mostrar el mensaje
            text_widget = tk.Text(dialog, wrap=tk.WORD, width=70, height=15)
            text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            text_widget.insert(tk.END, mensaje)
            text_widget.config(state=tk.DISABLED)
            
            # Frame para botones
            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=10)
            
            # Función para establecer resultado y cerrar diálogo
            def set_result(value):
                result.set(value)
                dialog.destroy()
            
            # Botones
            tk.Button(button_frame, text="Actualizar fecha", width=25, 
                    command=lambda: set_result("actualizar")).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Continuar con fecha actual", width=25,
                    command=lambda: set_result("continuar")).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Cancelar", width=15,
                    command=lambda: set_result("cancelar")).pack(side=tk.LEFT, padx=5)
            
            # Esperar hasta que el diálogo se cierre
            self.root.wait_window(dialog)
            
            # Procesar resultado
            if result.get() == "actualizar":
                # Actualizar la fecha en el calendario
                self.fecha_inicio.set_date(fecha_sugerida)
                # Devolver una señal especial indicando que se actualizó la fecha
                return "fecha_actualizada"
            elif result.get() == "continuar":
                return True
            else:  # cancelar
                return False
        else:
            # Comportamiento original cuando no hay fecha sugerida
            mensaje += "\n\n🔹 Se recomienda adelantar la planificación a una fecha anterior a la primera rotura."
            mensaje += "\n\n¿Desea continuar de todos modos?"
            
            # Mostrar diálogo de confirmación
            respuesta = messagebox.askyesno("Alerta de Stock Negativo", mensaje)
            return respuesta
    def comparar_resultado(self):
        """Función para comparar el resultado generado con el planning propuesto"""
        try:
            # Verificar que tenemos todos los datos necesarios
            if not self.dataset_date.get():
                messagebox.showerror("Error", "Debe seleccionar un dataset primero")
                return
                
            fecha_dataset = datetime.strptime(self.dataset_date.get(), '%d/%m/%Y')
            fecha_planificacion = self.fecha_inicio.get_date()
            
            # Obtener el nombre del archivo del calendario generado
            nombre_calendario = f"calendario_fd{fecha_dataset.strftime('%d-%m-%y')}_fi{fecha_planificacion.strftime('%d-%m-%Y')}.csv"
            
            # Verificar si existe el calendario generado
            if not os.path.exists(nombre_calendario):
                messagebox.showerror("Error", f"No se encontró el calendario generado: {nombre_calendario}")
                return
                
            # Cargar el calendario generado
            df_calendario = pd.read_csv(nombre_calendario, sep=';', encoding='utf-8-sig')
            
            # Cargar el planning propuesto de la carpeta Planning
            df_propuesto = cargar_planning_propuesto(fecha_planificacion, carpeta="Planning")
            
            # Si existe el planning propuesto, hacer la comparación
            if df_propuesto is not None:
                logger.info("Iniciando comparación entre calendario generado y planning propuesto...")
                resumen_comparacion = comparar_calendarios(df_propuesto, df_calendario, fecha_planificacion, fecha_dataset)
                
                # Generar mensaje de comparación formateado
                mensaje_comparacion = generar_mensaje_comparacion(resumen_comparacion)
                
                # Crear una ventana para mostrar los resultados de la comparación
                ventana_comparacion = tk.Toplevel(self.root)
                ventana_comparacion.title("Comparación de Plannings")
                ventana_comparacion.geometry("700x500")
                
                # Texto con scroll
                frame_scroll = ttk.Frame(ventana_comparacion)
                frame_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Scrollbar
                scrollbar = ttk.Scrollbar(frame_scroll)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Área de texto
                texto_comparacion = tk.Text(frame_scroll, wrap=tk.WORD, yscrollcommand=scrollbar.set)
                texto_comparacion.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Configurar scrollbar
                scrollbar.config(command=texto_comparacion.yview)
                
                # Insertar mensaje
                texto_comparacion.insert(tk.END, mensaje_comparacion)
                
                # Botón para cerrar
                ttk.Button(ventana_comparacion, text="Cerrar", command=ventana_comparacion.destroy).pack(pady=10)
                
                # Hacer read-only el texto
                texto_comparacion.config(state=tk.DISABLED)
            else:
                messagebox.showinfo("Información", "No se encontró un planning propuesto para comparar.")
                
        except Exception as e:
            logger.error(f"Error en comparación: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Error en comparación:\n{str(e)}")

    def generate_plan(self):
        if not self.validate_inputs():
            return
            
        try:
            # Obtener parámetros
            fecha_dataset = datetime.strptime(self.dataset_date.get(), '%d/%m/%Y')
            fecha_inicio = self.fecha_inicio.get_date()
            dias_planificacion = int(self.dias_planificacion.get())
            dias_no_habiles = float(self.dias_no_habiles.get())
            horas_mantenimiento = int(self.horas_mantenimiento.get())
            dias_cobertura = int(self.dias_cobertura.get())
            
            # Extraer nombre del dataset
            nombre_dataset = os.path.basename(self.dataset_path.get())
            
            # 1. Leer dataset y calcular fórmulas
            productos = leer_dataset(nombre_dataset)
            if not productos:
                raise ValueError("Error al leer el dataset")

            # Calcular fórmulas y obtener productos con stock negativo
            productos_validos, horas_disponibles, productos_stock_negativo = calcular_formulas(
                productos=productos,
                fecha_inicio=fecha_inicio.strftime('%d-%m-%Y'),
                fecha_dataset=fecha_dataset.strftime('%d-%m-%Y'),
                dias_planificacion=dias_planificacion,
                dias_no_habiles=dias_no_habiles,
                horas_mantenimiento=horas_mantenimiento,
                gui_mode=True  # Indicar que estamos en modo GUI para no mostrar en consola
            )

            # Guardar el número original de productos válidos (para logging)
            num_productos_originales = len(productos_validos)
            logger.info(f"Productos válidos iniciales: {num_productos_originales}")

            # Verificar si hay productos con stock negativo y mostrar alerta
            if productos_stock_negativo:
                # Buscar la fecha de rotura más temprana
                fecha_rotura_temprana = None
                for p in productos_stock_negativo:
                    try:
                        if 'dia_rotura' in p and p['dia_rotura'] != "N/A":
                            fecha_rotura = datetime.strptime(p['dia_rotura'], '%d/%m/%Y')
                            if fecha_rotura_temprana is None or fecha_rotura < fecha_rotura_temprana:
                                fecha_rotura_temprana = fecha_rotura
                    except Exception as e:
                        logger.warning(f"Error al procesar fecha de rotura: {str(e)}")
                
                # Si se encontró una fecha de rotura, restar un día para sugerir planificación
                fecha_sugerida = None
                if fecha_rotura_temprana:
                    fecha_sugerida = fecha_rotura_temprana - timedelta(days=1)
                    
                # Mostrar alerta y obtener respuesta
                respuesta = self.mostrar_alerta_stock_negativo(
                    productos_stock_negativo, 
                    fecha_dataset, 
                    fecha_sugerida
                )
                
                # Si el usuario seleccionó actualizar la fecha, cancelar el proceso actual
                if respuesta == "fecha_actualizada":
                    messagebox.showinfo("Fecha actualizada", 
                                    "La fecha de inicio ha sido actualizada con el valor recomendado. "
                                    "Por favor, vuelva a hacer clic en 'Generar Plan' para procesar con la nueva fecha.")
                    return
                
                # Si el usuario decidió cancelar, interrumpir el proceso
                if not respuesta:
                    logger.info("Proceso interrumpido por el usuario debido a stock negativo")
                    return

            if not productos_validos:
                raise ValueError("Error en los cálculos")

            # 2. Verificar pedidos pendientes
            productos_a_planificar_adicionales = []
            df_pedidos = leer_pedidos_pendientes(fecha_dataset)
            if df_pedidos is not None:
                productos_a_planificar_adicionales = verificar_pedidos(
                    productos=productos,
                    df_pedidos=df_pedidos,
                    fecha_dataset=fecha_dataset,
                    fecha_inicio=fecha_inicio,
                    dias_planificacion=dias_planificacion
                )
                logger.info(f"Productos adicionales por pedidos pendientes: {len(productos_a_planificar_adicionales)}")

            # Combinar productos válidos con los adicionales
            productos_a_planificar = productos_validos + productos_a_planificar_adicionales
            
            # Eliminar duplicados por código de artículo
            productos_dict = {}
            for p in productos_a_planificar:
                if hasattr(p, 'cod_art') and p.cod_art:
                    if p.cod_art not in productos_dict:
                        productos_dict[p.cod_art] = p
                    else:
                        # Si ya existe, mantener el que tenga menor cobertura o mayor demanda
                        existing = productos_dict[p.cod_art]
                        if (hasattr(p, 'cobertura_inicial') and hasattr(existing, 'cobertura_inicial') and
                            isinstance(p.cobertura_inicial, (int, float)) and 
                            isinstance(existing.cobertura_inicial, (int, float))):
                            if p.cobertura_inicial < existing.cobertura_inicial:
                                productos_dict[p.cod_art] = p
            
            productos_a_planificar = list(productos_dict.values())
            logger.info(f"Productos totales a planificar (después de eliminar duplicados): {len(productos_a_planificar)}")
            
            # 3. Aplicar Simplex
            productos_optimizados = aplicar_simplex(
                productos_validos=productos_a_planificar,
                horas_disponibles=horas_disponibles,
                dias_planificacion=dias_planificacion,
                dias_cobertura_base=dias_cobertura
            )
            
            if not productos_optimizados:
                logger.error("La optimización no produjo resultados. Verificando otras alternativas...")
                # Si la optimización falló pero teníamos productos válidos, intentar planificar los más urgentes
                if productos_a_planificar:
                    # Ordenar por cobertura (menor primero) y tomar los 10 más urgentes
                    productos_a_planificar.sort(key=lambda p: p.cobertura_inicial if isinstance(p.cobertura_inicial, (int, float)) else float('inf'))
                    productos_urgentes = productos_a_planificar[:min(10, len(productos_a_planificar))]
                    logger.info(f"Intentando planificar {len(productos_urgentes)} productos urgentes como alternativa")
                    
                    # Intentar asignar producción mínima a cada uno
                    for p in productos_urgentes:
                        if hasattr(p, 'cajas_hora_reales') and p.cajas_hora_reales > 0:
                            p.horas_necesarias = 2.0  # Mínimo 2 horas
                            p.cajas_a_producir = round(p.horas_necesarias * p.cajas_hora_reales)
                            p.cobertura_final_plan = (p.stock_inicial + p.cajas_a_producir) / p.demanda_media
                        else:
                            logger.warning(f"Producto {p.cod_art}: cajas_hora_reales es cero o no está definido")
                    
                    productos_optimizados = [p for p in productos_urgentes if hasattr(p, 'horas_necesarias') and p.horas_necesarias > 0]
                    logger.info(f"Productos planificados alternativamente: {len(productos_optimizados)}")
                    
                    if not productos_optimizados:
                        raise ValueError("No se pudo generar ningún plan de producción")
                else:
                    raise ValueError("No hay productos válidos para optimizar")
            
            # Log para verificación
            logger.info(f"Productos optimizados finales: {len(productos_optimizados)}")
            for i, p in enumerate(productos_optimizados[:5]):  # Mostrar solo los primeros 5 para no saturar el log
                logger.info(f"Producto {i+1}: {p.cod_art} - Horas: {p.horas_necesarias:.2f} - Cajas: {p.cajas_a_producir}")
            
            # 4. Exportar resultados
            resultado_ocupacion = exportar_resultados(
                productos_optimizados=productos_optimizados,
                productos=productos,
                fecha_dataset=fecha_dataset,
                fecha_planificacion=fecha_inicio,
                dias_planificacion=dias_planificacion,
                dias_cobertura_base=dias_cobertura
            )
            
            
            # Mostrar mensaje de éxito con información de ocupación
            mensaje_exito = f"Plan generado y exportado correctamente. Se planificaron {len(productos_optimizados)} productos."
            if resultado_ocupacion:
                mensaje_exito += f"\n\nOcupación de almacén al final de la planificación: {resultado_ocupacion['ocupacion_fin']['total_ubicaciones']} ubicaciones."
            
            
            messagebox.showinfo("Éxito", mensaje_exito)
            
        except Exception as e:
            logger.error(f"Error generando plan: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Error generando plan:\n{str(e)}")


def main():
    root = tk.Tk()
    app = PlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()