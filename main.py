import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
from tkcalendar import DateEntry
from csv_loader import leer_dataset, verificar_dataset_existe
from planner import calcular_formulas, aplicar_simplex, exportar_resultados

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('planner_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionPlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Planificación de Producción")
        self.root.geometry("1200x800")
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.dataset_date = tk.StringVar()
        self.planning_days = tk.StringVar()
        self.non_working_days = tk.StringVar()
        self.maintenance_hours = tk.StringVar()
        self.coverage_days = tk.StringVar(value="3")
        
        # Estado del sistema
        self.planning_in_progress = False
        self.productos = []
        self.productos_optimizados = []
        
        self.create_widgets()
        self.create_menu()
        
    def create_menu(self):
        """Crea la barra de menú principal"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Cargar Dataset", command=self.browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)
        
        # Menú Herramientas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Configuración", command=self.show_config_dialog)
        tools_menu.add_command(label="Validar Datos", command=self.validate_data)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Manual", command=self.show_help)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        
    def create_widgets(self):
        """Crea la interfaz principal"""
        # Panel principal
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo - Parámetros
        left_frame = ttk.LabelFrame(main_panel, text="Parámetros de Planificación", padding=10)
        main_panel.add(left_frame, weight=30)
        
        # Dataset
        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset", padding=5)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dataset_frame, text="Archivo:").pack(anchor=tk.W)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=40).pack(fill=tk.X, padx=5)
        ttk.Button(dataset_frame, text="Buscar", command=self.browse_file).pack(pady=5)
        
        ttk.Label(dataset_frame, text="Fecha Dataset:").pack(anchor=tk.W)
        ttk.Label(dataset_frame, textvariable=self.dataset_date).pack(anchor=tk.W, padx=5)
        
        # Parámetros de planificación
        params_frame = ttk.LabelFrame(left_frame, text="Configuración", padding=5)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Fecha inicio
        ttk.Label(params_frame, text="Fecha Inicio:").pack(anchor=tk.W)
        self.start_date = DateEntry(params_frame, width=12, background='darkblue',
                                  foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
        self.start_date.pack(anchor=tk.W, padx=5, pady=2)
        
        # Días planificación
        ttk.Label(params_frame, text="Días a Planificar:").pack(anchor=tk.W)
        ttk.Entry(params_frame, textvariable=self.planning_days, width=8).pack(anchor=tk.W, padx=5, pady=2)
        
        # Días no hábiles
        ttk.Label(params_frame, text="Días No Hábiles:").pack(anchor=tk.W)
        ttk.Entry(params_frame, textvariable=self.non_working_days, width=8).pack(anchor=tk.W, padx=5, pady=2)
        
        # Horas mantenimiento
        ttk.Label(params_frame, text="Horas Mantenimiento:").pack(anchor=tk.W)
        ttk.Entry(params_frame, textvariable=self.maintenance_hours, width=8).pack(anchor=tk.W, padx=5, pady=2)
        
        # Días cobertura
        ttk.Label(params_frame, text="Días Cobertura:").pack(anchor=tk.W)
        ttk.Entry(params_frame, textvariable=self.coverage_days, width=8).pack(anchor=tk.W, padx=5, pady=2)
        
        # Botones
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(pady=10)
        ttk.Button(buttons_frame, text="Generar Plan", command=self.generate_plan).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Limpiar", command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # Panel derecho - Resultados
        right_frame = ttk.LabelFrame(main_panel, text="Resultados", padding=10)
        main_panel.add(right_frame, weight=70)
        
        # Notebook para resultados
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de resumen
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Resumen")
        self.create_summary_view()
        
        # Pestaña de plan diario
        self.daily_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.daily_frame, text="Plan Diario")
        self.create_daily_view()
        
        # Pestaña de alertas
        self.alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alerts_frame, text="Alertas")
        self.create_alerts_view()
        
    def create_summary_view(self):
        """Crea la vista de resumen"""
        # Tabla de resumen
        columns = ('Código', 'Nombre', 'Grupo', 'Stock Inicial', 'Cajas Planificadas', 'Cobertura Final')
        self.summary_tree = ttk.Treeview(self.summary_frame, columns=columns, show='headings')
        
        for col in columns:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=100)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        x_scroll = ttk.Scrollbar(self.summary_frame, orient=tk.HORIZONTAL, command=self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Layout
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_daily_view(self):
        """Crea la vista de plan diario"""
        columns = ('Hora', 'Código', 'Nombre', 'Grupo', 'Cajas', 'Tiempo Setup', 'Tiempo Producción')
        self.daily_tree = ttk.Treeview(self.daily_frame, columns=columns, show='headings')
        
        for col in columns:
            self.daily_tree.heading(col, text=col)
            self.daily_tree.column(col, width=100)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(self.daily_frame, orient=tk.VERTICAL, command=self.daily_tree.yview)
        x_scroll = ttk.Scrollbar(self.daily_frame, orient=tk.HORIZONTAL, command=self.daily_tree.xview)
        self.daily_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Layout
        self.daily_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_alerts_view(self):
        """Crea la vista de alertas"""
        # Texto para alertas
        self.alerts_text = tk.Text(self.alerts_frame, wrap=tk.WORD, height=10)
        self.alerts_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scroll = ttk.Scrollbar(self.alerts_frame, command=self.alerts_text.yview)
        self.alerts_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_file(self):
        """Abre diálogo para seleccionar archivo"""
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
        """Extrae la fecha del dataset del nombre del archivo o contenido"""
        try:
            basename = os.path.basename(filename)
            if basename.startswith('Dataset '):
                date_str = basename[8:16]  # Formato DD-MM-YY
                date_obj = datetime.strptime(date_str, '%d-%m-%y')
                self.dataset_date.set(date_obj.strftime('%d/%m/%Y'))
                
                # Establecer fecha inicio un día después
                self.start_date.set_date(date_obj + timedelta(days=1))
            else:
                self.dataset_date.set("Formato de archivo no reconocido")
        except Exception as e:
            logger.error(f"Error extrayendo fecha: {str(e)}")
            self.dataset_date.set("Error en fecha")
            
    def validate_inputs(self):
        """Valida todos los campos de entrada"""
        try:
            if not self.dataset_path.get():
                raise ValueError("Seleccione un archivo dataset")
                
            if not verificar_dataset_existe(self.dataset_path.get()):
                raise ValueError("Archivo dataset no encontrado")
                
            planning_days = int(self.planning_days.get())
            if planning_days <= 0:
                raise ValueError("Los días de planificación deben ser positivos")
                
            non_working_days = float(self.non_working_days.get())
            if non_working_days < 0 or non_working_days >= planning_days:
                raise ValueError("Días no hábiles inválidos")
                
            maintenance_hours = int(self.maintenance_hours.get())
            if maintenance_hours < 0:
                raise ValueError("Las horas de mantenimiento no pueden ser negativas")
                
            coverage_days = int(self.coverage_days.get())
            if coverage_days <= 0:
                raise ValueError("Los días de cobertura deben ser positivos")
                
            # Validar fechas
            start_date = self.start_date.get_date()
            dataset_date = datetime.strptime(self.dataset_date.get(), '%d/%m/%Y').date()
            
            if start_date < dataset_date:
                raise ValueError("La fecha de inicio debe ser posterior a la fecha del dataset")
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Error de Validación", str(e))
            return False
        except Exception as e:
            logger.error(f"Error en validación: {str(e)}")
            messagebox.showerror("Error", "Por favor verifique todos los campos")
            return False
    def validate_data(self):
        """
        Valida la integridad de los datos cargados y muestra un reporte
        """
        try:
            if not self.dataset_path.get():
                messagebox.showwarning(
                    "Validación",
                    "No hay dataset cargado.\nPor favor, cargue un archivo primero."
                )
                return

            # Cargar y validar dataset
            productos = leer_dataset(self.dataset_path.get())
            if not productos:
                raise ValueError("Error cargando dataset")

            # Realizar validaciones
            validaciones = []
            
            # 1. Verificar productos sin configuración
            productos_sin_config = [p for p in productos if not hasattr(p, 'configuracion')]
            if productos_sin_config:
                validaciones.append(
                    f"⚠️ {len(productos_sin_config)} productos sin configuración en 'indicaciones articulo'"
                )

            # 2. Verificar productos con datos incompletos
            productos_incompletos = [
                p for p in productos if (
                    p.cajas_hora <= 0 or
                    not isinstance(p.demanda_media, (int, float)) or
                    not p.cod_gru
                )
            ]
            if productos_incompletos:
                validaciones.append(
                    f"⚠️ {len(productos_incompletos)} productos con datos incompletos"
                )

            # 3. Verificar stocks negativos
            stocks_negativos = [
                p for p in productos if (
                    p.disponible < 0 or
                    p.calidad < 0 or
                    p.stock_externo < 0
                )
            ]
            if stocks_negativos:
                validaciones.append(
                    f"⚠️ {len(stocks_negativos)} productos con stocks negativos"
                )

            # 4. Verificar demanda media
            sin_demanda = [p for p in productos if p.m_vta_15 <= 0]
            if sin_demanda:
                validaciones.append(
                    f"ℹ️ {len(sin_demanda)} productos sin ventas en últimos 15 días"
                )

            # Mostrar resultados
            if validaciones:
                mensaje = "Resultado de la validación:\n\n" + "\n".join(validaciones)
                messagebox.showwarning("Validación", mensaje)
            else:
                messagebox.showinfo(
                    "Validación",
                    f"✅ Dataset válido\nTotal productos: {len(productos)}"
                )

        except Exception as e:
            logger.error(f"Error en validación: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Error validando datos:\n{str(e)}"
            )
            
    def generate_plan(self):
        """Genera el plan de producción"""
        if not self.validate_inputs() or self.planning_in_progress:
            return
            
        try:
            self.planning_in_progress = True
            self.clear_results()
            
            # Cargar dataset
            logger.debug("Cargando dataset...")
            self.productos = leer_dataset(self.dataset_path.get())
            if not self.productos:
                raise ValueError("Error cargando dataset")
                
            # Preparar fechas y parámetros
            logger.debug("Preparando fechas y parámetros...")
            fecha_dataset = datetime.strptime(self.dataset_date.get(), '%d/%m/%Y')
            fecha_inicio = self.start_date.get_date()
            dias_planificacion = int(self.planning_days.get())
            dias_no_habiles = float(self.non_working_days.get())
            horas_mantenimiento = int(self.maintenance_hours.get())
            
            logger.debug(f"Parámetros: dias_planificacion={dias_planificacion}, "
                        f"dias_no_habiles={dias_no_habiles}, "
                        f"horas_mantenimiento={horas_mantenimiento}")
            
            # Calcular fórmulas
            logger.debug("Calculando fórmulas...")
            productos_validos, horas_disponibles = calcular_formulas(
                productos=self.productos,
                fecha_inicio=fecha_inicio.strftime('%d-%m-%Y'),
                fecha_dataset=fecha_dataset.strftime('%d-%m-%Y'),
                dias_planificacion=dias_planificacion,
                dias_no_habiles=dias_no_habiles,
                horas_mantenimiento=horas_mantenimiento
            )
            
            if not productos_validos:
                raise ValueError("Error en los cálculos iniciales")
                
            logger.debug(f"Productos válidos: {len(productos_validos)}")
            logger.debug(f"Horas disponibles: {horas_disponibles}")
            
            # Ejecutar optimización
            logger.debug("Ejecutando optimización...")
            self.productos_optimizados = aplicar_simplex(
                productos_validos=productos_validos,
                horas_disponibles=horas_disponibles,
                dias_planificacion=dias_planificacion,
                dias_cobertura_base=int(self.coverage_days.get())
            )
            
            if not self.productos_optimizados:
                raise ValueError("Error en la optimización")
                
            logger.debug(f"Productos optimizados: {len(self.productos_optimizados)}")
            
            # Exportar resultados
            logger.debug("Exportando resultados...")
            try:
                exportar_resultados(
                    productos_optimizados=self.productos_optimizados,
                    productos=self.productos,
                    fecha_dataset=fecha_dataset,
                    fecha_planificacion=fecha_inicio,
                    dias_planificacion=dias_planificacion,
                    dias_cobertura_base=int(self.coverage_days.get())
                )
            except Exception as e:
                logger.error(f"Error en exportación: {str(e)}")
                import traceback
                logger.error(f"Traceback de exportación:\n{traceback.format_exc()}")
                raise
            
            # Actualizar vistas
            logger.debug("Actualizando vistas...")
            self.update_summary_view()
            self.update_daily_view()
            self.check_alerts()
            
            messagebox.showinfo(
                "Éxito",
                f"Plan generado correctamente\n" +
                f"Productos planificados: {len(self.productos_optimizados)}"
            )
            
        except Exception as e:
            logger.error(f"Error generando plan: {str(e)}")
            import traceback
            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Error generando plan:\n{str(e)}")
        finally:
            self.planning_in_progress = False
            
    def update_summary_view(self):
        """Actualiza la vista de resumen con los resultados"""
        # Limpiar vista actual
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Ordenar productos por grupo y prioridad
        productos_ordenados = sorted(
            self.productos_optimizados,
            key=lambda p: (
                p.configuracion.orden_planificacion,
                p.cod_gru,
                p.configuracion.prioridad_semanal
            )
        )
        
        # Insertar datos
        for producto in productos_ordenados:
            self.summary_tree.insert('', 'end', values=(
                producto.cod_art,
                producto.nom_art,
                producto.cod_gru,
                f"{producto.stock_inicial:.0f}",
                f"{producto.cajas_a_producir:.0f}",
                f"{producto.cobertura_final_plan:.1f}" if hasattr(producto, 'cobertura_final_plan') else 'N/A'
            ))
            
    def update_daily_view(self):
        """Actualiza la vista de plan diario"""
        # Limpiar vista actual
        for item in self.daily_tree.get_children():
            self.daily_tree.delete(item)
        
        # Agrupar por día
        for dia in range(int(self.planning_days.get())):
            productos_dia = [p for p in self.productos_optimizados if hasattr(p, 'dia_planificado') and p.dia_planificado == dia]
            
            if productos_dia:
                # Agregar encabezado del día
                fecha = self.start_date.get_date() + timedelta(days=dia)
                dia_id = self.daily_tree.insert('', 'end', values=(
                    fecha.strftime('%d/%m/%Y'),
                    '', '', '', '', '', ''
                ))
                
                # Ordenar por hora de inicio
                productos_dia.sort(key=lambda p: p.hora_inicio if hasattr(p, 'hora_inicio') else 0)
                
                # Agregar productos del día
                for producto in productos_dia:
                    hora_inicio = producto.hora_inicio if hasattr(producto, 'hora_inicio') else 0
                    hora_str = f"{int(hora_inicio):02d}:{int((hora_inicio % 1) * 60):02d}"
                    
                    self.daily_tree.insert(dia_id, 'end', values=(
                        hora_str,
                        producto.cod_art,
                        producto.nom_art,
                        producto.cod_gru,
                        f"{producto.cajas_a_producir:.0f}",
                        f"{producto.configuracion.tiempo_cambio}min",
                        f"{producto.horas_necesarias:.1f}h"
                    ))
                    
    def check_alerts(self):
        """Verifica y muestra alertas del plan"""
        self.alerts_text.delete('1.0', tk.END)
        alerts = []
        
        # Verificar coberturas
        for producto in self.productos_optimizados:
            if hasattr(producto, 'cobertura_final_plan'):
                if producto.cobertura_final_plan < 3:
                    alerts.append(
                        f"⚠️ ALERTA: Cobertura baja para {producto.cod_art} - {producto.nom_art}\n" +
                        f"   Cobertura final: {producto.cobertura_final_plan:.1f} días\n"
                    )
        
        # Verificar capacidad de almacenamiento de manera segura
        total_ocupacion = sum(
            producto.cajas_a_producir / producto.configuracion.cajas_palet 
            for producto in self.productos_optimizados 
            if producto.cajas_a_producir > 0 and producto.configuracion.cajas_palet > 0
        )
        
        if total_ocupacion > 800:
            alerts.append(
                f"⚠️ ALERTA: Alta ocupación de almacén\n" +
                f"   Ubicaciones totales: {total_ocupacion:.0f}\n"
            )
        
        # Mostrar alertas
        if alerts:
            self.alerts_text.insert('1.0', "\n".join(alerts))
        else:
            self.alerts_text.insert('1.0', "✅ No hay alertas pendientes")
            
    def clear_form(self):
        """Limpia el formulario"""
        self.dataset_path.set('')
        self.dataset_date.set('')
        self.planning_days.set('')
        self.non_working_days.set('')
        self.maintenance_hours.set('')
        self.coverage_days.set('3')
        self.clear_results()
        
    def clear_results(self):
        """Limpia las vistas de resultados"""
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        for item in self.daily_tree.get_children():
            self.daily_tree.delete(item)
        self.alerts_text.delete('1.0', tk.END)
        
    def show_config_dialog(self):
        """Muestra diálogo de configuración"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        ttk.Label(dialog, text="Configuración del Sistema", font=('Helvetica', 12, 'bold')).pack(pady=10)
        # Aquí irían más opciones de configuración
        
    def show_help(self):
        """Muestra el manual de ayuda"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual de Usuario")
        dialog.geometry("600x400")
        
        text = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        
        text.insert('1.0', """
Manual de Usuario - Sistema de Planificación de Producción

1. Carga de Datos
   - Seleccione el archivo dataset usando el botón "Buscar"
   - El sistema detectará automáticamente la fecha del dataset

2. Configuración de Planificación
   - Establezca la fecha de inicio
   - Ingrese los días a planificar
   - Configure días no hábiles y horas de mantenimiento
   - Ajuste los días de cobertura mínima

3. Generación del Plan
   - Presione "Generar Plan" para iniciar la planificación
   - El sistema mostrará los resultados en las pestañas:
     * Resumen: Vista general de la planificación
     * Plan Diario: Detalle por día y hora
     * Alertas: Avisos importantes sobre el plan

4. Interpretación de Resultados
   - Verde: Planificación normal
   - Amarillo: Atención requerida
   - Rojo: Acción inmediata necesaria
        """)
        text.configure(state='disabled')
        
    def show_about(self):
        """Muestra información sobre el sistema"""
        messagebox.showinfo(
            "Acerca de",
            "Sistema de Planificación de Producción\n" +
            "Versión 1.0\n\n" +
            "Desarrollado para gestión y optimización\n" +
            "de la producción industrial."
        )

def main():
    try:
        root = tk.Tk()
        root.style = ttk.Style()
        root.style.theme_use('clam')
        app = ProductionPlannerGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Error en la aplicación: {str(e)}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        messagebox.showerror(
            "Error Fatal",
            "Error iniciando la aplicación.\n" +
            "Por favor revise los logs para más detalles."
        )

if __name__ == "__main__":
    main()