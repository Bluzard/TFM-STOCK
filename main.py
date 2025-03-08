import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import os
from csv_loader import leer_dataset, leer_pedidos_pendientes, verificar_dataset_existe
from planner import calcular_formulas, aplicar_simplex, exportar_resultados, verificar_pedidos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Planificación de Producción")
        self.root.geometry("600x450")
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.dataset_date = tk.StringVar()
        self.dias_planificacion = tk.StringVar(value="7")  # Valor por defecto
        self.dias_no_habiles = tk.StringVar(value="2")  # Valor por defecto
        self.horas_mantenimiento = tk.StringVar(value="8")  # Valor por defecto
        self.dias_cobertura = tk.StringVar(value="3")  # Valor por defecto
        
        self.create_widgets()
        
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
                                          values=["4", "5", "6", "7", "8", "9", "10", "11", "12"], 
                                          width=10, state="readonly")
        self.combo_horas_mant.current(4)  # Seleccionar 8 por defecto
        self.combo_horas_mant.grid(row=5, column=1, sticky=tk.W)
        
        # Días cobertura
        ttk.Label(main_frame, text="Días Cobertura:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.combo_dias_cobertura = ttk.Combobox(main_frame, textvariable=self.dias_cobertura, 
                                              values=["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], 
                                              width=10, state="readonly")
        self.combo_dias_cobertura.current(0)  # Seleccionar 3 por defecto
        self.combo_dias_cobertura.grid(row=6, column=1, sticky=tk.W)
        
        # Botón generar
        ttk.Button(main_frame, text="Generar Plan", command=self.generate_plan).grid(row=8, column=1, pady=20)
        
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
            if horas_mantenimiento < 4 or horas_mantenimiento > 12:
                raise ValueError("Las horas de mantenimiento deben estar entre 4 y 12")
                
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
            
    def mostrar_alerta_stock_negativo(self, productos_stock_negativo, fecha_dataset_dt):
        """Muestra una alerta visual con productos de stock negativo y pide confirmación"""
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
            
        mensaje += "\n\n🔹 Se recomienda adelantar la planificación a una fecha anterior a la primera rotura."
        mensaje += "\n\n¿Desea continuar de todos modos?"
        
        # Mostrar diálogo de confirmación
        respuesta = messagebox.askyesno("Alerta de Stock Negativo", mensaje)
        return respuesta

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

            # Modificar calcular_formulas para que devuelva también los productos con stock negativo
            productos_validos, horas_disponibles, productos_stock_negativo = calcular_formulas(
                productos=productos,
                fecha_inicio=fecha_inicio.strftime('%d-%m-%Y'),
                fecha_dataset=fecha_dataset.strftime('%d-%m-%Y'),
                dias_planificacion=dias_planificacion,
                dias_no_habiles=dias_no_habiles,
                horas_mantenimiento=horas_mantenimiento,
                gui_mode=True  # Indicar que estamos en modo GUI para no mostrar en consola
            )

            # Verificar si hay productos con stock negativo y mostrar alerta
            if productos_stock_negativo and not self.mostrar_alerta_stock_negativo(productos_stock_negativo, fecha_dataset):
                logger.info("Proceso interrumpido por el usuario debido a stock negativo")
                return

            if not productos_validos:
                raise ValueError("Error en los cálculos")

            # 2. Verificar pedidos pendientes
            df_pedidos = leer_pedidos_pendientes(fecha_dataset)
            if df_pedidos is not None:
                productos_a_planificar_adicionales = verificar_pedidos(
                    productos=productos,
                    df_pedidos=df_pedidos,
                    fecha_dataset=fecha_dataset,
                    fecha_inicio=fecha_inicio,
                    dias_planificacion=dias_planificacion
                )

                # Combinar productos válidos con los adicionales
                productos_a_planificar = productos_validos + productos_a_planificar_adicionales
            else:
                productos_a_planificar = productos_validos

            # 3. Aplicar Simplex
            productos_optimizados = aplicar_simplex(
                productos_validos=productos_a_planificar,
                horas_disponibles=horas_disponibles,
                dias_planificacion=dias_planificacion,
                dias_cobertura_base=dias_cobertura
            )
            
            if not productos_optimizados:
                raise ValueError("Error en la optimización")
            
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
            mensaje_exito = "Plan generado y exportado correctamente."
            if resultado_ocupacion:
                mensaje_exito += f"\n\nOcupación de almacén al final de la planificación: {resultado_ocupacion['ocupacion_fin']['total_ubicaciones']} ubicaciones."
            
            messagebox.showinfo("Éxito", mensaje_exito)
            
        except Exception as e:
            logger.error(f"Error generando plan: {str(e)}")
            messagebox.showerror("Error", f"Error generando plan:\n{str(e)}")

def main():
    root = tk.Tk()
    app = PlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()