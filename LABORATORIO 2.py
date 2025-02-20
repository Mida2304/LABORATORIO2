import numpy as np
import matplotlib.pyplot as plt
import sys
import wfdb
from scipy.stats import norm
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QPushButton, QVBoxLayout, QGridLayout, QWidget, QMessageBox, QTextEdit, QFileDialog
)
from PyQt6.QtCore import Qt


class Principal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LABORATORIO 2")
        self.setGeometry(200, 200, 800, 600)

        # Widget principal
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self)

        # Estilo general
        self.setStyleSheet("""
            QMainWindow {
                background-color: #E3DFF3;  /* Azul claro */
            }
            QPushButton {
                background-color: #A3B8E3; /* Azul pastel */
                color: black;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7289DA;  /* Azul más oscuro al pasar el cursor */
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4A6FA5;  /* Azul oscuro */
            }
            QTextEdit {
                background-color: #D4E1F5;
                color: #2C3E50;
                border-radius: 5px;
                font-size: 14px;
            }
        """)

        # Layout principal
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Botón para cargar archivos
        self.load_button = QPushButton("Cargar Archivo .hea")
        self.load_button.clicked.connect(self.load_file)
  

        # Layout en cuadrícula para organizar los botones
        grid_layout = QGridLayout()

        # Botones para convolución
        self.convolucion_Midalys_button = QPushButton("Convolución 1")
        self.convolucion_Manuela_button = QPushButton("Convolución 2")
        self.plot_signals_button = QPushButton("Graficar Señales")
        self.calc_corr_button = QPushButton("Calcular Correlación")
        self.plot_corr_button = QPushButton("Graficar Correlación")

        # Botones de estadísticas
        self.media_button = QPushButton("MEDIA")
        self.mediana_button = QPushButton("MEDIANA")
        self.moda_button = QPushButton("MODA")
        self.varianza_button = QPushButton("VARIANZA")
        self.rango_button = QPushButton("RANGO")
        self.histograma_button = QPushButton("HISTOGRAMA")
        self.desviacion_button = QPushButton("DESVIACIÓN ESTÁNDAR")
        self.probabilidad_ecg_button = QPushButton("Calcular Probabilidad ECG")
        self.fft_button = QPushButton("Transformada de Fourier")

        # Botones transformada
        self.media_tf_button = QPushButton("Calcular Media Tf")
        self.mediana_tf_button = QPushButton("Calcular Mediana Tf")
        self.desviacion_tf_button = QPushButton("Calcular Desviación Estándar Tf")
        self.histograma_tf_button = QPushButton("Mostrar Histograma Tf")

        # Conectar botones a sus funciones
        self.convolucion_Midalys_button.clicked.connect(self.convolucion_Midalys)
        self.convolucion_Manuela_button.clicked.connect(self.convolucion_Manuela)
        self.plot_signals_button.clicked.connect(self.plot_signals)
        self.calc_corr_button.clicked.connect(self.calculate_correlation)
        self.plot_corr_button.clicked.connect(self.plot_correlation)
        
        # Conectar botones a funciones
        self.media_button.clicked.connect(self.calcular_media)
        self.mediana_button.clicked.connect(self.calcular_mediana)
        self.moda_button.clicked.connect(self.calcular_moda)
        self.varianza_button.clicked.connect(self.calcular_varianza)
        self.rango_button.clicked.connect(self.calcular_rango)
        self.histograma_button.clicked.connect(self.mostrar_histograma)
        self.desviacion_button.clicked.connect(self.calcular_desviacion_estandar)
        self.probabilidad_ecg_button.clicked.connect(self.calcular_probabilidad_ecg)
        self.fft_button.clicked.connect(self.analizar_transformada_fourier)
        self.media_tf_button.clicked.connect(self.media_tf)
        self.mediana_tf_button.clicked.connect(self.mediana_tf)
        self.desviacion_tf_button.clicked.connect(self.desviacion_tf)
        self.histograma_tf_button.clicked.connect(self.histograma_tf)



        # Añadir botones al layout
        self.layout.addWidget(self.load_button)
        grid_layout.addWidget(self.convolucion_Midalys_button, 0, 0)
        grid_layout.addWidget(self.convolucion_Manuela_button, 1, 0)
        grid_layout.addWidget(self.plot_signals_button, 0, 1)
        grid_layout.addWidget(self.calc_corr_button, 1, 1)
        grid_layout.addWidget(self.plot_corr_button, 2,1)
        grid_layout.addWidget(self.media_button, 0, 2)
        grid_layout.addWidget(self.mediana_button, 1, 2)
        grid_layout.addWidget(self.moda_button, 2, 2)
        grid_layout.addWidget(self.varianza_button, 3, 2)
        grid_layout.addWidget(self.rango_button, 4, 2)
        grid_layout.addWidget(self.desviacion_button, 5, 2)
        grid_layout.addWidget(self.histograma_button, 6, 2)
        grid_layout.addWidget(self.probabilidad_ecg_button, 7, 2)
        grid_layout.addWidget(self.fft_button, 0, 3)
        grid_layout.addWidget(self.media_tf_button, 1, 3)
        grid_layout.addWidget(self.mediana_tf_button, 2, 3)
        grid_layout.addWidget(self.desviacion_tf_button, 3, 3)
        grid_layout.addWidget(self.histograma_tf_button, 4, 3)

        # Área de texto para el registro
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)

        # Agregar el grid layout al layout principal
        self.layout.addLayout(grid_layout)

        self.setLayout(self.layout)

        # Variable para almacenar la señal cargada
        self.signal = None

        # Definir señales
        self.fs = 1 / (1.25e-3)  # Frecuencia de muestreo
        self.n = np.arange(9)  # Valores de n
        self.Ts = 1.25e-3  # Periodo de muestreo
        self.x1 = np.cos(2 * np.pi * 100 * self.n * self.Ts)
        self.x2 = np.sin(2 * np.pi * 100 * self.n * self.Ts)
        self.corr = None

        
#PUNTO 1
    def convolucion_Midalys(self):
        try:
            h = np.array([5, 6, 0, 0, 6, 1, 2])  
            x = np.array([1, 0, 8, 7, 6, 1, 6, 2, 6, 5])
            y = np.convolve(h, x, mode='full')  # Convolución

            self.log_area.append(f"Convolución Midalys resultado: {y}")

            # Graficar la convolución
            plt.figure(figsize=(10, 4))
            plt.stem(np.arange(len(y)), y, basefmt=' ')
            plt.title("Convolución Midalys")
            plt.xlabel("n")
            plt.ylabel("Amplitud")
            plt.grid()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la convolución: {e}")

    def convolucion_Manuela(self):
        try:
            h = np.array([5, 6, 0, 0, 6, 7, 1])  
            x = np.array([1, 0, 0, 0, 8, 3, 3, 1, 5, 2]) 
            y = np.convolve(h, x, mode='full')  # Convolución

            self.log_area.append(f"Convolución Manuela resultado: {y}")

            # Graficar la convolución
            plt.figure(figsize=(10, 4))
            plt.stem(np.arange(len(y)), y, basefmt=' ')
            plt.title("Convolución Manuela")
            plt.xlabel("n")
            plt.ylabel("Amplitud")
            plt.grid()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la convolución: {e}")
#PUNTO 2

    def plot_signals(self):
        """ Graficar las señales x1[n] y x2[n] """
        plt.figure(figsize=(12, 5))

        plt.subplot(2, 1, 1)
        plt.stem(self.n, self.x1, basefmt=' ')
        plt.title("Señal x1[n] = cos(2π100nTs)")
        plt.xlabel("n")
        plt.ylabel("Amplitud")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.stem(self.n, self.x2, basefmt=' ')
        plt.title("Señal x2[n] = sin(2π100nTs)")
        plt.xlabel("n")
        plt.ylabel("Amplitud")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def calculate_correlation(self):
        """ Calcular la correlación cruzada entre x1[n] y x2[n] """
        self.corr = np.correlate(self.x1, self.x2, mode='full')
        self.log_area.append(f"Correlación calculada: {self.corr}")

    def plot_correlation(self):
        """ Graficar la correlación cruzada """
        if self.corr is None:
            self.log_area.append("Primero calcule la correlación.")
            return
        
        lags = np.arange(-len(self.x1) + 1, len(self.x1))
        plt.figure(figsize=(6, 4))
        plt.stem(lags, self.corr, basefmt=' ')
        plt.title("Correlación entre x1[n] y x2[n]")
        plt.xlabel("Desplazamiento (lag)")
        plt.ylabel("Amplitud")
        plt.grid()
        plt.show()
        

#PUNTO 3
    def load_file(self):
        # Seleccionar archivo .hea
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo .hea", "", "Archivos .hea (*.hea)")

        if file_path:
            try:
                # Elimina la extensión del archivo para cargar el registro
                record_path = file_path.rsplit('.', 1)[0]
                record = wfdb.rdrecord(record_path)

                # Almacena la señal
                self.signal = record.p_signal
                self.log_area.append(f"Archivo cargado: {file_path}\nDimensiones de la señal: {self.signal.shape}")

                # Mostrar la señal ECG en una gráfica
                plt.figure(figsize=(18, 4))
                plt.plot(self.signal[:, 0], color='pink')
                plt.title("Señal ECG")
                plt.xlabel("Tiempo ")
                plt.ylabel("Amplitud (mV)")
                plt.grid()
                plt.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo: {e}")

    def analizar_transformada_fourier(self):
        if self.signal is not None:
            # Extraer la señal ECG (primer canal)
            ecg_signal = self.signal[:, 0]
            N = len(ecg_signal)  # Número de puntos

            if hasattr(self, "fs") and self.fs > 0:
                T = 1 / self.fs  # Período de muestreo
            else:
                QMessageBox.warning(self, "Error", "La frecuencia de muestreo (fs) no está definida o no es válida.")
                return

            freq = np.fft.fftfreq(N, T)  # Frecuencias correspondientes
            fft_values = np.fft.fft(ecg_signal)  # Transformada de Fourier
            self.log_area.append(f"Transformada de Fourier calculada con {N} puntos.")

            # Calcular la densidad espectral de potencia (PSD)
            psd_values = np.abs(fft_values) ** 2 / N
            self.log_area.append(f"Densidad espectral de potencia calculada.")

            # Graficar el espectro de la señal
            plt.figure(figsize=(12, 5))

            plt.subplot(2, 1, 1)
            plt.plot(freq[:N//2], np.abs(fft_values[:N//2]), color='#7FFFD4')  # Magnitud de la FFT
            plt.title("Espectro de la señal ECG")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Amplitud")
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(freq[:N//2], psd_values[:N//2], color='orange')  # PSD
            plt.title("Densidad espectral de potencia (PSD)")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Potencia")
            plt.grid()

            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")


            
    def media_tf(self):
        if self.signal is not None:
            N = len(self.signal[:, 0])
            T = 1 / self.fs  
            freq = np.fft.fftfreq(N, T)[:N // 2]  # Frecuencias positivas
            mean_freq = np.mean(freq)

            self.log_area.append(f"Frecuencia Media: {mean_freq:.2f} Hz")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def mediana_tf (self):
        if self.signal is not None:
            N = len(self.signal[:, 0])
            T = 1 / self.fs  
            freq = np.fft.fftfreq(N, T)[:N // 2]  # Frecuencias positivas
            median_freq = np.median(freq)

            self.log_area.append(f"Frecuencia Mediana: {median_freq:.2f} Hz")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def desviacion_tf (self):
        if self.signal is not None:
            N = len(self.signal[:, 0])
            T = 1 / self.fs  
            freq = np.fft.fftfreq(N, T)[:N // 2]  # Frecuencias positivas
            std_freq = np.std(freq)

            self.log_area.append(f"Desviación Estándar: {std_freq:.2f} Hz")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def histograma_tf (self):
        if self.signal is not None:
            N = len(self.signal[:, 0])
            T = 1 / self.fs  
            freq = np.fft.fftfreq(N, T)[:N // 2]  # Frecuencias positivas

            plt.figure(figsize=(10, 5))
            plt.hist(freq, bins=30, color='yellow', alpha=0.7, edgecolor='black')
            plt.title("Histograma de Frecuencias")
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Frecuencia de aparición")
            plt.grid()
            plt.show()
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")


    def calcular_media(self):
        if self.signal is not None:
            media = np.nanmean(self.signal[:, 0])
            self.log_area.append(f"Media calculada: {media:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def calcular_mediana(self):
        if self.signal is not None:
            mediana = np.nanmedian(self.signal[:, 0])
            self.log_area.append(f"Mediana calculada: {mediana:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def calcular_moda(self):
        if self.signal is not None:
            data = self.signal[:, 0]
            # Calcular el histograma
            counts, bins = np.histogram(data, bins=10)
            # Encontrar el índice del intervalo con la mayor frecuencia
            index_max = np.argmax(counts)
            # Calcular el centro del intervalo modal
            moda_continua = (bins[index_max] + bins[index_max + 1]) / 2
            self.log_area.append(f"Moda continua calculada: {moda_continua:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def calcular_varianza(self):
        if self.signal is not None:
            varianza = np.nanvar(self.signal[:, 0])
            self.log_area.append(f"Varianza calculada: {varianza:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def calcular_rango(self):
        if self.signal is not None:
            rango = np.ptp(self.signal[:, 0])
            self.log_area.append(f"Rango calculado: {rango:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")

    def calcular_desviacion_estandar(self):
        if self.signal is not None:
            desviacion_estandar = np.nanstd(self.signal[:, 0])
            self.log_area.append(f"Desviación estándar calculada: {desviacion_estandar:.6f}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")


    def calcular_probabilidad_ecg(self):
        try:
            if self.signal is not None:
                data = self.signal[:, 0]
                mu = np.mean(data)
                sigma = np.std(data)

                rango_min, rango_max = np.percentile(data, [5, 95])
                probabilidad = norm.cdf(rango_max, mu, sigma) - norm.cdf(rango_min, mu, sigma)

                self.log_area.append(f"Probabilidad en rango [{rango_min:.2f}, {rango_max:.2f}]: {probabilidad:.4%}")

                # Crear el gráfico
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
                y = norm.pdf(x, mu, sigma)

                plt.figure(figsize=(8, 5))
                plt.plot(x, y, label="Distribución ECG", color='purple')

                # Sombrear el área dentro del rango
                x_fill = np.linspace(rango_min, rango_max, 1000)
                y_fill = norm.pdf(x_fill, mu, sigma)
                plt.fill_between(x_fill, y_fill, alpha=0.3, color='#D8BFD8', label="Rango [5%, 95%]")

                plt.title(f"Probabilidad: {probabilidad:.4%}")
                plt.xlabel("Amplitud ECG")
                plt.ylabel("Densidad de probabilidad")
                plt.legend()
                plt.grid(True)

                plt.pause(0.1)  # Evita que la GUI se congele
                plt.show()
            else:
                QMessageBox.warning(self, "Error", "No hay datos cargados.")
        except Exception as e:
            print(f"Error en calcular_probabilidad_ecg: {e}")
            QMessageBox.critical(self, "Error", f"Se produjo un error: {e}")
            
    def mostrar_histograma(self):
        if self.signal is not None:
            try:
                data = self.signal[:, 0]  # Extraer los datos de la señal
                mu = np.mean(data)  # Media
                sigma = np.std(data)  # Desviación estándar

                # Crear histograma con densidad
                plt.figure(figsize=(8, 5))
                count, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.6, color='green', label="Histograma")

                # Crear la curva de Gauss correctamente ajustada
                x = np.linspace(min(data), max(data), 1000)
                gauss_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                # Ajustar la escala de la curva de Gauss al histograma
                gauss_curve *= np.max(count) / np.max(gauss_curve)

                plt.plot(x, gauss_curve, color='blue', linewidth=2, label="Campana de Gauss")

                # Etiquetas y mejoras
                plt.title("Histograma y Distribución Normal Ajustada")
                plt.xlabel("Amplitud ECG")
                plt.ylabel("Densidad de Probabilidad")
                plt.legend()
                plt.grid(True)

                plt.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al mostrar histograma: {e}")
        else:
            QMessageBox.warning(self, "Error", "No hay datos cargados.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = Principal()
    viewer.show()
    sys.exit(app.exec())

