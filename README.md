<h1 align="center"> LABORATORIO 2. CONVOLUCIÓN Y CORRELACIÓN </h1>

Autores: Midalys Vanessa Aux y Manuela Martinez 

# Introducción

El análisis de señales es de gran importancia en el área de la ingenieria, como la Biomédica, Mecatrónica, entre otras, puesto que permite estudiar y manipular los distintos tipos que existen tales como las de audio, imágenes, o incluso señales eléctricas biológicas. El análsis de estás es esencial para el diseño, mejoramiento y entendimiento de los sistemas de comunicación y control de sistemas dinámicos varios ya existentes, o incluso para aquellos que aún no tienen su propio entendimiento.

Dentro de las herramientas poderosas de este campo, las cuales nos permitiran llegar precisamente a este fin, el cual es entender de mejor manera una señal, se encuentran las operaciones matemáticas correspondientes a la Convolución, Correlación y La Transformada de Fourier, puesto que nos van a permitir modelar, filtrar, comparar y transformar señales en el dominio adecuado, ya sea en el dominio del tiempo o de la frecuencia. Pero, ¿De qué forma estas operaciones nos facilitan analizar de forma adecuada una señal?

##### • Convolución: 
[^1^] La convolución es una operación matemática que se utiliza para describir cómo una señal de entrada puede ser modificada por un sistema. En otras palabras, la convolución te dice la forma en que cambia la señal al momento de pasar a través de un filtro o sistema. 
[^1^]: Convolución. MATLAB & Simulink. (n.d.). https://la.mathworks.com/discovery/convolution.html 

En el conteto del procesamiento de señales, esta operación se usa principalmente para analizar sistemas lineales e invariantes en el tiempo. Es esencial para el entendimiento del comportamiento de los sistemas cuando una señal de entrada interactua con ellos como en los casos de filtración, ecualización, entre otros.

##### • Correlación: 
[^2^] La correlación permite medir la similitud entre dos señales en función de su desplazamiento temporal. Si se tienen dos señales esta herramienta ayudara a divisar la similitud de una con la otra a medida que se desplazan a lo largo del tiempo. 
[^2^]: Ortega, C. (2023, February 23). ¿Qué es el coeficiente de correlación de pearson?. QuestionPro. https://www.questionpro.com/blog/es/coeficiente-de-correlacion-de-pearson/

Es util para encontrar patrones dentro de una señal o para comparar una señal con un modelo ya existente o con alguna referencia. Por ejemplo, en el procesamiento de imágenes, se usa para poder detectar características o coincidencias de una imágen con respecto a la otra. Así mismo, es útil en el análisis de señales para medir el grado de dependencia entre ellas.

##### • Transformada de Fourier: 
[^3^] La transformada de Fourier descompone una señal en sus componentes de frecuencia. En otras palabras, convierte una señal en el dominio del tiempo en una representación que permite divisar la distribución de sus frecuencias. 
[^3^]: FFT. MathWorks. (n.d.). https://la.mathworks.com/help/matlab/math/fourier-transforms.html 

Esto es de gran utilidad puesto que muchas señales complejas pueden analizarse de manera más fácil cuando se descomponen en componentes de frecuecia simples (funciones de senos y de cosenos). Esta herramienta es fundamental en el análisis espectral puesto que muestra las frecuencuas que están presentes en una señal y cómo estas se distribuyen.

## Analisis
Para la facilidad de cada uno de los lectores, se opto por la creacion de una interfaz grafica que se encuentra dividida en tres partes, siendo la primera: convolucion, la segunda: correlacion y la tercera y cuarta parte el punto tres de la guia de este laboratorio:
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/Interfaz.png?raw=true" width="60%" />

Para la primera parte de este laboratorio, se tiene en cuenta la convolución, para este caso se utilizó a X(n1)= codigo universitario de la persona o sujeto que utilizara el codigo y X(n2)= los digitos de la cedula de ciudadania, para este caso, se realizaron los calculos en primera medida de manera manual y posteriormente utilizando una funcion de la libreria Numpy el comando convolve para que realice el proceso con las dos variables:
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/convolucion%20.png?raw=true" width="60%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/convolucion%201.png?raw=true" width="40%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/convolucion%202.png?raw=true" width="40%" />

A continuacion se evidencian los graficos realizados a mano:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/midd.jpg?raw=true" width="40%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/manum.jpg?raw=true" width="40%" />

En cuanto a la segunda parte del laboratorio, en primera instancia se grafican las 2 funciones a las cuales se les realizaran la correlacion, esto para observar cada uno de los graficos y observar el cambio despues de aplicar la funcion correlate de la libreria Numpy:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/gracorr.png?raw=true" width="40%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/calccorrrrr.png?raw=true" width="40%" />

A continuacion se observa el grafico de la correlacion:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/punto%202.png?raw=true" width="40%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/punto%202%20correlacion.png?raw=true" width="40%" />


En cuanto al tercer Item, para este apartado se utilizara una señal ECG previamente seleccionada desde el repositorio Physionet descargando los archivos .hea y .dat para la respectova extraccion de datos:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/se%C3%B1al%20ecg.png?raw=true" width="60%" />
A esta señal, se le hara el respectivo calculo de los estadisticos, del mismo modo, se le aplicara la transformada de Fourier usando la funcion: fft_values = np.fft.fft(ecg_signal), esto para observar el resultado de la transformada, la densidad espectral de potencia y el espectro de la señal:

A continuacion se observan el resultado de los estadisticos:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/estadiscticos%20.png?raw=true" width="20%" />

A continuacion se observan tanto el histograma como el calculo de la probabilidad:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/histogramaaa.png?raw=true" width="40%" />
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/probabilidad.png?raw=true" width="40%" />

La funcion que se aplico en la parte de la transformada de Fourier es:

<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/transformadaaaaa.png?raw=true" width="40%" />

  Posteriormente, al agregar la funcion para que se observen los graficos de la transformada se observa lo siguiente:
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/espectro%20y%20densidad.png?raw=true" width="40%" />

De la misma manera, se calculan los estadisticos descriptivos,ahora no de la funcion que varia en el tiempo sino en la frecuencia:
<img src="https://github.com/Mida2304/LABORATORIO2/blob/master/frecuencia%20estadisticos.png?raw=true" width="40%" />



## Instrucciones
*1.Creación del entorno Python:*
- Utilizando las librerias numpy,matplotlib.pyplot,sys, norm y Qt esto para realizar todas las partes tanto de la interfaz como del calculo de estadisticos, convolucion y correlacion.
- Se crea una interfaz utilizando Pyqt6 para el diseño de la misma que para este apartado, por decision de estetica se decidio que fuera de color azul.
- Se establecen los botones para las tres partes de este laboratorio como se evidencian en la interfaz.
- Se crean las variables de en definicion, tales como se pueden ver en el analisis esto con todas las funciones.
  
*2. Extracción de datos:*
- Se descargan los datos desde Physionet que es un repositorio de datos de investigación médica de libre acceso.
- Se  seleccionan los archivos .dat y .hea de la señal EMG en este caso.
- Se realiza el calculo de la convolucion manualmente para la obtencion del grafico y posteriormente compararlo con la obtenida por la funcion.
- Para la correlacion se utiliza una funcion brindada por la libreria Numpy, subiendo cada dato y realizando cada ajuste que aparece en el apartado 2.
- Con la señal EGM usando la transformada de fourier se obtienen los respectivos valores en terminos de la preguencia.


*3. Proceso de análisis:*
-La señal se descarga y se procesa en Python
-Todo el análisis se realiza mediante código, lo que permitió evaluar los datos de manera precisa.

###### - Se carga la señal usando wfdb.rdrecord(), y se extraen los datos de la señal en signal_data.
###### - Se creauna replica de la gráfica para observar la señal original utilizando matplotlib, esto con el fin de observar las modificaciones por agregar ruidos.
###### - Se calculan los estadisticos utilizando la librería numpy y manualmente para reforzar la comprensión de los métodos.
###### - Se calcula el histograma y función de probabilidad tanto manualmente como utilizando la función predefinida para visualizar la distribución de los valores.
###### - Se calculan las partes de la convolucion y correlacion ysando la libreria Numpy

## Requisitos
- Contar con Python 3.9.0 instalado.
- Tener acceso a los archivos .dat y .hea.
- Instalar las librerías necesarias para ejecutar el código correctamente.
- Contar con conocimiento sobre programacion en Python.

## Modo de Uso

Por favor, cite de la siguiente manera:
Aux, M.; Martinez, M., Convolución y correlación. 21 de Febrero de 2025

## Información 
est.midalys.aux@unimilitar.edu.co y est.manuela.martin@unimilitar.edu.co
