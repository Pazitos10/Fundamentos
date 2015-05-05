###Universidad Nacional de la Patagonia San Juan Bosco - Facultad de Ingeniería - Fundamentos Teóricos de la Informática - Trabajo Final - Machine learning: Algoritmos de Clasificación para reconocimiento de dígitos

#####Introducción
Para el presente trabajo se propone la realización de una aplicación que permita al usuario dibujar un número (entre 0 y 9) y la aplicación determine de qué dígito se trata a través de la utilización de algoritmos de clasificación.
A su vez se desplegará información sobre la precisión global de cada algoritmo utilizado.


#####Objetivos
Construir una aplicación que reconozca/clasifique dígitos del 0 al 9, en base a información suministrada por el usuario.     
Aplicar conceptos de: 
Algoritmos de clasificación.
Procesamiento de imágenes (nivel básico).

#####Palabras clave
Dígitos, Reconocimiento, Clasificación, Estadística, Machine Learning, Características  (features), Clases, Supervisado, No Supervisado, Support Vector Machines, Extra Trees Classifiers, K-Nearest Neighbors Classifiers.

===========

######Implementación de la aplicación
El trabajo es presentado en forma de aplicación web, montado sobre un servidor apache, el cual corre scripts python utilizando la técnica cgi del lado del servidor.
===========

######Configuración del servidor:
El servidor elegido fue apache en su versión 2.4.7 y fue instalado sobre un Ubuntu 14.04 con el comando: sudo apt-get install apache2  y requiere una configuración mínima, que consiste en:     
- Agregar el directorio para el proyecto en el archivo /etc/apache2/apache.conf
- Habilitar la ejecución de dichos scripts al editar el archivo /etc/apache2/conf-available/serve-cgi-bin.conf 
  La ubicación del archivo puede variar según la versión/plataforma en la cual se instale apache.

Donde es necesario otorgar un path en el cual corran los archivos de python con la directiva ScriptAlias /cgi-bin/ <path_residencia_scripts>
Luego debe indicarse el directorio con el mismo path de residencia de los scripts y dentro de esa sección deben completarse las directivas Options y AddHandler.
===========

######Configuración de los scripts python para la utilización de cgi:
Habiendo determinado el directorio de los scripts dentro del árbol del proyecto, para hacer uso de la técnica cgi se realiza el import de los paquetes: cgi y cgitb, en el archivo que sea necesario. Su utilización básica requiere que en un documento html se utilice un formulario cuyo atributo action contenga el path del script python que ejecuta la lógica del servidor y se supone, está esperando los datos. Luego el o los campos con datos de interés, tienen que tener su atributo name para que los datos viajen al servidor y por último un botón para que se ejecute la acción de submit del formulario.
===========

######Utilización:
Es necesario instalar las dependencias de python ubicadas en el archivo requirements.txt incluido en el proyecto. Para instalar todos los paquetes necesarios se utiliza pip, el instalador de paquetes de python, como sigue: pip install -r requirements.txt .
Luego, es necesario abrir el browser y en la barra de búsqueda colocar: localhost/ultimateRecognizer/html/base.html , es claro que la ubicación del proyecto ya sido configurada en apache con la directiva ServerName localhost en el archivo /etc/apache2/apache.conf.
La aplicación se le presenta al usuario con un canvas en el que puede dibujar el dígito a reconocer mediante la utilización del mouse y puede borrar el contenido del canvas cuando desee haciendo clic en el botón Limpiar lienzo. Una vez dibujado el dígito, puede presionar el botón Reconocer para obtener la información.

La información obtenida del canvas se envía al servidor, y puede ser procesada gracias a las librerías cgi y PIL/Pillow que permite trabajar los datos en forma de imagen.


Una vez obtenidos los datos correspondientes al dibujo del usuario, se los convierte en una imagen en escala de grises, luego se la redimensiona a un tamaño de  8x8 para poder ser enviada a los clasificadores y comparada con las imágenes del dataset de dígitos que provee la librería scikit learn.

Internamente, se crea una instancia de nuestro clasificador que devuelve una instancia de cada uno de los 4 clasificadores utilizados. Puede verse la implementación en el archivo tinyClassifier.py incluido en el proyecto.
Luego, cada uno de estos recibe conjuntos de datos separados en, uno con datos para el entrenamiento y uno con datos para pruebas, y luego esa información sirve de entrada al método Fit(entrenar) que cada clasificador posee. A su vez los clasificadores ejecutan el método predict(predecir) con los datos de la imagen procesada anteriormente, que le llega por parámetro a la función get_results.

Una vez procesada la imagen, los clasificadores devuelven la información correspondiente, la cual el servidor se encarga de enviar al cliente. El cliente(Browser) muestra la información en formato HTML para que el usuario pueda interpretar los resultados.
===========

