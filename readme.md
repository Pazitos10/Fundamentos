### Universidad Nacional de la Patagonia San Juan Bosco - Facultad de Ingeniería
#### Fundamentos Teóricos de la Informática - Trabajo Final
#### Machine learning: Algoritmos de Clasificación para reconocimiento de dígitos

##### Introducción
Para el presente trabajo se propone la realización de una aplicación que permita al usuario dibujar un número (entre 0 y 9) y la aplicación determine de qué dígito se trata a través de la utilización de algoritmos de clasificación.
A su vez se desplegará información sobre la precisión global de cada algoritmo utilizado.


##### Objetivos
Construir una aplicación que reconozca/clasifique dígitos del 0 al 9, en base a información suministrada por el usuario.     
Aplicar conceptos de: 
* Algoritmos de clasificación.
* Procesamiento de imágenes (nivel básico).

##### Palabras clave

Dígitos, Reconocimiento, Clasificación, Estadística, Machine Learning, Características  (features), Clases, Supervisado, No Supervisado, Support Vector Machines, Extra Trees Classifiers, K-Nearest Neighbors Classifiers.

---

###### Implementación de la aplicación

El trabajo es presentado en forma de aplicación web, utilizando el microframework "Flask".

###### Utilización:

Es necesario instalar las dependencias de python ubicadas en el archivo **requirements.txt** incluido en el proyecto. Para instalar todos los paquetes necesarios se utiliza **pip**, el instalador de paquetes de python, como sigue: 
	
	pip install -r requirements.txt

Luego, es necesario abrir el browser y en la barra de búsqueda colocar: **localhost:5000**.

La aplicación se le presenta al usuario con un canvas en el que puede dibujar el dígito a reconocer mediante la utilización del mouse y puede borrar el contenido del canvas cuando desee haciendo clic en el botón Limpiar lienzo. Una vez dibujado el dígito, puede presionar el botón Reconocer para obtener la información.

![](https://raw.githubusercontent.com/Pazitos10/Fundamentos/master/ultimateRecognizer/statics/img/test.png)

La información obtenida del canvas se envía al servidor, y puede ser procesada gracias a la librería Pillow que permite trabajar los datos en forma de imagen.

Una vez obtenidos los datos correspondientes al dibujo del usuario, se los convierte en una imagen en escala de grises, luego se la redimensiona a un tamaño de 8x8 para poder ser enviada a los clasificadores y comparada con las imágenes del dataset de dígitos que provee la librería scikit learn.

Internamente, se crea una instancia de nuestro manager de clasificadores que gestiona una instancia de cada uno de los 4 clasificadores utilizados. Puede verse la implementación en el archivo classifiersManager.py incluido en el proyecto.

Una vez procesada la imagen, los clasificadores devuelven la información correspondiente a las predicciones llevadas a cabo y es dispuesta en la aplicación web para que el usuario interprete los resultados.

![](https://raw.githubusercontent.com/Pazitos10/Fundamentos/master/ultimateRecognizer/statics/img/res_test.png)

---

###### Requerimientos:
Si bien se mencionó anteriormente, todos los requerimientos se encuentran detallados en el archivo **requirements.txt**, incluido en el proyecto. Aún así, se mencionan a continuación:

	Flask==0.12.1
	ipdb==0.10.2
	ipython==5.3.0
	ipython-genutils==0.2.0
	itsdangerous==0.24
	Jinja2==2.9.6
	MarkupSafe==1.0
	matplotlib==2.0.0
	numpy==1.12.1
	olefile==0.44
	packaging==16.8
	pexpect==4.2.1
	pickleshare==0.7.4
	Pillow==4.0.0
	prompt-toolkit==1.0.14
	ptyprocess==0.5.1
	Pygments==2.2.0
	pyparsing==2.2.0
	python-dateutil==2.6.0
	pytz==2017.2
	scikit-learn==0.18.1
	scipy==0.19.0
	simplegeneric==0.8.1
	six==1.10.0
	sklearn==0.0
	traitlets==4.3.2
	wcwidth==0.1.7
	Werkzeug==0.12.1
