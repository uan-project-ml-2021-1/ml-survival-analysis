@ECHO OFF 
TITLE My System Info
ECHO Instalando modulos(recuerde tener como variable de entorno la carpeta donde se instalo Python)

ECHO ==========================
ECHO Instalando Librerias...
ECHO ============================
START /wait pip install pandas
ECHO pandas Instalado
ECHO ============================
START /wait pip install numpy
ECHO numpy Instalado
ECHO ============================
START /wait pip install sklearn
ECHO scikit learn Instalado
ECHO ============================
ECHO Instalacion de librerias finalizada
ECHO ============================
PAUSE