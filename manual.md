# Proyecto de clasificación de modelos de aprendizaje supervisado

Para ejecutar el proyecto de forma efectiva es necesario clasificar los datos preprocesados

para esto en el folder de `sampling` ejecutar con:


Uso:

    python preprocess.py <name/path of the input file> <name of the output file> <name of the log file>

Dónde preprocess.py es el nombre del fichero de ejecución de python para preprocesar los datos, 
seguido del nombre del archivo de entrada (la data cruda) y la salida, con extensión (delimitada por comas, csv)
y un archivo de log para ver su impresión en un registro

Ejemplo de uso:

```py
python preprocess.py raw_data.csv preprocessed_data.csv preprocessing.log
```

Luego de la data preprocesada es necesario tabularla y graficarla para visualizar su estado en preprocesado
para esto se usa el script de `plot_variance.py`, incluido también en el folder de sampling

Ejemplo de uso:
```Python
Usage:
python plot_variance.py <name of input file>

Usage examples:
python plot_variance.py preprocessed_data.csv
```