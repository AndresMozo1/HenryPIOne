import pandas as pd 
import numpy as np



def countNulls(dataframe, decimales=2):
    """
    creo esta función que devuelve información sobre los valores nulos en un DataFrame.

    Parámetros:
    - dataframe: DataFrame de pandas que se va a analizar.
    - decimales: Número de decimales para redondear el porcentaje de nulos (por defecto, 2).

    Retorna:
    - DataFrame con información sobre columnas, número de nulos y porcentaje de nulos.
    """

    # Crear un DataFrame con información sobre nulos
    dfNulls = pd.DataFrame({
        "Columna": dataframe.columns,
        "Número de Nulos": dataframe.isnull().sum(),
        "Porcentaje de Nulos": (dataframe.isnull().sum() / len(dataframe)) * 100.0
    })

    # Redondear el porcentaje de nulos y agregar el símbolo de porcentaje
    dfNulls["Porcentaje de Nulos"] = dfNulls["Porcentaje de Nulos"].round(decimales).astype(str) + "%"

    # Devolver el DataFrame con la información de nulos
    return dfNulls


def countDuplicates(dataframe):
    """
    Esta función cuenta los valores duplicados en cada columna de un DataFrame.

    Parámetros:
    - dataframe: DataFrame de pandas que se va a analizar.

    Retorna:
    - DataFrame con información sobre columnas, cantidad de valores duplicados y porcentaje de duplicados.
    """

    # Crear un diccionario para almacenar la información
    duplicateDates = {
        "Columna": [],
        "Cantidad de Duplicados": [],
        "Porcentaje de Duplicados": []
    }

    # Iterar sobre las columnas del DataFrame
    for columna in dataframe.columns:
        # Contar la cantidad de valores duplicados en cada columna
        duplicateCount = dataframe[columna].duplicated().sum()

        # Calcular el porcentaje de duplicados
        porcentaje_duplicados = round((duplicateCount / len(dataframe)) * 100.0, 2)

        # Agregar la información al diccionario
        duplicateDates["Columna"].append(columna)
        duplicateDates["Cantidad de Duplicados"].append(duplicateCount)
        duplicateDates["Porcentaje de Duplicados"].append(porcentaje_duplicados)

    # Crear un DataFrame a partir del diccionario
    dfDuplicates = pd.DataFrame(duplicateDates)

    # Devolver el DataFrame con la información de duplicados
    return dfDuplicates


# Definimos una función que calculará un valor de sentimiento
def sentimentScore(text):
    if pd.isnull(text) or text == "":
        return 1  # Valor neutral si el texto está vacío o es NaN
    elif isinstance(text, str):
        # Realizamos análisis de sentimiento
        sentiment = sia.polarity_scores(text)
        compound_score = sentiment["compound"]
        if compound_score >= -0.05:
            return 2 # Positivo
        elif compound_score <= -0.05:
            return 0 # Negativo
        else:
            return 1 # Neutral
    else:
        return 1  # Valor neutral para datos que no son de tipo cadena

import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def yearGenPerHour(df):

    '''Esta función recorre el DataFrame de entrada df, calcula el tiempo total de juego para cada género y año de lanzamiento, y devuelve un nuevo DataFrame con esta información. Los comentarios explican cada paso del proceso para una mejor comprensión.'''
    result = []  # Inicializamos una lista vacía para almacenar los resultados

    # Obtenemos los géneros únicos
    generosUnicos = df['genres'].explode().unique()

    for genero in generosUnicos:
        # Filtramos el DataFrame para el género actual
        dfGenero = df[df['genres'].apply(lambda x: genero in x)]
        
        # Obtenemos los años de lanzamiento únicos para el género actual
        añosUnicos = dfGenero['release_year'].unique()
        
        for año in añosUnicos:
            # Calculamos el tiempo total de juego para el género y año actual
            tiempoJugadoHoras = dfGenero[dfGenero['release_year'] == año]['playtime_forever'].sum()
            
            # Agregamos el resultado a la lista
            result.append({'géneros': genero, 'añoLanzamiento': año, 'tiempoJugado': tiempoJugadoHoras})
    
    # Creamos un DataFrame a partir de la lista de resultados
    return pd.DataFrame(result)


def hoursPerUserGenYear(df):
    
    if isinstance(df['genres'].iloc[0], list):#  convertir a una cadena separada por comas
        df['genres'] = df['genres'].apply(lambda x: ','.join(x))

    # Agrupe por usuario, género y año, sumando las horas de juego
    results = df.groupby(['user_id', 'genres', 'release_year'])['playtime_forever'].sum().reset_index()
    results.rename(columns={'playtime_forever': 'playtime'}, inplace=True)
    
    return results

def mixture(df):
    # La función toma un DataFrame como entrada y crea una cadena de texto concatenando varias columnas del DataFrame
    
    # Concatenación de las columnas con un espacio en blanco entre cada valor
    return (
        df["user_id"] + " " +           # ID del usuario
        df["item_name"] + " " +         # Nombre del artículo
        df["items_count"] + " " +       # Cantidad de ítems
        df["playtime_forever"] + " " +  # Tiempo de juego acumulado
        df["developer"] + " " +         # Desarrollador del juego
        df["genres"] + " " +            # Géneros del juego
        df["price"] + " " +             # Precio del juego
        df["release_year"] + " " +      # Año de lanzamiento del juego
        df["posted_year"] + " " +       # Año de publicación del comentario
        df["item_id"] + " " +           # ID del artículo
        df["recommend"] + " " +         # Recomendación del juego por el usuario
        df["sentiment_analysis"]        # Análisis de sentimientos del comentario
    )


# Creamos una función que elimine registros aleatorios de un dataframe
# Esto con el fin de reducir el peso del archivo que generaremos para la consulta
def randomElim(df, n):
    if n > len(df):
        raise ValueError("Cantidad no permitida")

    alIndex = np.random.choice(len(df), n, replace=False)
    df = df.drop(alIndex)

    return df





