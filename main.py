from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np 

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello, world!"}

@app.get("/PlayTimeGenre/{genero}")
async def PlayTimeGenre(genero: str):
    
    '''Recibe un género  y devuelve el año de lanzamiento con más horas jugadas.
      Digite exactamente igual  alguno de los siguientes generos para realizar la consulta 

Indie                        
Action                       
Casual                        
Adventure                     
Strategy                      
Simulation                    
RPG                           
Free to Play                  
Early Access                  
Sports                        
Massively Multiplayer         
Racing                        
Design &amp; Illustration      
Utilities                      
Web Publishing                 
Animation &amp; Modeling       
Education                      
Video Production               
Software Training              
Audio Production                
Photo Editing                   
Accounting                       

'''
   
    
  
    dfGenere = pd.read_parquet("Data/endpoint_one")
    dfGenere = dfGenere[dfGenere["géneros"] == genero]
    yearWMHours = list(dfGenere[dfGenere["tiempoJugado"] == dfGenere["tiempoJugado"].max()]["añoLanzamiento"])[0]
    return {f"Año de lanzamiento con más horas jugadas  {genero}": yearWMHours}
   



@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: str):
    
    
    '''Recibe un género y devuelve el usuario con más horas jugadas, junto al acumulado de horas por año.
       Digite exactamente igual  alguno de los siguientes generos para realizar la consulta.

Indie                        
Action                       
Casual                        
Adventure                     
Strategy                      
Simulation                    
RPG                           
Free to Play                  
Early Access                  
Sports                        
Massively Multiplayer         
Racing                        
Design &amp; Illustration      
Utilities                      
Web Publishing                 
Animation &amp; Modeling       
Education                      
Video Production               
Software Training              
Audio Production                
Photo Editing                   
Accounting                       
    '''
    
    dfEnd2 = pd.read_parquet("Data/endpoint_two")
    dfEnd2['playtime'] = round(dfEnd2['playtime'] / 60, 2)
    dfEspGen = dfEnd2[dfEnd2['genres'] == genero]
    uWMH = dfEspGen.loc[dfEspGen['playtime'].idxmax()]['user_id']
    userPYH = dfEspGen[dfEspGen['user_id'] == uWMH]
    userPYH = userPYH.groupby('release_year')['playtime'].sum().reset_index()
    userPYH = userPYH.rename(columns={'release_year': 'Año', 'playtime': 'Horas'})
    listPY = userPYH.to_dict(orient='records')
    return { f"Usuario con más horas jugadas para {genero}": uWMH, "Horas jugadas": listPY}  
        
        
@app.get("/UsersRecommend/{year}")
async def UsersRecommend(year: int):
    
    '''Recibe un año y devuelve los 3 juegos más recomendados por los usuarios australianos en el año especificado.  ej 2011 , 2012 , 2013, 2014 '''


    
    df = pd.read_parquet("Data/dfAustralianUserReviews")
    df_year = df[df["posted_year"] == year]
    df_recommend = df_year[df_year["recommend"] == True]
    df_sentiment = df_recommend[df_recommend["sentiment_analysis"].isin([2, 1])]
    df_sentiment = df_sentiment[df_sentiment["title"] != "No especificado"]
    df_sentiment["recommend"] = df_sentiment["recommend"].astype(int)
    recommendations = df_sentiment.groupby("title")["recommend"].sum()
    recommendations = recommendations.sort_values(ascending=False)
    top_3_games = recommendations.head(3).index.tolist()
    if len(top_3_games) >= 3:
        return [{"Puesto 1": top_3_games[0]}, {"Puesto 2": top_3_games[1]}, {"Puesto 3": top_3_games[2]}]
    


@app.get("/UsersNotRecommend/{year}")
async def UUsersNotRecommend(year: int):
    """
    Recibe un año y devuelve las 3 desarrolladoras con más juegos con reseñas negativas en el año especificado. ej 2011 , 2012 , 2013 , 2014 ,    
    """
    
    df = pd.read_parquet("Data/dfAustralianUserReviews") 
    dfYear = df[df["posted_year"] == year] 
    dfNotRecommend = dfYear[dfYear["recommend"] == False] 
    dfSentiment = dfNotRecommend[dfNotRecommend["sentiment_analysis"] == 0]
    dfSentiment = dfSentiment[dfSentiment["developer"] != "Otro"] 
    recommendations2 = dfSentiment.groupby("developer")["recommend"].sum() 
    recommendations2 = recommendations2.sort_values(ascending=True) 
    topNegative = recommendations2.head(3).index.tolist()

    if len(topNegative) >= 3:
        return [{"Puesto 1": topNegative[0]}, {"Puesto 2": topNegative[1]}, {"Puesto 3": topNegative[2]}]
    
    
    
@app.get("/sentiment_analysis/{año}")
async def sentiment_analysis(año: int):
    """
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios
    que se encuentren categorizados con un análisis de sentimiento.

    Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
    """
    # Cargar el DataFrame desde la ruta del archivo.
    df = pd.read_parquet("Data/dfAustralianUserReviews")

    # Filtrar los registros con el año de lanzamiento especificado.
    df_year = df[df["posted_year"] == año]

    # Contar la cantidad de veces que aparecen los valores específicos para cada etiqueta de análisis de sentimiento.
    negative_count = int((df_year["sentiment_analysis"] == 0).sum())
    neutral_count = int((df_year["sentiment_analysis"] == 1).sum())
    positive_count = int((df_year["sentiment_analysis"] == 2).sum())

    # Crear el diccionario con el formato requerido.
    result_dict = {
        "Negative": negative_count,
        "Neutral": neutral_count,
        "Positive": positive_count
    }

    return result_dict





@app.get("/recomendacion_juego/{id_producto}")
async def recomendacion_juego(id_producto: int):
    """ Recibe un id de juego y devuelve las 5 recomendaciones más similares a un juego específico."""
    # Lectura de los archivos necesarios
    dfSimilarity = pd.read_parquet("Data/similarity")
    inde_x = pd.read_csv("Data/indexModel")
    filtered = pd.read_parquet("Data/modelFiltered")

    if id_producto not in inde_x['item_id'].values:
        return f"El ID de producto {id_producto} no está en el archivo de índices."

    indc = inde_x.loc[inde_x['item_id'] == id_producto].index[0]
    similarityes = list(enumerate(dfSimilarity[indc]))
    similarityes = sorted(similarityes, key=lambda x: x[1], reverse=True)
    similarityes = similarityes[1:6]
    indexGames = [int(i[0]) for i in similarityes]

    recomendations = [
        f"Recomendación {i+1}: {filtered['item_name'].iloc[indexGames[i]]}" 
        for i in range(len(indexGames))
    ]

    return recomendations



    
from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np 

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello, world!"}

@app.get("/PlayTimeGenre/{genero}")
async def PlayTimeGenre(genero: str):
    
    '''Recibe un género  y devuelve el año de lanzamiento con más horas jugadas.
      Digite exactamente igual  alguno de los siguientes generos para realizar la consulta 

Indie                        
Action                       
Casual                        
Adventure                     
Strategy                      
Simulation                    
RPG                           
Free to Play                  
Early Access                  
Sports                        
Massively Multiplayer         
Racing                        
Design &amp; Illustration      
Utilities                      
Web Publishing                 
Animation &amp; Modeling       
Education                      
Video Production               
Software Training              
Audio Production                
Photo Editing                   
Accounting                       

'''
   
    
  
    dfGenere = pd.read_parquet("Data/endpoint_one")
    dfGenere = dfGenere[dfGenere["géneros"] == genero]
    yearWMHours = list(dfGenere[dfGenere["tiempoJugado"] == dfGenere["tiempoJugado"].max()]["añoLanzamiento"])[0]
    return {f"Año de lanzamiento con más horas jugadas  {genero}": yearWMHours}
   



@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: str):
    
    
    '''Recibe un género y devuelve el usuario con más horas jugadas, junto al acumulado de horas por año.
       Digite exactamente igual  alguno de los siguientes generos para realizar la consulta.

Indie                        
Action                       
Casual                        
Adventure                     
Strategy                      
Simulation                    
RPG                           
Free to Play                  
Early Access                  
Sports                        
Massively Multiplayer         
Racing                        
Design &amp; Illustration      
Utilities                      
Web Publishing                 
Animation &amp; Modeling       
Education                      
Video Production               
Software Training              
Audio Production                
Photo Editing                   
Accounting                       
    '''
    
    dfEnd2 = pd.read_parquet("Data/endpoint_two")
    dfEnd2['playtime'] = round(dfEnd2['playtime'] / 60, 2)
    dfEspGen = dfEnd2[dfEnd2['genres'] == genero]
    uWMH = dfEspGen.loc[dfEspGen['playtime'].idxmax()]['user_id']
    userPYH = dfEspGen[dfEspGen['user_id'] == uWMH]
    userPYH = userPYH.groupby('release_year')['playtime'].sum().reset_index()
    userPYH = userPYH.rename(columns={'release_year': 'Año', 'playtime': 'Horas'})
    listPY = userPYH.to_dict(orient='records')
    return { f"Usuario con más horas jugadas para {genero}": uWMH, "Horas jugadas": listPY}  
        
        
@app.get("/UsersRecommend/{year}")
async def UsersRecommend(year: int):
    
    '''Recibe un año y devuelve los 3 juegos más recomendados por los usuarios australianos en el año especificado.  ej 2011 , 2012 , 2013, 2014 '''


    
    df = pd.read_parquet("Data/dfAustralianUserReviews")
    df_year = df[df["posted_year"] == year]
    df_recommend = df_year[df_year["recommend"] == True]
    df_sentiment = df_recommend[df_recommend["sentiment_analysis"].isin([2, 1])]
    df_sentiment = df_sentiment[df_sentiment["title"] != "No especificado"]
    df_sentiment["recommend"] = df_sentiment["recommend"].astype(int)
    recommendations = df_sentiment.groupby("title")["recommend"].sum()
    recommendations = recommendations.sort_values(ascending=False)
    top_3_games = recommendations.head(3).index.tolist()
    if len(top_3_games) >= 3:
        return [{"Puesto 1": top_3_games[0]}, {"Puesto 2": top_3_games[1]}, {"Puesto 3": top_3_games[2]}]
    


@app.get("/UsersNotRecommend/{year}")
async def UUsersNotRecommend(year: int):
    """
    Recibe un año y devuelve las 3 desarrolladoras con más juegos con reseñas negativas en el año especificado. ej 2011 , 2012 , 2013 , 2014 ,    
    """
    
    df = pd.read_parquet("Data/dfAustralianUserReviews") 
    dfYear = df[df["posted_year"] == year] 
    dfNotRecommend = dfYear[dfYear["recommend"] == False] 
    dfSentiment = dfNotRecommend[dfNotRecommend["sentiment_analysis"] == 0]
    dfSentiment = dfSentiment[dfSentiment["developer"] != "Otro"] 
    recommendations2 = dfSentiment.groupby("developer")["recommend"].sum() 
    recommendations2 = recommendations2.sort_values(ascending=True) 
    topNegative = recommendations2.head(3).index.tolist()

    if len(topNegative) >= 3:
        return [{"Puesto 1": topNegative[0]}, {"Puesto 2": topNegative[1]}, {"Puesto 3": topNegative[2]}]
    
    
    
@app.get("/sentiment_analysis/{año}")
async def sentiment_analysis(año: int):
    """
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios
    que se encuentren categorizados con un análisis de sentimiento.

    Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
    """
    # Cargar el DataFrame desde la ruta del archivo.
    df = pd.read_parquet("Data/dfAustralianUserReviews")

    # Filtrar los registros con el año de lanzamiento especificado.
    df_year = df[df["posted_year"] == año]

    # Contar la cantidad de veces que aparecen los valores específicos para cada etiqueta de análisis de sentimiento.
    negative_count = int((df_year["sentiment_analysis"] == 0).sum())
    neutral_count = int((df_year["sentiment_analysis"] == 1).sum())
    positive_count = int((df_year["sentiment_analysis"] == 2).sum())

    # Crear el diccionario con el formato requerido.
    result_dict = {
        "Negative": negative_count,
        "Neutral": neutral_count,
        "Positive": positive_count
    }

    return result_dict





@app.get("/recomendacion_juego/{id_producto}")
async def recomendacion_juego(id_producto: int):
    """ Recibe un id de juego y devuelve las 5 recomendaciones más similares a un juego específico."""
    # Lectura de los archivos necesarios
    dfSimilarity = pd.read_parquet("Data/similarity")
    inde_x = pd.read_csv("Data/indexModel")
    filtered = pd.read_parquet("Data/modelFiltered")

    if id_producto not in inde_x['item_id'].values:
        return f"El ID de producto {id_producto} no está en el archivo de índices."

    indc = inde_x.loc[inde_x['item_id'] == id_producto].index[0]
    similarityes = list(enumerate(dfSimilarity[indc]))
    similarityes = sorted(similarityes, key=lambda x: x[1], reverse=True)
    similarityes = similarityes[1:6]
    indexGames = [int(i[0]) for i in similarityes]

    recomendations = [
        f"Recomendación {i+1}: {filtered['item_name'].iloc[indexGames[i]]}" 
        for i in range(len(indexGames))
    ]

    return recomendations



    
