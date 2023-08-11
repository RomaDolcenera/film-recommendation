from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app= FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


dfm = pd.read_csv("./movies_clean.csv",sep = ',', encoding="UTF-8", low_memory=False)
dfc = pd.read_csv("./credits_clean.csv")

@app.get("/")
def read_root():
    return {"mensaje": "Bienvenido a la API de películas"}


#Se ingresa un idioma, deve retornar la cantidad de peliculas estrenadas en ese idioma.
@app.get("/peliculas_idioma/{idioma}")
def read_item(idioma: str):
    idioma = idioma.lower()
    result_filter = dfm.loc[dfm['original_language'].str.lower() == idioma]
    if result_filter.empty:
        return {"error": "Ingrese un idioma válido"}
    else:
        suma = result_filter.shape[0]
        mensaje = f'{suma} cantidad de películas fueron estrenadas en idioma {idioma}'
        return {"mensaje": mensaje}
        

#Se ingresa una pelicula, deve retornar duracion y el año de lanzamiento.
@app.get("/peliculas_duracion/{titulo}")
def read_item(titulo: str):
    title = titulo.lower()
    result_filter = dfm.loc[dfm['title'].str.lower() == title]
    if result_filter.empty:
        return {"error": "La pelicula ingresada no se encuentra en la base de datos, pruebe con el nombre completo o ingrese otra pelicula"}
    else:
        release_year = result_filter['release_year'].values[0]
        runtime = result_filter['runtime'].values[0]
        mensaje = f'La película {titulo} fue estrenada en el año {release_year} y tiene una duración de {runtime} minutos'
        return {"mensaje": mensaje}

# def franquicia( Franquicia: str ): Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio. Ejemplo de retorno: La franquicia X posee X peliculas, una ganancia total de x y una ganancia promedio de xx
@app.get("/franquicia/{franquicia}")
def read_item(franquicia: str):
    franquicia = franquicia.lower()
    result_filter = dfm.loc[dfm['belongs_to_collection'].apply(lambda x: isinstance(x, str) and franquicia in x.lower())]

    if result_filter.empty:
        return {"error": "La franquicia ingresada no se encuentro en la base de datos, pruebe con el nombre completo o ingrese otra franquicia"}
    else:
        suma = result_filter.shape[0]
        ganancia_total = result_filter['revenue'].sum()
        ganancia_promedio = result_filter['revenue'].mean()
        mensaje = f'La franquicia {franquicia} posee {suma} peliculas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}'
        return {"mensaje": mensaje}

#Se ingresa un país y retornana la cantidad de peliculas producidas en el mismo.
@app.get("/peliculas_pais/{pais}")
def read_item(pais: str):
    pais = pais.lower()
    result_filter = dfm.loc[dfm['production_countries'].str.lower().apply(lambda x: pais in x)]
    if result_filter.empty:
        return {"error": "El país ingresado no se encuentro en la base de datos, pruebe con el nombre completo o ingrese otro país"}
    else:
        suma = result_filter.shape[0]
        mensaje = f'El país {pais} produjo {suma} películas'
        return {"mensaje": mensaje}

#def productoras_exitosas( Productora: str ): Se ingresa la productora, entregandote el revunue total y la cantidad de peliculas que realizo.
@app.get("/productoras_exitosas/{productora}")
def read_item(productora: str):
    productora = productora.lower()
    result_filter = dfm.loc[dfm['production_companies'].str.lower().apply(lambda x: productora in x)]
    if result_filter.empty:
        return {"error": "La productora ingresada no se encuentro en la base de datos, pruebe con el nombre completo o ingrese otra productora"}
    else:
        suma = result_filter.shape[0]
        ganancia_total = result_filter['revenue'].sum()
        mensaje = f'La productora {productora} realizó {suma} películas, con una ganancia total de {ganancia_total}'
        return {"mensaje": mensaje}


#Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma, en formato lista.
@app.get("/get_director/{director}")
def read_item(director: str):
    name_director = director.lower()
    result_filter = dfc.loc[dfc['directors'].str.lower().apply(lambda x: name_director in x)]
    if result_filter.empty:
        return {"error": "El director ingresado no se encuentro en la base de datos, pruebe con el nombre completo o ingrese otro director"}
    else:
        id_movies = result_filter['id'].to_list()
        movies_director = dfm[dfm['id'].isin(id_movies)]
        best_movie = movies_director.loc[movies_director['popularity'].idxmax()]
        column_rename = {'title': 'Titulo', 'release_year': 'Año de lanzamiento', 'return': 'Retorno', 'budget': 'Costo', 'revenue': 'Ganancia'}
        list_movies = movies_director[['title','release_year', 'return', 'budget', 'revenue']].rename(columns=column_rename).to_dict('records')
        return {
            "Director": name_director,
            "Mejor Pelicula": best_movie['title'],
            "Lista de Peliculas": list_movies
            }


@app.get("/recomendacion/{titulo}")
def read_item(titulo: str):

    def get_recommendations(titulo, dfm):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(dfm['title'])
        search_vector = tfidf_vectorizer.transform([titulo])
        similar_score = cosine_similarity(search_vector, tfidf_matrix).flatten()
        dfm['score'] = similar_score
        dfm = dfm.sort_values(by='score', ascending=False)
        result_dict = {}
        for index in range(5):
            result_dict[dfm.iloc[index]['title']] = dfm.iloc[index]['score']
        return result_dict

    recommend = get_recommendations(titulo, dfm)

    if len(recommend) < 0:
        return {"error": "La pelicula ingresada no se encuentro en la base de datos, pruebe con el nombre completo o ingrese otra pelicula"}
    else:
        return {"Peliculas similares recomendadas": recommend}
    
@app.get("/health")
def health():
    return {"status": "ok"}