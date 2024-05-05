#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import emoji
import nltk
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Inicializar NLTK (si no está inicializado)
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Inicializar Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    
    # Leer el DataFrame

    # Definir las funciones antes de usarlas en process_data()

    archivo_csv = request.files['archivo_csv']
    #global df
    df = pd.read_csv(archivo_csv, encoding='utf-8')

    # Convertir la columna de fechas a tipo datetime si no está en ese formato
    df['date'] = pd.to_datetime(df['date'])

    # Definir el rango de fechas deseado (puedes obtener estas fechas del usuario de alguna manera)
    fecha_inicio = request.form['fecha_inicio']
    fecha_fin = request.form['fecha_fin']

    # Acotar el DataFrame al rango de fechas ingresado por el usuario
    df = df[(df['date'] >= fecha_inicio) & (df['date'] <= fecha_fin)]

    stop_words = set(stopwords.words('english'))
    documentos_tokenizados = [" ".join([token for token in nltk.word_tokenize(documento.lower()) if token.isalnum() and token not in stop_words]) for documento in df['translated_tweet']]

    # Representación de datos utilizando TF-IDF
    vectorizador_tfidf = TfidfVectorizer()
    X = vectorizador_tfidf.fit_transform(documentos_tokenizados)

    # Método de la silueta para determinar el número óptimo de clusters
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)

    # Encontrar el número óptimo de clusters
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  
    def get_clusters(texts):
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans.fit(X)
        clusters_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        clusters_df['Cluster'] = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        cluster_keywords = []
        for center in cluster_centers:
            keywords = [vectorizer.get_feature_names_out()[i] for i in center.argsort()[:-10 - 1:-1]]
            cluster_keywords.append(keywords)
        return clusters_df, plot_cluster_keywords(cluster_keywords)



    def analyze_sentiment(comment):
        sid = SentimentIntensityAnalyzer()
        compound_score = sid.polarity_scores(comment)['compound']
        if compound_score > 0.05:
            sentiment = 'Positivo'
        elif compound_score < -0.05:
            sentiment = 'Negativo'
        else:
            sentiment = 'Neutral'
        return sentiment, compound_score

    def get_top_phrases(texts):
        stop_words = set(stopwords.words('english'))
        phrases = []
        for text in texts:
            words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
            three_word_phrases = list(ngrams(words, 3))
            phrases.extend(three_word_phrases)
        phrase_counts = Counter(phrases)
        top_phrases = phrase_counts.most_common(10)
        top_phrases_df = pd.DataFrame(top_phrases, columns=['Phrase', 'Frequency'])
        return top_phrases_df

    def get_clusters(texts):
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans.fit(X)
        clusters_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        clusters_df['Cluster'] = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        cluster_keywords = []
        for center in cluster_centers:
            keywords = [vectorizer.get_feature_names_out()[i] for i in center.argsort()[:-10 - 1:-1]]
            cluster_keywords.append(keywords)
        return clusters_df, plot_cluster_keywords(cluster_keywords)

    def plot_cluster_keywords(cluster_keywords):
        fig = go.Figure()
        for i, keywords in enumerate(cluster_keywords):
            fig.add_trace(go.Scatter(x=np.arange(len(keywords)), y=np.ones(len(keywords)) * i,
                                    mode='markers+text',
                                    marker=dict(size=10),
                                    text=keywords,
                                    textposition="top center",
                                    name=f'Cluster {i+1}'))
        fig.update_layout(title='Palabras clave por cluster',
                        yaxis=dict(title='Cluster'),
                        xaxis=dict(title='Palabra'))
        return fig

    def plot_sentiment_variation(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['Sentimiento'].apply(lambda x: x[1]),
                                mode='lines',
                                name='Sentimiento',
                                line=dict(color='royalblue', width=2)))
        fig.update_layout(title='Variación del Sentimiento con el Tiempo',
                        xaxis=dict(title='Fecha'),
                        yaxis=dict(title='Sentimiento'))
        return fig




    # Realizar el análisis de sentimientos
    df['Sentimiento'] = df['translated_tweet'].apply(analyze_sentiment)

    # Realizar el análisis de frecuencia de frases compuestas
    top_phrases_df = get_top_phrases(df['translated_tweet'])

    # Realizar el análisis de clusters
    clusters_df, fig_clusters = get_clusters(df['translated_tweet'])

    # Convertir la figura de clusters de Plotly a HTML
    clusters_html = fig_clusters.to_html(full_html=False)

    # Obtener la gráfica de variación del sentimiento con el tiempo
    fig_sentiment = plot_sentiment_variation(df)

    # Convertir la figura de Plotly de variación del sentimiento a HTML
    sentiment_html = fig_sentiment.to_html(full_html=False)

    # Convertir DataFrame de las frases más comunes a HTML
    top_phrases_html = top_phrases_df.to_html(index=False)

    return render_template('resultados.html', clusters_html=clusters_html, sentiment_html=sentiment_html, top_phrases_html=top_phrases_html)

if __name__ == '__main__':
    app.run(debug=True)

