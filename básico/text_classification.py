import string
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# Limpia el texto eliminando pronombres personales, stopwords y signos de puntuación
def limpia_texto(docs, logging=False):
    textos = []
    contador = 1
    for doc in docs:
        if contador % 1000 == 0 and logging:
            print("Procesados {} de {} documentos".format(contador, len(docs)))
        contador += 1
        doc = nlp(doc, disable=["parser", "ner"])   # Desactivamos parser y ner, no los necesitamos -> agilizamos
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
        tokens = [tok for tok in tokens if tok not in STOP_WORDS and tok not in string.punctuation]
        tokens = " ".join(tokens)
        textos.append(tokens)
    return pd.Series(textos)


"""
Importación y carga del dataset.
El dataset contiene 2.507 títulos cortos de papers de investigación, clasificados en 5 categorías.
"""

dataset_filename = "data\\research_paper.csv"
df = pd.read_csv(dataset_filename)

print("\nComprobamos que no existen valores nulos:")
print("-----------------------------------------")
print(df.isnull().sum())

# Dividimos los datos en los dataset de entrenamiento y test
print("\nDividimos los datos en los dataset de entrenamiento y test:")
print("------------------------------------------------------------")
X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)
print("Tamaño dataset entrenamiento: {}".format(X_train.shape))
print("Tamaño dataset test: {}".format(X_test.shape))

# Dibujamos la distribución de los papers según categoría
fig = plt.figure(figsize=(8, 4))
sns.barplot(x=X_train["Conference"].unique(),
            y=X_train["Conference"].value_counts())
plt.show()

# Preprocesamiento de texto. Vamos a encontrar las palabras más utilizadas para la primera y segunda categoría
# INFOCOM e ISCAS, respectivamente
nlp = spacy.load("en")

# Seleccionamos los papers de las categorías INFOCOM e ISCAS
INFOCOM_texto = [texto for texto in X_train[X_train["Conference"] == "INFOCOM"]["Title"]]
ISCAS_texto = [texto for texto in X_train[X_train["Conference"] == "ISCAS"]["Title"]]

# Limpiamos el texto
INFOCOM_limpio = limpia_texto(INFOCOM_texto, logging=True)
INFOCOM_limpio = " ".join(INFOCOM_limpio).split()

ISCAS_limpio = limpia_texto(ISCAS_texto, logging=True)
ISCAS_limpio = " ".join(ISCAS_limpio).split()

# Generamos un diccionario con las palabas más utilizadas
INFOCOM_counts = Counter(INFOCOM_limpio)
ISCAS_counts = Counter(ISCAS_limpio)

# Seleccionamos las 20 palabas más usadas
INFOCOM_palabras_comunes = [word[0] for word in INFOCOM_counts.most_common(20)]
INFOCOM_counts_comunes = [word[1] for word in INFOCOM_counts.most_common(20)]

# Dibujamos un grafico de barras para visualizar las 20 palabras más comunes
fig = plt.figure(figsize=(18, 6))
sns.barplot(x=INFOCOM_palabras_comunes, y=INFOCOM_counts_comunes)
plt.title('Palabras más comunes usadas en los papers de la categoría INFOCOM')
plt.show()

ISCAS_palabras_comunes = [word[0] for word in ISCAS_counts.most_common(20)]
ISCAS_counts_comunes = [word[1] for word in ISCAS_counts.most_common(20)]

fig = plt.figure(figsize=(18, 6))
sns.barplot(x=ISCAS_palabras_comunes, y=ISCAS_counts_comunes)
plt.title('Palabras más comunes usadas en los papers de la categoría ISCAS')
plt.show()
