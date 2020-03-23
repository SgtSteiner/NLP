import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


EJEMPLO_TEXT = "España va a sufrir la primera ola de calor del verano a partir del miércoles 1 de agosto, " \
               "que se extenderá \"por lo menos hasta el domingo\", ha confirmado en un aviso especial la Agencia Estatal " \
               "de Meteorología (Aemet), que ya alertó el pasado viernes de que era muy probable que se produjera. " \
               "Como valores máximos, según precisa a este diario uno de sus portavoces, Rubén del Campo, " \
               "se pueden llegar a alcanzar los 45 grados en los valles del Guadalquivir y del Guadiana. Pero lo más " \
               "destacable del fenómeno, que del solo se escaparán Almería, Ceuta y Melilla y Canarias, es que las " \
               "mínimas también van a ser muy altas y se espera que en el suroeste los termómetros no bajen de los 25 " \
               "grados por la noche. El Sr. Julián es un campeón."

EJEMPLO_DOS = "Esta situación sofocante persiste hasta el domingo. \"El lunes parece que bajan un poco las " \
              "temperaturas por el noroeste, pero aún es pronto para dar por remitida la ola el lunes\", explica. De " \
              "momento, también es pronto para saber si los avisos llegarán al nivel rojo, el máximo en una escala de " \
              "tres en el que el amarillo es el más bajo —no hay riesgo para la población en general aunque sí para " \
              "alguna actividad concreta—; el naranja, el intermedio —riesgo importante por fenómenos meteorológicos " \
              "no habituales y con cierto grado de peligro para las actividades usuales— y el rojo —riesgo extremo, " \
              "con fenómenos infrecuentes de intensidad excepcional y con un nivel de riesgo para la población muy " \
              "alto—. "

# Sentence tokenize
frases = sent_tokenize(EJEMPLO_TEXT, language="spanish")
print(frases)

# Word tokenize
palabras = word_tokenize(EJEMPLO_TEXT, language="spanish")
print(palabras)

# Stop words (palabras "de relleno")
stop_words = stopwords.words("spanish")
sentencias_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
print(sentencias_filtradas)

# Stem
ps = PorterStemmer()
palabras_stem = [ps.stem(palabra) for palabra in palabras]
print(palabras_stem)

custom_sent_tokenizer = PunktSentenceTokenizer(EJEMPLO_TEXT)
tokenized = custom_sent_tokenizer.tokenize(EJEMPLO_DOS)
for i in tokenized:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    print(tagged)
