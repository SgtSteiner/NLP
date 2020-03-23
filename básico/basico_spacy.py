import spacy
from spacy import displacy
from spacy.lang.es.stop_words import STOP_WORDS

nlp = spacy.load("es")
ejemplo = "La marcha de Cristiano Ronaldo, digerida con aparente naturalidad por la directiva del Real Madrid, " \
          "ha puesto al club en tensión ante la necesidad de reforzar una plantilla que exhibió síntomas de desgaste " \
          "en la última Liga. Si el mensaje hacia el exterior es de optimismo tras la conquista de la Decimotercera, " \
          "las fuentes consultadas en las oficinas de Chamartín señalan un debate profundo. Estos observadores " \
          "advierten que el presidente Florentino Pérez quería a Mbappé y a Neymar, pero que las negativas del " \
          "francés en 2017 y el brasileño en 2018, han forzado un cambio de planes en una institución que se preparó " \
          "para desembolsar más de 350 millones de euros para conseguir una primera figura mundial. Ante la duda, " \
          "ahora los estrategas madridistas optan por sondear el mercado, esperar y, si acaso, ahorrar en un " \
          "escenario insólito. Solo Hazard, de entre los atacantes contactados que más brillaron en la temporada " \
          "pasada, se ha manifestado claramente decidido a fichar por el Madrid. Pero el Madrid le ha respondido con " \
          "un gentil formalismo evasivo."

doc = nlp(ejemplo)

tokens = [token for token in doc]
non_stop = [word for word in doc if not word.is_stop]
for word in doc:
    print(word.text,
          word.lemma_,
          word.pos_,
          word.tag_,
          word.dep_,
          word.shape_,
          word.is_stop,)

html = displacy.render(doc, style='dep', page=True, options={'distance': 70})
print(html)

for ent in doc.ents:
    print(ent.text,
          ent.label_)
