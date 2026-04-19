import spacy

nlp = spacy.load("en_core_web_sm")
text = "Kiran is working on an AI project in Pakistan in 2026."
doc = nlp(text)
for ent in doc.ents:
    print(f"Entity : {ent.text} | Label : {ent.label_} ({spacy.explain(ent.label_)})")