import spacy
import re
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Încărcăm modelul pentru limba engleză, care poate realiza operații precum identificarea părților de vorbire și analiza sintactică.
nlp = spacy.load("en_core_web_sm")

# Text de test
text = "The doctor examined the patient in the hospital."

# Procesăm textul cu spaCy si vom obtine in obiectul doc tokenurile si informatiile lingvistice identificate
doc = nlp(text)

# Afișăm cuvintele și părțile lor de vorbire; un token poate fi un cuvant sau un semn de punctuatie
for token in doc:
    print(token.text, "->", token.pos_) #pos_ = part of speech(partea de vorbire identificata) pt token

# 1. Caracteristica TTR(Type-Token Ratio): masoara cat la suta din totalul cuvintelor din text sunt distincte
# Cu cât TTR este mai mare, cu atât vocabularul este mai divers.
def calculate_ttr(doc):
    # Selectăm doar cuvintele, fără punctuație și spații
    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha # Verifică dacă tokenul conține doar caractere alfabetice. Astfel, ignorăm punctuația și numerele.
    ]

    # Evităm împărțirea la zero
    if not words:
        return 0.0

    # Numărul de cuvinte distincte / numărul total de cuvinte
    return len(set(words)) / len(words)

# Testăm funcția
text = "The car is fast. The car is red."
doc = nlp(text)

print("Text:", text)
print("TTR:", calculate_ttr(doc))

# 2. Caracteristica Words per sentence — măsoară lungimea medie a propozițiilor dintr-un text.

def calculate_words_per_sentence(doc):
    # Obținem propozițiile identificate de spaCy
    sentences = list(doc.sents)

    # Evităm împărțirea la zero
    if not sentences:
        return 0.0

    # Numărăm doar cuvintele, fără punctuație și numere
    words = [token for token in doc if token.is_alpha]

    # Calculăm numărul mediu de cuvinte pe propoziție
    return len(words) / len(sentences) #(Numărul total de cuvinte\Numărul total de propoziții)

# Testăm primele două caracteristici
text = "The car is fast. The engine works very well."
doc = nlp(text)

print("Text:", text)
print("TTR:", calculate_ttr(doc))
print("Words per sentence:", calculate_words_per_sentence(doc))

# 3. Caracteristica Nouns per sentence: măsoară câte substantive apar, în medie, într-o propoziție.
# Pentru această caracteristică vom număra atât substantivele comune (NOUN), cât și substantivele proprii (PROPN).

def calculate_nouns_per_sentence(doc):
    # Obținem propozițiile identificate de spaCy
    sentences = list(doc.sents)

    # Evităm împărțirea la zero
    if not sentences:
        return 0.0

    # Numărăm substantivele comune și proprii
    nouns = [
        token for token in doc
        if token.pos_ in ("NOUN", "PROPN")
    ]

    # Calculăm media substantivelor pe propoziție
    return len(nouns) / len(sentences) # Numărul total de substantive/Numărul total de propoziții

# Testăm primele trei caracteristici
text = "The doctor examined the patient. The nurse works in the hospital."
doc = nlp(text)

print("Text:", text)
print("TTR:", calculate_ttr(doc))
print("Words per sentence:", calculate_words_per_sentence(doc))
print("Nouns per sentence:", calculate_nouns_per_sentence(doc))

# 4. Caracteristica Dependency Parse Depth: este procesul prin care spaCy identifică relațiile gramaticale dintre cuvintele unei propoziții.
def calculate_dependency_depth(doc):
    depths = []

    for token in doc:
        # Ignorăm punctuația și alte tokenuri nealfabetice
        if not token.is_alpha:
            continue

        depth = 0
        current = token

        # Urcăm în arbore până ajungem la rădăcină
        while current.head != current:
            depth += 1
            current = current.head

        depths.append(depth)

    # Evităm împărțirea la zero
    if not depths:
        return 0.0

    # Calculăm adâncimea medie
    return sum(depths) / len(depths)

text = "The doctor examined the patient."
doc = nlp(text)

print("Text:", text)
print("Dependency depth:", calculate_dependency_depth(doc))

# Afișăm pentru fiecare token legătura și adâncimea
for token in doc:
    if token.is_alpha:
        depth = 0
        current = token

        while current.head != current:
            depth += 1
            current = current.head

        print(
            token.text,
            "| head:", token.head.text,
            "| dependency:", token.dep_,
            "| depth:", depth
        )

# 5. Caracteristica Average Word Length (lungimea medie a cuvintelor)
def calculate_avg_word_length(doc):
    # Selectăm doar tokenurile alfabetice
    words = [token.text for token in doc if token.is_alpha]

    # Evităm împărțirea la zero
    if not words:
        return 0.0

    # Calculăm numărul total de caractere
    total_characters = sum(len(word) for word in words)

    # Calculăm lungimea medie a cuvintelor
    return total_characters / len(words)

text = "The doctor examined the patient."
doc = nlp(text)

print("Text:", text)
print("Average word length:", calculate_avg_word_length(doc))

# 6. Caracteristica Named Entity Density: o Named Entity (entitate numită) este o expresie care identifică o persoană, o organizație, o locație, o dată sau alte tipuri de entități.
def calculate_named_entity_density(doc):
    # Selectăm doar cuvintele alfabetice
    words = [token for token in doc if token.is_alpha]

    # Evităm împărțirea la zero
    if not words:
        return 0.0

    # Numărăm cuvintele care fac parte dintr-o entitate numită
    entity_words = [
        token for token in words
        if token.ent_iob_ != "O"
    ]

    # Calculăm densitatea entităților
    return len(entity_words) / len(words)

text = "Barack Obama visited Paris in 2015."
doc = nlp(text)

print("Text:", text)
print("Named entity density:", calculate_named_entity_density(doc))

print("\nEntități identificate:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

# 7. Caracteristica Adjective Ratio (proporția adjectivelor)
def calculate_adjective_ratio(doc):
    # Selectăm doar cuvintele alfabetice
    words = [token for token in doc if token.is_alpha]

    # Evităm împărțirea la zero
    if not words:
        return 0.0

    # Selectăm adjectivele
    adjectives = [
        token for token in words
        if token.pos_ == "ADJ"
    ]

    # Calculăm proporția adjectivelor
    return len(adjectives) / len(words)

text = "The red car has a powerful engine."
doc = nlp(text)

print("Text:", text)
print("Adjective ratio:", calculate_adjective_ratio(doc))

print("\nAdjective identificate:")
for token in doc:
    if token.pos_ == "ADJ":
        print(token.text)


# Extragerea celor 7 caracteristici lingvistice din întregul dataset 20 Newsgroups
categories = [
    "sci.med",
    "rec.autos",
    "talk.politics.mideast",
    "comp.graphics"
]

dataset = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

print("Număr total de articole:", len(dataset.data))

def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_articles = [
    clean_text(article)
    for article in dataset.data
]

# Lista în care vom stoca rezultatele; Creăm o listă în care vom adăuga câte un dicționar pentru fiecare articol.
results = []

# Procesăm articolele în loturi
docs = nlp.pipe(cleaned_articles, batch_size=64)

for i, doc in enumerate(docs):

    # Identificăm categoria articolului
    category = dataset.target_names[dataset.target[i]]

    # Verificăm dacă articolul conține cuvinte
    has_words = any(token.is_alpha for token in doc)

    # Calculăm cele 7 caracteristici
    features = {
        "article_id": i,
        "category": category,
        "ttr": calculate_ttr(doc) if has_words else None,
        "words_per_sentence": calculate_words_per_sentence(doc) if has_words else None,
        "nouns_per_sentence": calculate_nouns_per_sentence(doc) if has_words else None,
        "dependency_depth": calculate_dependency_depth(doc) if has_words else None,
        "avg_word_length": calculate_avg_word_length(doc) if has_words else None,
        "named_entity_density": calculate_named_entity_density(doc) if has_words else None,
        "adjective_ratio": calculate_adjective_ratio(doc) if has_words else None
    }

    results.append(features)

    # Afișăm progresul la fiecare 100 de articole
    if (i + 1) % 100 == 0:
        print(f"Procesate {i + 1}/{len(cleaned_articles)} articole")

# Convertim rezultatele într-un tabel pandas
df = pd.DataFrame(results)

print("\nPrimele 5 articole:")
print(df.head())


# Directorul pentru rezultatele cerinței 2
output_dir = Path("Lab01_Text_Preprocessing/results/linguistic_features")
output_dir.mkdir(parents=True, exist_ok=True)

# Salvăm tabelul
df.to_csv(output_dir / "features.csv", index=False)

print("\nRezultatele au fost salvate în features.csv")
print("Dimensiunea tabelului:", df.shape)

print("\nNumăr de articole pe categorie:")
print(df["category"].value_counts())

print("\nValori lipsă pentru fiecare coloană:")
print(df.isna().sum())

print("\nStatistici descriptive:")
print(df.describe().round(3))


# Plot feature distributions per class to observe discriminative patterns.

