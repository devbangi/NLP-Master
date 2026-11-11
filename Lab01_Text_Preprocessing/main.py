import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path

# 1.1.Load the dataset and select the 4 categories.

from sklearn.datasets import fetch_20newsgroups
# Setul de date 20 Newsgroups este unul dintre cele mai cunoscute seturi de date din domeniul NLP (Procesarea Limbajului Natural) și al învățării automate (Machine Learning), folosit în special pentru sarcini de clasificare a textului și clustering.

# Cele 4 categorii cerute în laborator
categories = [
    "sci.med",
    "rec.autos",
    "talk.politics.mideast",
    "comp.graphics"
]

# Încărcăm toate articolele din categoriile selectate
dataset = fetch_20newsgroups(
    subset="all",
    categories=categories
)

# Afișăm informații generale despre dataset
print("Număr total de articole:", len(dataset.data))
print("Categorii:", dataset.target_names)

# Inspectăm primul articol
print("\nPrimul articol:\n")
print(dataset.data[0][:1000])

# Categoria din care face parte primul articol
print("\nCategoria articolului:")
print(dataset.target_names[dataset.target[0]])


# 1.2.Remove URLs and collapse repeated whitespace to prepare the formal text for analysis.


def clean_text(text):
    # Eliminăm URL-urile care încep cu http://, https:// sau www.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Înlocuim spațiile, taburile și liniile noi consecutive
    # cu un singur spațiu
    text = re.sub(r"\s+", " ", text)

    return text.strip()

cleaned_articles = []

for article in dataset.data:
    cleaned_articles.append(clean_text(article))

print("\nArticol original:")
print(dataset.data[0][:500])

print("\nArticol curățat:")
print(cleaned_articles[0][:500])

print("\nNumăr articole curățate:", len(cleaned_articles))

test_text = "Visit https://example.com    for more information."
print(clean_text(test_text))


# 1.2.Generate a word cloud with content words for all articles in each category.

# Content words — cuvinte care transmit informații despre subiect: doctor, disease, engine, graphics.
# Stopwords — cuvinte foarte frecvente, precum the, and, is, of, care de obicei nu ne ajută să identificăm subiectul unui document.
# Pentru acest laborator voi folosi lista de stopwords din biblioteca wordcloud.

# Grupăm articolele curățate pe categorii
articles_by_category = {}

for category in dataset.target_names:
    articles_by_category[category] = []

for article, target in zip(cleaned_articles, dataset.target):
    category = dataset.target_names[target]
    articles_by_category[category].append(article)

# Verificăm numărul de articole din fiecare categorie
for category, articles in articles_by_category.items():
    print(f"{category}: {len(articles)} articole")

output_dir = Path("Lab01_Text_Preprocessing/results/original")
output_dir.mkdir(parents=True, exist_ok=True)
# Generăm câte un word cloud pentru fiecare categorie
for category, articles in articles_by_category.items():

    # Combinăm toate articolele categoriei într-un singur text
    category_text = " ".join(articles)

    # Construim word cloud-ul
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=100,
        collocations=False
    ).generate(category_text)

    # Afișăm imaginea
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {category}")
    plt.savefig(
        output_dir / f"{category}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

# OBSERVAȚIE:
# Norii de cuvinte (word clouds) conțin nu doar cuvinte specifice subiectului (de ex., "car",
# "engine", "medical", "image"), ci și termeni precum "Subject",
# "Organization", "University", "Posting" și "NNTP".
#
# Acești termeni provin în principal din antetele articolelor (headers), metadate și
# mesajele citate din setul de date original 20 Newsgroups.
#
# Deși cuvintele de legătură (stopwords) sunt eliminate, acești termeni legați de metadate rămân
# deoarece nu sunt stopwords standard în limba engleză.
#
# Pentru a investiga influența lor, vom genera un al doilea set de
# nori de cuvinte după eliminarea antetelor (headers), subsolurilor (footers) și a textului citat.

# Load the dataset again, this time removing metadata and quoted text
dataset_no_metadata = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

cleaned_articles_no_metadata = [
    clean_text(article) for article in dataset_no_metadata.data
]

articles_by_category_no_metadata = {}

for category in dataset_no_metadata.target_names:
    articles_by_category_no_metadata[category] = []

for article, target in zip(
    cleaned_articles_no_metadata,
    dataset_no_metadata.target
):
    category = dataset_no_metadata.target_names[target]
    articles_by_category_no_metadata[category].append(article)


# Generate word clouds without metadata
output_dir = Path("Lab01_Text_Preprocessing/results/without_metadata")
output_dir.mkdir(parents=True, exist_ok=True)

for category, articles in articles_by_category_no_metadata.items():

    category_text = " ".join(articles)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=100,
        collocations=False
    ).generate(category_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {category} (without metadata)")

    plt.savefig(
        output_dir / f"{category}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
