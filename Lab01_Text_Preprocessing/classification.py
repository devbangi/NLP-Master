
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
import numpy as np
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Încărcăm caracteristicile calculate anterior
csv_path = Path("Lab01_Text_Preprocessing/results/linguistic_features/features.csv")
df = pd.read_csv(csv_path)

# Eliminăm articolele fără caracteristici
df = df.dropna()

# Selectăm cele 7 caracteristici lingvistice
feature_columns = [
    "ttr",
    "words_per_sentence",
    "nouns_per_sentence",
    "dependency_depth",
    "avg_word_length",
    "named_entity_density",
    "adjective_ratio"
]

X = df[feature_columns]
y = df["category"]

# Împărțim datele: 80% antrenare, 20% testare
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

print("\nDistribuția categoriilor în train:")
print(y_train.value_counts(normalize=True).round(3))

print("\nDistribuția categoriilor în test:")
print(y_test.value_counts(normalize=True).round(3))


# Creăm un pipeline: mai întâi normalizează datele, apoi antrenează SVM-ul
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", random_state=42)
)

# Antrenăm modelul folosind datele de train
svm_model.fit(X_train, y_train)

# Prezicem categoriile articolelor din setul de test
y_pred = svm_model.predict(X_test)

# Evaluăm performanța
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== SVM - Caracteristici lingvistice ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred, zero_division=0))


# Încărcăm textele corespunzătoare articolelor din CSV
dataset = fetch_20newsgroups(
    subset="all",
    categories=[
        "sci.med", "rec.autos",
        "talk.politics.mideast", "comp.graphics"
    ],
    remove=("headers", "footers", "quotes")
)

# Aplicăm aceeași curățare Regex
def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()

# Păstrăm exact articolele din experimentul anterior
texts = [
    clean_text(dataset.data[i])
    for i in df["article_id"]
]

# Folosim aceiași indici train/test
train_texts = pd.Series(texts, index=df.index).loc[X_train.index]
test_texts = pd.Series(texts, index=df.index).loc[X_test.index]

# Bag-of-Words + SVM liniar
bow_model = make_pipeline(
    CountVectorizer(max_features=10000),
    LinearSVC(
        C=0.1,
        dual=True,
        max_iter=20000,
        random_state=42
    )
)

bow_model.fit(train_texts, y_train)
bow_pred = bow_model.predict(test_texts)

print("\n=== SVM - Bag-of-Words ===")
print(f"Accuracy: {accuracy_score(y_test, bow_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, bow_pred, average='macro'):.4f}")
print(classification_report(y_test, bow_pred, zero_division=0))

# Reprezentarea Bag-of-Words
vectorizer = CountVectorizer(max_features=10000)

X_train_bow = vectorizer.fit_transform(train_texts)
X_test_bow = vectorizer.transform(test_texts)

# Normalizăm cele 7 caracteristici lingvistice
scaler = StandardScaler()

X_train_ling = scaler.fit_transform(X_train)
X_test_ling = scaler.transform(X_test)

# Combinăm BoW cu cele 7 caracteristici
X_train_combined = hstack([
    X_train_bow,
    csr_matrix(X_train_ling)
]).tocsr()

X_test_combined = hstack([
    X_test_bow,
    csr_matrix(X_test_ling)
]).tocsr()

# Antrenăm un SVM pe reprezentarea combinată
combined_model = LinearSVC(
    C=0.1,
    dual=False,
    max_iter=20000,
    random_state=42
)

combined_model.fit(X_train_combined, y_train)
combined_pred = combined_model.predict(X_test_combined)

print("\n=== SVM - BoW + Caracteristici lingvistice ===")
print(f"Accuracy: {accuracy_score(y_test, combined_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, combined_pred, average='macro'):.4f}")

# Acum vom reprezenta fiecare articol printr-un vector numeric de 100 de dimensiuni.
# === Word2Vec: reprezentarea articolelor ===
# Tokenizăm textele, folosind aceeași împărțire train/test
train_tokens = [
    re.findall(r"[a-z]+", text.lower())
    for text in train_texts
]

test_tokens = [
    re.findall(r"[a-z]+", text.lower())
    for text in test_texts
]

# Antrenăm Word2Vec doar pe articolele de train
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=1,
    seed=42,
    epochs=10
)

# Calculăm media vectorilor cuvintelor dintr-un articol
def document_embedding(tokens, model):
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

# Construim reprezentările pentru train și test
X_train_w2v = np.array([
    document_embedding(tokens, w2v_model)
    for tokens in train_tokens
])

X_test_w2v = np.array([
    document_embedding(tokens, w2v_model)
    for tokens in test_tokens
])

print("\n=== Word2Vec ===")
print("Train:", X_train_w2v.shape)
print("Test:", X_test_w2v.shape)
print("Dimensiunea vocabularului:", len(w2v_model.wv))

# === MLP + Word2Vec ===

mlp_model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        random_state=42
    )
)

# Antrenăm rețeaua neuronală
mlp_model.fit(X_train_w2v, y_train)

# Prezicem categoriile articolelor de test
mlp_pred = mlp_model.predict(X_test_w2v)

# Evaluăm performanța
print("\n=== MLP - Word2Vec ===")
print(f"Accuracy: {accuracy_score(y_test, mlp_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, mlp_pred, average='macro'):.4f}")

print("\nRaport de clasificare:")
print(classification_report(y_test, mlp_pred, zero_division=0))

# === MLP + Word2Vec + Caracteristici lingvistice ===

# Combinăm vectorii Word2Vec cu cele 7 caracteristici
X_train_w2v_combined = np.hstack([
    X_train_w2v,
    X_train.to_numpy()
])

X_test_w2v_combined = np.hstack([
    X_test_w2v,
    X_test.to_numpy()
])

# Antrenăm MLP pe cele 107 caracteristici
mlp_combined = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        random_state=42
    )
)

mlp_combined.fit(X_train_w2v_combined, y_train)

combined_mlp_pred = mlp_combined.predict(X_test_w2v_combined)

print("\n=== MLP - Word2Vec + Caracteristici lingvistice ===")
print(f"Accuracy: {accuracy_score(y_test, combined_mlp_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, combined_mlp_pred, average='macro'):.4f}")

# === SVM + Word2Vec ===

svm_w2v = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", random_state=42)
)

svm_w2v.fit(X_train_w2v, y_train)
svm_w2v_pred = svm_w2v.predict(X_test_w2v)

print("\n=== SVM - Word2Vec ===")
print(f"Accuracy: {accuracy_score(y_test, svm_w2v_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, svm_w2v_pred, average='macro'):.4f}")


# === MLP + Bag-of-Words ===

mlp_bow = make_pipeline(
    TruncatedSVD(
        n_components=100,
        random_state=42
    ),
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        random_state=42
    )
)

# Antrenăm modelul pe reprezentarea BoW
mlp_bow.fit(X_train_bow, y_train)

# Facem predicții
mlp_bow_pred = mlp_bow.predict(X_test_bow)

# Evaluăm modelul
print("\n=== MLP - Bag-of-Words + SVD ===")
print(f"Accuracy: {accuracy_score(y_test, mlp_bow_pred):.4f}")
print(f"Macro-F1: {f1_score(y_test, mlp_bow_pred, average='macro'):.4f}")


# === Compararea și salvarea rezultatelor ===

experiments = [
    ("SVM - Linguistic", y_pred),
    ("SVM - BoW", bow_pred),
    ("SVM - BoW + Linguistic", combined_pred),
    ("SVM - Word2Vec", svm_w2v_pred),
    ("MLP - Word2Vec", mlp_pred),
    ("MLP - Word2Vec + Linguistic", combined_mlp_pred),
    ("MLP - BoW + SVD", mlp_bow_pred)
]

results = []

for model_name, predictions in experiments:
    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro"
        )
    })

# Construim tabelul cu rezultatele
results_df = pd.DataFrame(results)

# Directorul pentru rezultatele clasificării
results_dir = Path("Lab01_Text_Preprocessing/results/classification")
results_dir.mkdir(parents=True, exist_ok=True)

# Salvăm tabelul CSV
results_df.to_csv(
    results_dir / "model_comparison.csv",
    index=False
)

print("\n=== Compararea modelelor ===")
print(results_df.round(4).to_string(index=False))

# Transformăm tabelul pentru reprezentare grafică
plot_df = results_df.melt(
    id_vars="model",
    value_vars=["accuracy", "macro_f1"],
    var_name="metric",
    value_name="score"
)

# Generăm graficul
plt.figure(figsize=(12, 7))

sns.barplot(
    data=plot_df,
    x="score",
    y="model",
    hue="metric"
)

plt.title("Compararea performantei modelelor")
plt.xlabel("Scor")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.legend(title="Metrică")
plt.tight_layout()

plt.savefig(
    results_dir / "model_comparison.png",
    dpi=150
)

plt.close()

print("\nRezultatele si graficul au fost salvate.")
