
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Încărcăm datele calculate cu spaCy
output_dir = Path("Lab01_Text_Preprocessing/results/linguistic_features")
df = pd.read_csv(output_dir / "features.csv")

# Eliminăm articolele fără caracteristici calculate
df = df.dropna()

# Caracteristicile și denumirile pentru grafice
features = {
    "ttr": "Diversitatea vocabularului (TTR)",
    "words_per_sentence": "Cuvinte pe propoziție",
    "nouns_per_sentence": "Substantive pe propoziție",
    "dependency_depth": "Adâncimea sintactică medie",
    "avg_word_length": "Lungimea medie a cuvintelor",
    "named_entity_density": "Densitatea entităților numite",
    "adjective_ratio": "Proporția adjectivelor"
}

# Denumiri mai scurte pentru categorii
category_names = {
    "comp.graphics": "Grafică",
    "rec.autos": "Automobile",
    "sci.med": "Medicină",
    "talk.politics.mideast": "Politică"
}

df["category_name"] = df["category"].map(category_names)

# Director separat pentru noile grafice
plots_dir = output_dir / "improved_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

for feature, title in features.items():

    # Calculăm mediana pe fiecare categorie
    medians = (
        df.groupby("category_name")[feature]
        .median()
        .sort_values()
    )

    # 1. Grafic cu bare orizontale
    plt.figure(figsize=(10, 5))

    bars = plt.barh(medians.index, medians.values)

    # Afișăm valoarea la capătul fiecărei bare
    plt.bar_label(
        bars,
        fmt="%.3f",
        padding=5
    )

    plt.title(f"{title} - comparația medianelor")
    plt.xlabel("Mediana")
    plt.xlim(0, max(medians.values) * 1.2)
    plt.tight_layout()

    plt.savefig(
        plots_dir / f"median_{feature}.png",
        dpi=150
    )
    plt.close()

    # 2. Boxplot fără afișarea punctelor extreme
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="category_name",
        y=feature,
        showfliers=False
    )

    plt.title(f"{title} - distribuția pe categorii")
    plt.xlabel("Categorie")
    plt.ylabel(title)
    plt.tight_layout()

    plt.savefig(
        plots_dir / f"distribution_{feature}.png",
        dpi=150
    )
    plt.close()

print("Au fost generate 14 grafice.")
