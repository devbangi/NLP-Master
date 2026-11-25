# Clasificarea automată a documentelor juridice în limba română

**Proiect individual – Natural Language Processing (NLP), master**

## Despre proiect

Scopul proiectului este clasificarea automată a actelor legislative europene în **21 de domenii EUROVOC**, folosind texte în limba română din [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex). Sarcina este de tip **multi-label**: un document poate primi simultan mai multe categorii.

Comparăm trei abordări:

1. **Fine-tuning de encodere** – XLM-RoBERTa și Legal-XLM-Longformer.
2. **Few-shot LLM + RAG** – Qwen3-4B primește exemple etichetate, recuperate automat din train, fără antrenarea modelului.
3. **SFT cu QLoRA** – adaptarea Qwen3-4B prin antrenarea unor adaptoare LoRA. **În curs de antrenare; nu există încă rezultate finale.**

## Datele utilizate

| Partiție | Documente |
|---|---:|
| Train | 15.920 |
| Validation | 5.000 |
| Test | 5.000 |
| **Total** | **25.920** |

Din setul original de train a fost eliminat un duplicat textual. Analiza exploratorie a evidențiat documente cu lungimi foarte variate și categorii dezechilibrate. Datele se păstrează local și nu sunt publicate în repository.

## Metodologie

### Modelul 1 – Encodere fine-tuned

- **XLM-RoBERTa-base:** maximum 512 tokeni; 6 epoci în două faze (3 + 3).
- **Legal-XLM-Longformer-base:** maximum 2.048 tokeni; 6 epoci; selecția checkpoint-ului pe validation.
- Ambele modele prezic independent cele 21 de etichete. Pragurile globale au fost alese pe validation: **0,40** pentru XLM-RoBERTa și **0,45** pentru Longformer.

### Modelul 2 – Few-shot Qwen + RAG

Am folosit **Qwen3-4B-Instruct-2507**, cuantizat la **4 biți (NF4)** pentru a rula pe GPU-ul disponibil.

Pentru fiecare document:

1. **TF-IDF** identifică cele mai asemănătoare 6 documente din train, pe baza similarității cosinus.
2. Textul nou, exemplele recuperate și etichetele lor sunt incluse într-un prompt pentru Qwen.
3. Qwen generează un **array JSON** cu categoriile EUROVOC prezise.

Am comparat prompturi cu 3 și 6 exemple fixe și varianta cu recuperare dinamică (RAG). Pe un eșantion preliminar de 50 de documente de validation, RAG a avut rezultate mai bune; configurația aleasă a fost apoi verificată pe 300 de documente de validation (**Micro-F1 0,7921; Macro-F1 0,6824**). În experiment, documentul nou a fost limitat la 2.000 de caractere, iar fiecare exemplu recuperat la 1.000 de caractere.

Ca verificare suplimentară, am comparat RAG și cu un baseline **TF-IDF nearest-neighbor**, fără LLM. Această analiză arată că recuperarea documentelor juridice asemănătoare contribuie semnificativ la performanță.

### Modelul 3 – SFT cu QLoRA (în desfășurare)

Folosim același Qwen3-4B, cu greutățile de bază cuantizate la 4 biți și înghețate; antrenăm doar **adaptoarele LoRA** (aproximativ **2,95 milioane de parametri**, adică **0,0733%** din total).

Configurația pregătită: 15.920 documente de train, maximum 2.048 tokeni per exemplu (din care maximum 1.400 tokeni pentru document), maximum 6 epoci, monitorizarea *validation loss* pe 500 de exemple, **early stopping** și selectarea celui mai bun checkpoint. Antrenarea folosește BF16. **Rezultatele vor fi adăugate după finalizarea evaluării.**

## Rezultate finale pe test

Toate rezultatele de mai jos sunt calculate pe **aceleași 5.000 de documente de test**.

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| XLM-RoBERTa | 0,7861 | 0,6467 |
| **Legal-XLM-Longformer** | **0,8067** | **0,6870** |
| Qwen3-4B + TF-IDF RAG | 0,7468 | 0,6433 |

**Micro-F1** măsoară performanța globală asupra etichetelor; **Macro-F1** oferă aceeași pondere fiecărei categorii, inclusiv celor rare.

**Până în prezent, Legal-XLM-Longformer obține cele mai bune rezultate.** Qwen + RAG clasifică fără fine-tuning, dar este mai lent la inferență și are scoruri mai mici. Diferențele dintre modele nu pot fi atribuite exclusiv arhitecturii, deoarece diferă și contextul disponibil și modul de adaptare.

## Organizarea proiectului

```text
Project_multi_eurlex/
├── Data/                         # Date locale (nepublicate)
├── Models/                       # Greutăți și adaptoare locale (nepublicate)
├── Notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_finetuning_xlm_roberta.ipynb
│   ├── 03_finetuning_legal_xlm_longformer.ipynb
│   ├── 04_few_shot_llm.ipynb
│   └── 05_sft_lora_qwen.ipynb    # În lucru
├── Results/                      # Metrici, grafice și predicții
└── README.md
```

**Mediu:** WSL2 Ubuntu, Python 3.11, PyTorch, Transformers, scikit-learn, PEFT și TRL; NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM). Seturile de date și greutățile modelelor nu sunt incluse în Git.

## Surse principale

- [MultiEURLEX – dataset](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [Legal-XLM-Longformer-base](https://huggingface.co/joelniklaus/legal-xlm-longformer-base)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [LoRA – Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
