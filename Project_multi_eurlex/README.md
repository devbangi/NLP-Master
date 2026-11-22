# Clasificarea automată a documentelor juridice în limba română

**Proiect individual – Natural Language Processing (NLP), Master**

## 1. Descrierea proiectului

Proiectul investighează clasificarea automată multi-label a documentelor juridice în limba română, utilizând datasetul **MultiEURLEX**.

Scopul este identificarea automată a domeniilor tematice EUROVOC asociate unui document legislativ. Deoarece același document poate aparține mai multor categorii, problema este formulată ca o sarcină de clasificare multi-label.

Proiectul urmărește compararea mai multor abordări bazate pe modele Transformer:

- **Modelul 1 – Fine-tuning encoder:** XLM-RoBERTa-base și Legal-XLM-Longformer-base.
- **Modelul 2 – Few-shot prompting:** clasificare utilizând un LLM de maximum 7–8 miliarde de parametri.
- **Modelul 3 – Supervised Fine-Tuning (SFT):** adaptarea unui LLM prin LoRA.

În această etapă au fost finalizate analiza exploratorie a datelor și experimentele cu cele două encodere.

## 2. Dataset

A fost utilizat datasetul MultiEURLEX, care conține documente legislative europene și etichete tematice EUROVOC.

Experimentul utilizează documentele în limba română și cele **21 de categorii generale EUROVOC**.

| Set de date | Număr de documente |
|---|---:|
| Train | 15.920 |
| Validation | 5.000 |
| Test | 5.000 |
| **Total** | **25.920** |

Setul de antrenare provine din partiția originală de 15.921 de documente, din care a fost eliminat un duplicat textual.

Analiza exploratorie a evidențiat documente de lungimi foarte diferite și un dezechilibru important între categoriile tematice.

**Datasetul nu este inclus în repository.** Acesta trebuie obținut separat din sursa originală.

Sursă: [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex)

## 3. Modele encoder

### 3.1. XLM-RoBERTa-base

Model de tip encoder multilingv, utilizat ca baseline pentru clasificarea multi-label.

Configurație experimentală:

- Model preantrenat: `FacebookAI/xlm-roberta-base`
- Context maxim: 512 tokeni
- Număr de categorii: 21
- Fine-tuning în două faze: 3 + 3 epoci
- Learning rate inițial: `2e-5`
- Learning rate pentru continuarea antrenării: `5e-6`
- Selecția checkpoint-ului: Micro-F1 pe validation

### 3.2. Legal-XLM-Longformer-base

Model de tip encoder specializat pentru texte juridice multilingve, capabil să proceseze contexte extinse.

Configurație experimentală:

- Model preantrenat: `joelniklaus/legal-xlm-longformer-base`
- Context maxim utilizat: 2.048 tokeni
- Capacitate maximă a modelului: 4.096 tokeni
- Număr de categorii: 21
- Maximum 6 epoci de fine-tuning
- Learning rate: `2e-5`
- Batch size: 1
- Gradient accumulation: 8
- FP16 și gradient checkpointing
- Selecția checkpoint-ului: Micro-F1 pe validation

Ambele modele au fost evaluate folosind aceleași partiții de date și aceeași ordine a categoriilor.

## 4. Optimizarea pragurilor de clasificare

Pentru clasificarea multi-label, fiecare model produce câte o probabilitate pentru fiecare dintre cele 21 de categorii.

Au fost analizate trei strategii:

1. Prag fix de `0.5`.
2. Prag global selectat pe validation pentru maximizarea Micro-F1.
3. Praguri individuale pentru fiecare categorie, optimizate pe validation.

Pentru comparația principală au fost selectate următoarele praguri globale:

| Model | Prag |
|---|---:|
| XLM-RoBERTa | 0,40 |
| Legal-XLM-Longformer | 0,45 |

Pragurile individuale au fost păstrate ca experiment suplimentar, având un risc mai mare de supraajustare pe setul de validare.

## 5. Rezultate experimentale

### 5.1. Rezultate pe validation

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| XLM-RoBERTa | 0,820872 | 0,681307 |
| **Legal-XLM-Longformer** | **0,837415** | **0,720935** |

### 5.2. Rezultate finale pe test

Evaluarea a fost realizată pe aceleași 5.000 de documente de test, utilizând configurațiile selectate anterior pe validation.

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| XLM-RoBERTa | 0,786074 | 0,646729 |
| **Legal-XLM-Longformer** | **0,806692** | **0,687046** |

Legal-XLM-Longformer depășește XLM-RoBERTa cu aproximativ:

- **2,06 puncte procentuale Micro-F1**
- **4,03 puncte procentuale Macro-F1**

În analiza individuală a celor 21 de categorii EUROVOC, Legal-XLM-Longformer a obținut F1 superior pentru **20 dintre cele 21 de categorii**.

Cele mai mari îmbunătățiri au fost observate pentru categoriile *organizații internaționale*, *știință* și *politică*. Totuși, performanța pe unele categorii rare rămâne limitată.

## 6. Concluzii preliminare

Legal-XLM-Longformer a fost selectat drept encoderul principal al proiectului, deoarece a obținut cea mai bună performanță Micro-F1 pe validation, iar avantajul față de XLM-RoBERTa s-a menținut pe test.

Comparația nu izolează exclusiv efectul lungimii contextului, întrucât modelele diferă și în privința arhitecturii și a datelor de preantrenare.

Rezultatele evidențiază și importanța optimizării pragurilor de clasificare și a utilizării Macro-F1 pentru evaluarea categoriilor dezechilibrate.

## 7. Structura proiectului

```text
Project_multi_eurlex/
├── Data/                   # Dataset local, exclus din Git
├── Models/                 # Modele antrenate, excluse din Git
├── Notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_finetuning_xlm_roberta.ipynb
│   └── 03_finetuning_legal_xlm_longformer.ipynb
├── Results/                # Metrici, grafice si comparatii
└── README.md
```

Fișierele `.npz` cu predicții și greutățile modelelor sunt păstrate local, fără a fi publicate în repository.

## 8. Mediu experimental

Experimentele encoder au fost realizate într-un mediu Linux/WSL, utilizând:

- Python 3.11
- PyTorch 2.15.0+cu132
- Transformers 5.23.0
- NVIDIA GeForce RTX 5070 Laptop GPU (7,96 GB VRAM)

Biblioteci principale: `transformers`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.

## 9. Etapele următoare

Proiectul va continua cu:

- Clasificarea prin few-shot prompting utilizând un LLM.
- Adaptarea unui LLM prin Supervised Fine-Tuning cu LoRA.
- Compararea abordărilor și realizarea unui demo de clasificare.

## 10. Referințe

- [MultiEURLEX – Hugging Face](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- [XLM-RoBERTa – Hugging Face](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [Legal-XLM-Longformer – Hugging Face](https://huggingface.co/joelniklaus/legal-xlm-longformer-base)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)