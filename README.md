# Financial Fraud Detection con architetture MLP 

## Progetto e Obiettivi
Il presente repository contiene lo sviluppo e l'analisi di un sistema di Fraud Detection basato su reti neurali Deep Learning (Multi-Layer Perceptron). L'obiettivo principale è l'identificazione di transazioni fraudolente in dataset tabellari sbilanciati, con un focus particolare sulla massimizzazione della Recall e sull'ottimizzazione delle procedure di Feature Engineering per dati finanziari.

## Architettura del Modello
Il modello è implementato in PyTorch e presenta la seguente struttura:
* **Input Layer**: Dimensionato in base alle feature preprocessate (33, 63 o 65 input).
* **Hidden Layers**: 7 livelli lineari con dimensionamento variabile (da 64 a 1024 neuroni).
* **Regolarizzazione**: Integrazione di livelli di Dropout (settabili tra 0.01 e 0.75) per il controllo dell'overfitting.
* **Funzioni di Attivazione**: ReLU per i livelli intermedi e CrossEntropyLoss per la classificazione binaria in uscita.
* **Ottimizzatore**: Adam Optimizer con learning rate scheduling manuale.

## Metodologia di Preprocessing e Feature Engineering
La pipeline di preparazione dei dati include tecniche avanzate per gestire la natura specifica dei record bancari:
* **Log-Transformation**: Applicazione di log2(x+1) sui volumi transazionali per normalizzare distribuzioni ad alta skewness.
* **Target Encoding con Smoothing**: Gestione di variabili categoriche ad alta cardinalità (localizzazione geografica e profilo professionale) tramite mappatura della probabilità del target corretta con fattore di smoothing per evitare il data leakage e l'overfitting su categorie rare.
* **Data Binning**: Discretizzazione degli importi in intervalli logaritmici seguiti da One-Hot Encoding per migliorare la capacità di separazione lineare del modello.

## Analisi Sperimentale
Sono stati condotti test comparativi su due configurazioni principali:
1. **Baseline Model**: Architettura semplificata senza l'ausilio di dati geografici e professionali.
2. **Enhanced Model (Target Encoded)**: Integrazione delle feature ad alta cardinalità processate tramite Smooth Target Encoding, risultante in un incremento della precisione metrica sui segmenti di clientela specifici.

