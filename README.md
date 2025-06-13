# mon_projet

## Structure du projet

```
mon_projet/
│
├── app/                      # Application Flask
│   ├── __init__.py
│   ├── app.py                # Point d'entrée Flask
│   ├── config.py             # Configurations (ex: WIKI_TRIGGER, GOOGLE_TRIGGER)
│   ├── memory.py             # Gestion mémoire/session utilisateur
│   └── __pycache__/
│
├── data/                     # Données pour entraînement / corpus
│   ├── corpus_formate.txt    # Corpus formaté dialogues (Utilisateur/Assistant)
│
├── entrainement/             # Scripts liés à l’entraînement du modèle
│   ├── finetune.py           # Script d’entraînement MiniGPT (finetuning)
│   ├── model.py              # Définition MiniGPT (architecture)
│   ├── tokenizer_train.py    # Script pour entraîner le tokenizer BPE
│   ├── utils_training.py     # Fonctions communes entraînement
│   └── __pycache__/
│   └── model-output/         # Sorties de modèles entraînés
│
├── tokenizer/                # Tokenizer BPE
│   ├── merges.txt
│   └── vocab.json
│
├── utils/                    # Fonctions utilitaires
│   ├── __init__.py
│   ├── google_search.py      # Recherche Google
│   ├── wikipedia_search.py   # Recherche Wikipédia
│   ├── mon_extractor.py
│   └── monchatbot.py         # Chargement et génération avec MiniGPT
│   └── __pycache__/
│
├── templates/                # Templates HTML (Flask/Jinja2)
│   └── index.html            # Interface utilisateur
│
├── static/                   # Fichiers statiques frontend
│   ├── css/
│   └── js/
│
├── convertisseur.py          # Script de conversion (fonction inconnue)
├── Nettoyage.py              # Script de nettoyage (fonction inconnue)
├── .env                      # Variables d’environnement (optionnel)
├── .gitignore
└── README.md                 # Documentation projet
```