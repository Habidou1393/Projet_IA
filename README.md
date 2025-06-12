mon_projet/
│
├── app/                      # Application Flask
│   ├── __init__.py
│   ├── app.py                # Point d'entrée Flask
│   ├── config.py             # Configs (ex: WIKI_TRIGGER, GOOGLE_TRIGGER)
│   └── memory.py             # Gestion mémoire/session utilisateur
│
├── utils/                    # Fonctions utilitaires
│   ├── __init__.py
│   ├── google_search.py      # Recherche Google
│   ├── wikipedia_search.py   # Recherche Wikipédia
│   └── monchatbot.py         # Chargement et génération avec MiniGPT
│
├── templates/                # Templates HTML (Flask/Jinja2)
│   └── index.html            # Interface utilisateur
│
├── static/                   # Fichiers statiques frontend
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js           # JS côté client (messages, affichage, etc.)
│
├── data/                     # Données pour entraînement / corpus
│   ├── mon_corpus.txt        # Corpus brut initial
│   ├── corpus_formate.txt    # Corpus formaté dialogues (Utilisateur/Assistant)
│   └── frwiki-latest-pages-articles.xml.bz2  # Dump brut Wikipédia
│   └── frwiki_extrait/       # Texte extrait JSON via wikiextractor
│
├── entrainement/             # Scripts liés à l’entraînement du modèle
│   ├── convertir_corpus.py   # Script pour formater corpus en dialogues
│   ├── tokenizer_train.py    # Script pour entraîner le tokenizer BPE
│   ├── finetune.py           # Script d’entraînement MiniGPT (finetuning)
│   ├── model.py              # Définition MiniGPT (architecture)
│   └── utils_training.py     # (Nouveau) Fonctions communes entraînement
│
├── .env                      # Variables d’environnement (optionnel)
└── README.md                 # Documentation projet