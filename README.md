mon_projet/
│
├── app/
│   ├── __init__.py
│   ├── app.py               # Flask app (point d'entrée)
│   ├── config.py            # Contient la clé MISTRAL_API_KEY, et d'autres configs
│   └── memory.py            # Gère le cache ou mémoire de session
│
├── utils/
│   ├── __init__.py
│   ├── google_search.py
│   ├── wikipedia_search.py
│   ├── Calcul_Maths.py      # Ton module de résolution mathématique
│   ├── Mistral_API.py       # ✅ Contient la requête POST à l'API Mistral
│   └── monchatbot.py        # ✅ Cœur logique : routing, détection, prompts, etc.
│
├── templates/
│   └── index.html           # Interface utilisateur HTML (Flask Jinja2)
│
├── static/
│   ├── css/
│   │   └── style.css        # Ton design (mode sombre, bulles, etc.)
│   └── js/
│       └── main.js          # JS client : envoi des messages, affichage, LaTeX, etc.
│
├── data/
│   └── mon_corpus.txt       # ✅ Ton jeu de données d'entraînement personnalisé
│
├── entrainement/ 
│   ├── finetune.py          # ✅ Script pour entraîner GPT2 ou TinyLlama avec HuggingFace
│   └── model-output/        # Répertoire de sortie du modèle fine-tuné
│
├── .env                     # Variables sensibles : MISTRAL_API_KEY, etc.
└── README.md                # Documentation du projet
