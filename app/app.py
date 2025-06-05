import os
import sys
import logging
from flask import Flask, request, jsonify, render_template

# === Configuration des chemins ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

UTILS_PATH = os.path.join(ROOT_DIR, 'utils')
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

# === Import du cœur du chatbot ===
try:
    from utils.monchatbot import obtenir_la_response
except ImportError as e:
    raise ImportError(f"Erreur d'importation depuis utils/monchatbot.py : {e}")

# === Flask app ===
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, 'templates'),
    static_folder=os.path.join(ROOT_DIR, 'static')
)

logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify(response="Veuillez écrire quelque chose."), 400

        response_text = obtenir_la_response(message)
        return jsonify(response=response_text)

    except Exception as e:
        app.logger.error(f"Erreur lors du traitement : {e}", exc_info=True)
        return jsonify(response="Erreur interne."), 500

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
