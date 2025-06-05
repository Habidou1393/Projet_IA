import logging
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from typing import Optional

logger = logging.getLogger(__name__)

def recherche_google(query: str, logger: Optional[logging.Logger] = None, num_results: int = 3) -> Optional[str]:
    if logger is None:
        logger = logging.getLogger(__name__)

    query = query.strip()
    if not query:
        logger.warning("Requête Google vide.")
        return None

    try:
        logger.info(f"Recherche Google lancée pour : '{query}'")
        urls = list(search(query, num_results=num_results, lang="fr"))

        if not urls:
            logger.warning(f"Aucun résultat Google pour '{query}'")
            return None

        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                paragraphes = soup.find_all("p")
                contenu = " ".join(
                    p.get_text(strip=True)
                    for p in paragraphes[:4]
                    if len(p.get_text(strip=True)) > 60
                )

                if contenu:
                    titre = soup.title.string.strip() if soup.title and soup.title.string else url
                    logger.info(f"Contenu trouvé sur : {url}")

                    # Retourne simplement un extrait de texte avec le lien source
                    return f"{contenu[:500]}...<br><a href='{url}' target='_blank' rel='noopener noreferrer'>{titre}</a>"

            except Exception as e:
                logger.warning(f"Erreur en lisant {url} : {e}")

        logger.warning("Aucun contenu exploitable trouvé dans les résultats Google.")
        return None

    except Exception as e:
        logger.error(f"Erreur globale Google pour '{query}' : {e}", exc_info=True)
        return None
