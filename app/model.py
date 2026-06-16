import os
import io
import json
import base64
import anthropic
from PIL import Image

CLAUDE_MODEL = "claude-haiku-4-5"

VALID_LABELS = {"plastique", "verre", "metal", "cardboard", "paper", "organique", "electronique", "textile", "dangereux", "bois", "trash"}

PROMPT = """Analysez cette image et identifiez TOUTES les composantes distinctes visibles (corps, bouchon, étiquette, couvercle, emballage, etc.).

Répondez avec UNIQUEMENT un objet JSON valide, sans explication ni markdown :
{"components": [{"partie": "Corps", "label": "verre", "confidence": 0.95, "conseil": "Rincez avant de déposer dans la colonne verte."}, {"partie": "Bouchon", "label": "metal", "confidence": 0.88, "conseil": "Retirez le bouchon et mettez-le dans le bac jaune."}]}

Règles :
- Listez chaque composante séparément si elle est d'une matière différente
- Si l'objet est d'une seule matière, retournez une seule composante
- Le nom de la "partie" doit être en français (Corps, Bouchon, Étiquette, Couvercle, Emballage, Fond, etc.)
- Le "label" doit être exactement l'un de :
  "plastique"    -> bouteilles, sacs, gobelets, pailles, mousse plastique
  "verre"        -> bouteilles en verre, bocaux, verre cassé
  "metal"        -> canettes, papier alu, capsules, couvercles métal
  "cardboard"    -> cartons, boîtes en carton, briques alimentaires
  "paper"        -> journaux, mouchoirs, sacs en papier, magazines
  "organique"    -> déchets alimentaires, matière organique
  "electronique" -> piles, câbles, appareils électroniques, chargeurs
  "textile"      -> vêtements, chaussures, tissu, rideaux, literie
  "dangereux"    -> peinture, produits chimiques, médicaments, seringues
  "bois"         -> meubles en bois, planches, palettes, branches
  "trash"        -> déchet non recyclable ou non identifiable
- "conseil" : conseil pratique court en français pour aider l'utilisateur à bien trier cette composante
- confidence doit être un float entre 0.0 et 1.0
- Ordonnez les composantes par confidence décroissante"""


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def run_inference(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
    )

    data = json.loads(_clean_json(response.content[0].text))
    raw_components = data.get("components", [])

    components = []
    for c in raw_components:
        label = c.get("label", "trash")
        if label not in VALID_LABELS:
            label = "trash"
        confidence = max(0.0, min(1.0, float(c.get("confidence", 0.5))))
        components.append({
            "partie": c.get("partie", "Objet"),
            "label": label,
            "confidence": round(confidence, 3),
            "conseil": c.get("conseil", ""),
        })

    if not components:
        components = [{"partie": "Objet", "label": "trash", "confidence": 0.5}]

    top = components[0]
    return {
        "components": components,
        "detections": [{"label": c["label"], "confidence": c["confidence"]} for c in components],
        "count": len(components),
        "top_label": top["label"],
        "top_confidence": top["confidence"],
        "source": "claude",
    }
