# Seldon

Assistant vocal polyvalent, modulaire et local par défaut.

```
micro ─► STT (faster-whisper + VAD) ─► identification du locuteur (Resemblyzer)
                                              │
                                              ▼
          cerveau : LLM local (serveur llama.cpp `llama serve`, ou Ollama) ou Claude
                                              │ appels d'outils
              ┌───────────────┬───────────────┼────────────────┐
              ▼               ▼               ▼                ▼
          speak (TTS)   run_shell_command  capture_image   … futures actions
          Piper
```

Le cerveau reçoit « [date | locuteur : X] texte » et décide quelles actions
lancer. La réponse à l'utilisateur est elle-même une action (`speak`).

## Installation

```bash
git clone git@github.com:golden0kamia/Seldon.git && cd Seldon
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # évite ~3 Go de CUDA
.venv/bin/pip install -e '.[all,dev]'
```

Le cerveau par défaut (`llamacpp`) se connecte à un serveur `llama serve`
déjà lancé (`http://127.0.0.1:9931/v1`, API compatible OpenAI, avec appels
d'outils). Le modèle chargé sur le serveur est utilisé automatiquement.
Pour une voix réactive, privilégier un modèle qui tient entièrement en VRAM
et laisser `thinking: false`.

Paquets système : `libportaudio2` (micro/haut-parleur), `espeak-ng` (seulement pour le TTS pyttsx3).

## Utilisation

```bash
python -m seldon                    # mode vocal complet (config.yaml)
python -m seldon --text             # clavier + console ; "Bastien: bonjour" simule un locuteur
python -m seldon.tools.enroll Bastien   # enregistre une empreinte vocale
pytest                              # tests (sans micro, caméra ni modèle)
```

Commandes locales (traitées sans passer par le modèle, au clavier ou à la voix) :
`exit` / `quitter`, `nouvelle` / `oublie tout`, `conversations [mots-clés]`,
`reprendre N|mots-clés`, `supprimer N`, `actions`, `aide`. Le « / » initial
est facultatif (obligatoire pour passer des arguments).

### Interface graphique (Qt 6)

```bash
.venv/bin/pip install -e '.[gui]'
seldon-gui                          # ou : python -m seldon.gui [-c config.yaml] [dossier]
```

Deux pages, choisies en haut de la fenêtre :

- **⚙ Paramètres** (Ctrl+1) : formulaire généré depuis `config.yaml` et les
  signatures des plugins (moteurs disponibles, options connues avec leur valeur
  par défaut). Les commentaires du fichier sont conservés à l'enregistrement.
- **✎ Éditeur** (Ctrl+2) : arborescence des fichiers (dossier Seldon, `data/`,
  `cwd` des actions, dossiers récents ; création, renommage, corbeille) et
  éditeur à onglets, jamais vide (document « Sans titre » par défaut).
  Coloration Python, YAML, JSON, Markdown, TOML, shell. Raccourcis : Ctrl+N
  nouveau document, Ctrl+S / Ctrl+Maj+S enregistrer (sous), Ctrl+F / Ctrl+H
  rechercher / remplacer, Ctrl+/ commenter. Un fichier modifié ailleurs (par
  Seldon, Claude Code…) est rechargé automatiquement.

Thème néomorphique « cassette futurism », accent vert, en variante claire ou
sombre (Affichage › Thème, bouton ☾/☀ ou Ctrl+Maj+L ; « Automatique » suit le
thème du système). Polices Share Tech et Share Tech Mono (Google Fonts, à
installer sur le système ; IBM Plex à défaut). Couleurs et polices dans
`seldon/gui/theme.py`.

### Mémoire des conversations

Chaque conversation est sauvegardée dans `data/conversations/` (transcription +
historique complet du cerveau) après chaque échange. Après 10 min de silence,
une nouvelle conversation commence ; l'ancienne reste disponible. Pour reprendre
un sujet, il suffit de le demander (« on reprend notre discussion sur les
tomates ? ») : le cerveau cherche et recharge la conversation via l'outil
`conversations`. Au clavier : `/conversations tomates` puis `/reprendre 1`. Pour en ajouter une : décorateur `@command` dans
`seldon/core/commands.py`.

Commandes terminal (`mode: allowlist`) : celles en lecture seule (`ls`, `grep`,
`df`, `git status`…) s'exécutent directement ; toute commande qui peut modifier
quelque chose (programme hors liste, redirection vers un fichier, `find -delete`,
`systemctl restart`…) demande une confirmation au clavier. Les commandes
dangereuses (`rm -rf /`, `mkfs`…) sont toujours refusées. Voir `config.yaml`.

## Architecture

| Dossier | Rôle | Contrat |
|---|---|---|
| `seldon/stt/` | Parole → texte | `STTEngine.listen()` produit des `Utterance` |
| `seldon/speakers/` | Qui parle ? | `SpeakerIdentifier.identify(utterance)` |
| `seldon/brain/` | Décision | `Brain.handle(utterance, actions, execute)` |
| `seldon/actions/` | Outils du cerveau | `Action.run(params, ctx) -> ActionResult` |
| `seldon/tts/` | Texte → parole | `TTSEngine.speak(text)` |
| `seldon/core/` | Pipeline, registre de plugins, bus d'événements | |
| `seldon/gui/` | Interface Qt 6 (paramètres, fichiers, éditeur, thème) | lit les plugins par AST, sans les importer |

### Ajouter une action

```python
# seldon/actions/meteo.py
from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionResult

@register("actions", "meteo")
class Weather(Action):
    name = "get_weather"
    description = "Donne la météo actuelle d'une ville."
    input_schema = {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"], "additionalProperties": False}

    async def run(self, params, ctx):
        return ActionResult(text=f"Il fait beau à {params['city']}.")
```

Puis l'ajouter au dictionnaire `_BUILTINS` de `seldon/core/registry.py` et à la
liste `actions:` de `config.yaml`. Une classe hors du projet peut aussi être
référencée directement : `engine: mon_paquet.module:MaClasse`.

Le même mécanisme vaut pour les autres étages (nouveau STT, TTS, cerveau…).
Pour observer le pipeline sans le modifier (journal, interface, domotique…),
abonnez-vous aux événements : `assistant.events.on("action.end", handler)`.
Les services partagés (TTS, STT…) sont accessibles aux actions via `ctx.services`.
