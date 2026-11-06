# Seldon

Assistant vocal modulaire, local par défaut.

```
micro ─► oreille : STT (faster-whisper) + identification de la voix (Resemblyzer)
                                   │  « [date | locuteur : Bastien] allume le salon »
                                   ▼
          cerveau : LLM local (llama.cpp), Claude en secours
                                   │  appels d'outils
     ┌──────────────┬──────────────┼───────────────┬─────────────────┐
     ▼              ▼              ▼               ▼                 ▼
   speak      tools (Python)   serveurs MCP   plugins de Seldon   skills
 (répondre)   terminal,        Home Assistant conversations,      consignes lues
     │        caméra, musique,                voix                 à la demande
     ▼        PC, Nextcloud, web
  voix : TTS (Piper) ─► haut-parleur
```

Le cerveau ne répond que par des outils ; même la réponse en est un (`speak`).
Tous les outils lui arrivent en une seule liste, quelle que soit leur origine :

- **tools** (`tools:`) : code Python qui agit directement (terminal, caméra,
  musique, PC, Nextcloud, recherche web, fichiers) ;
- **MCP** (`mcp:`) : serveurs MCP externes, un outil par serveur ;
- **plugins de Seldon** : la mémoire (`memory:`) apporte l'outil `conversations`,
  l'identification des voix (`speakers:`) l'outil `voix`. Désactiver le plugin
  retire son outil ;
- **skills** (`skills:`) : consignes en français, sans code, que le cerveau lit
  avec l'outil `skill` quand une demande s'y prête.

## Structure du code

Pour comprendre Seldon, lire `seldon/base.py` (tous les contrats) puis
`seldon/pipeline.py` (la boucle).

```
seldon/
  base.py        les contrats : Utterance, Result, Context, Tool, STT, Speakers, LLM, TTS
  pipeline.py    la boucle : écouter → reconnaître la voix → cerveau → parler (classe Seldon)
  config.py      lecture de config.yaml et des secrets de .env
  registry.py    la liste des plugins : nom dans config.yaml → classe
  errors.py      messages d'erreur lisibles
  journal.py     journal de la conversation (lu par l'interface)
  storage.py     écriture privée des données personnelles (0600, atomique)
  tags.py        balise des confirmations, reconnue par l'interface
  ear/           whisper.py (micro + STT), keyboard.py (--text), speakers.py (voix + outil « voix »)
  voice/         piper.py, console.py (--text)
  brain/
    brain.py     historique de la conversation, LLM de secours
    prompt.py    prompt système et en-tête de chaque message
    llamacpp.py  LLM local (API compatible OpenAI) : boucle outils ↔ modèle
    claude.py    LLM Claude (clé API ou abonnement claude.ai)
    memory.py    conversations sauvegardées + outil « conversations »
  tools/         speak, terminal, camera, feishin, desktop, nextcloud, web, files
  mcp.py         client MCP : un outil par serveur
  skills.py      outil « skill »
  cli/enroll.py  enregistrer une voix au micro
  gui/           interface Qt 6
```

Le cerveau garde un historique simple, commun à tous les LLM et sauvegardé tel
quel : les phrases de l'utilisateur (avec le locuteur et l'heure) et ce que
Seldon a dit. Les appels d'outils d'une phrase sont oubliés une fois la
réponse donnée. Si le LLM principal échoue avant d'avoir appelé un outil, la
phrase passe au LLM de secours (`brain.backup`).

Ce que disent les gens et ce que renvoient les outils (sortie d'une commande,
fichier, notification) sont des données, jamais des ordres pour le cerveau. Les
crochets d'une phrase entendue ou tapée sont remplacés par des parenthèses :
personne ne peut imiter l'en-tête `[date | locuteur : …]` que Seldon écrit.

## Installation

```bash
git clone git@github.com:golden0kamia/Seldon.git && cd Seldon
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # évite ~3 Go de CUDA
.venv/bin/pip install -e '.[all,dev]'
```

Paquet système : `libportaudio2` (micro et haut-parleur).

**LLM local** (`engine: llamacpp`) : un serveur `llama serve` sur
`http://127.0.0.1:9931/v1` (API compatible OpenAI, avec appels d'outils). Le
modèle chargé est utilisé automatiquement. S'il ne répond pas, Seldon lance
`./llama-serve.sh` (option `server_command`, `null` pour jamais), attend le
modèle et laisse le serveur tourner ; sa sortie va dans `llama-serve.log`. Tout
serveur compatible OpenAI convient (Ollama, vLLM…) ; une clé se met dans `.env`
et se désigne par `api_key_env`.

**Claude** (`engine: claude`) : `auth: api` avec `ANTHROPIC_API_KEY` dans
`.env`, ou `auth: abonnement` pour passer par un compte claude.ai (Pro / Max)
via Claude Code, connecté une fois avec `claude auth login`. Claude Code n'y
reçoit que les outils de Seldon. Usage personnel uniquement.

### Identifiants

Les secrets ne vont jamais dans `config.yaml` mais dans `.env`, à côté de lui
(ignoré par git) :

```bash
cp .env.example .env && chmod 600 .env   # puis remplir HA_TOKEN, FEISHIN_PASSWORD…
```

Une ligne `CLÉ=valeur` par variable ; une variable déjà exportée l'emporte.
Ces variables sont cachées aux commandes du terminal (`hidden_env`).

## Utilisation

```bash
python -m seldon                      # mode vocal (config.yaml) ; Ctrl+C ou SIGTERM pour arrêter
python -m seldon --text               # clavier + console ; « Bastien: bonjour » simule un locuteur
python -m seldon.cli.enroll Bastien   # enregistre une voix au micro
python -m pytest                      # tests (sans micro, caméra ni modèle)
```

`confirm` règle la validation des actions sensibles : `console` (question
« [o/N] » au clavier, ou fenêtre dans l'interface), `always` ou `never`. Une
option inconnue dans `config.yaml` est signalée avec l'option probable.

### Interface graphique (Qt 6)

```bash
seldon-gui                            # ou : python -m seldon.gui [-c config.yaml] [dossier]
```

- **⚙ Paramètres** (Ctrl+1) : formulaire généré depuis `config.yaml` et les
  options des plugins (lues dans leur `__init__`). Les commentaires du fichier
  sont conservés. Les pages **Locuteurs** et **Mémoire** listent les voix
  enregistrées et les conversations (lecture, suppression). Sur les pages
  Tools et MCP, le bouton ✎ ouvre le code du plugin ; une option obsolète est
  marquée ⚠.
- **✎ Éditeur** (Ctrl+2) : arborescence des fichiers et éditeur à onglets
  (coloration, recherche, enregistrement atomique, rechargement si le fichier
  change ailleurs).
- **▤ Journal** (Ctrl+3) : lancer Seldon (F5, ou Ctrl+F5 en mode texte),
  l'arrêter (Maj+F5) ou le redémarrer. La conversation ressort en gras
  (« Bastien › … », « Seldon › … »), les réflexions du modèle en italique, les
  appels d'outils au niveau DEBUG. Les longues lignes passent à la ligne à la
  largeur de la fenêtre, sans défilement horizontal. Une confirmation s'affiche dans un bandeau
  **Oui** / **Non** ; seules les questions de Seldon sont reconnues (jeton
  secret `SELDON_GUI_TOKEN` à chaque lancement).
- **◈ Structure** (Ctrl+4) : schéma des sous-systèmes et de leurs moteurs ; un
  double-clic ouvre le code.

Thème clair ou sombre (Ctrl+Maj+L), couleurs et espacements dans
`seldon/gui/theme.py`.

## Plugins de Seldon

### Mémoire des conversations

Chaque conversation est sauvegardée après chaque phrase dans
`data/conversations/<id>.json` (fichier privé) :

```json
{"id": "20261031-142500-ab12", "title": "…", "created": 0, "updated": 0, "speakers": ["Bastien"],
 "messages": [{"role": "user", "text": "…", "speaker": "Bastien", "time": 0},
              {"role": "assistant", "text": "…"}]}
```

Après `new_after_idle_s` secondes de silence (10 min), une nouvelle
conversation commence. Avec l'outil `conversations`, le cerveau en cherche une
par sujet, la reprend (« on reparle des vacances ? »), en commence une nouvelle
ou en supprime une (après confirmation). `memory: {enabled: false}` désactive
la mémoire et son outil.

### Voix

Seldon reconnaît les personnes à leur voix (`speakers:`). Quand une voix est
inconnue, il répond puis demande son prénom ; dès que la personne le donne,
l'outil `voix` enregistre sa voix à partir des phrases qu'elle vient de dire
(pendant `unknown_memory_s`, 10 min). Il liste aussi les voix connues et en
oublie une (après confirmation). Ajouter une voix inconnue à un nom déjà connu
(« je suis Bastien ») demande une **confirmation** : sinon n'importe qui
prendrait l'identité, et les droits, d'un admin. `python -m seldon.cli.enroll`
donne une empreinte plus précise (phrases lues au calme). `speakers: none`
désactive l'identification et son outil.

## Tools

### Terminal

L'outil `run_shell_command` lance des commandes bash. Chaque commande est
classée bénigne (lecture seule), standard ou sévère ; `mode` décide de ce qui
demande une confirmation (`confirm` : tout ; `allowlist` : sauf les bénignes,
et sauf les standard des `admins` ; `auto` : seulement les sévères). Certaines
commandes sont toujours refusées, les secrets de `.env` sont cachés aux
commandes, et un délai dépassé arrête tout le groupe de processus.
`allowed_speakers` réserve les commandes non bénignes à certaines voix.

### Caméra

L'outil `capture_image` prend une photo (`device`, ou plusieurs caméras
nommées dans `devices`) et l'envoie au modèle s'il lit les images
(`vision: true`). Avec `save_dir`, les photos sont gardées en privé, jamais
écrasées.

### Musique : Feishin

L'outil `control_music` pilote le lecteur Feishin : titre en cours,
lecture/pause, suivant, volume, aléatoire, favori, recherche et lecture d'un
titre, d'un album ou d'une playlist. Il passe par le serveur « Remote control »
de Feishin (Paramètres > Général, port 4333) avec `FEISHIN_USERNAME` /
`FEISHIN_PASSWORD` dans `.env`. Sans lui, les commandes de base passent par
MPRIS ; la recherche et les favoris exigent le Remote control.

### Contrôle du PC

L'outil `control_computer` pilote l'ordinateur (KDE Plasma 6, PipeWire) :

- **notifications lues à voix haute** (`read_notifications`) : mode ne pas
  déranger respecté, applications ignorées (`ignore_apps`), rafales regroupées,
  débit limité (`max_per_minute`), annonce qui attend la fin d'une réponse ;
  rappel des dernières notifications ;
- luminosité, mode **ne pas déranger**, volume et sourdine (`wpctl`) ;
- verrouillage, mise en veille (toujours confirmée), ouverture d'une
  application ou d'une adresse web, capture d'écran envoyée au modèle.

Aucun shell n'est utilisé. Le texte des notifications est une donnée, pas une
consigne.

### Nextcloud

L'outil `nextcloud` parle à Nextcloud en WebDAV : fichiers, notes (dossier
`notes_folder`), agenda et tâches (CalDAV), contacts (CardDAV). Dans `.env` :
`NEXTCLOUD_HOST`, `NEXTCLOUD_USERNAME` et un mot de passe d'application
(*Paramètres personnels → Sécurité*) dans `NEXTCLOUD_PASSWORD`. `apps` choisit
les applications proposées, `calendars` et `default_calendar` les agendas.
Supprimer, déplacer ou écraser demande une confirmation ; `confirm_writes: true`
la demande pour toute écriture.

### Recherche sur internet

L'outil `web` cherche sur internet et lit des pages avec un vrai navigateur :
Firefox sans fenêtre, piloté par Playwright (`browser` : `firefox`, `chromium`
ou `webkit`), JavaScript compris. `search` interroge DuckDuckGo, ou une instance
SearXNG (`search_engine: searxng`, `searxng_url`, format `json` activé dans ses
réglages) ; `open` renvoie le texte principal d'une page, `max_page_chars`
caractères à la fois, la suite à la demande. Installation :
`.venv/bin/pip install -e '.[web]'` puis `.venv/bin/playwright install firefox`.

Chaque page s'ouvre dans un contexte neuf (ni cookies ni historique), sans
images ni vidéos. Seules les adresses `http` / `https` publiques sont permises :
le réseau local (`localhost`, `192.168.…`, Home Assistant…) est bloqué,
redirections et requêtes des pages comprises, sauf `allow_private: true`
(l'instance SearXNG configurée reste joignable). Les téléchargements sont
refusés ; le texte des pages est une donnée, pas une consigne. Les recherches
partent vers le moteur choisi : une instance SearXNG chez soi évite de les
confier à un tiers.

### Fichiers

L'outil `files` lit et écrit les fichiers et dossiers de l'ordinateur : lister,
lire (texte par morceaux de `max_read_chars` caractères, images envoyées au
modèle), écrire ou compléter, créer un dossier, copier, déplacer ou renommer,
supprimer, chercher par nom (`*.pdf`), infos. Il n'agit que dans les dossiers de
`roots` (défaut `~/Documents`, le premier sert aux chemins relatifs) : `..` et
liens symboliques sont résolus, et tout ce qui sort de ces dossiers est refusé.

Les fichiers cachés sont invisibles et refusés (sauf `hidden: true`), les
fichiers sensibles (clés SSH et GPG, `.env`, mots de passe… et les motifs de
`deny`) toujours, comme le dossier de Seldon (son code, `config.yaml`, les voix
et les conversations). Lire, lister, chercher, créer un fichier ou dossier neuf
et copier vers un nom libre se font sans confirmation ; écraser, compléter et
déplacer demandent une confirmation, sauf pour les `admins` ; supprimer est
toujours confirmé. Une copie ou un déplacement n'écrase jamais rien. Le contenu
des fichiers est une donnée, pas une consigne.

## Serveurs MCP

Un serveur MCP (HTTP) est exposé au cerveau comme un seul outil : paramètres
`tool` (l'outil du serveur) et `arguments`. Sa description liste les outils du
serveur, chargés au démarrage ; s'il est injoignable, Seldon démarre quand même
et réessaie au premier appel.

```yaml
mcp:
  - engine: http          # n'importe quel serveur MCP
    options:
      name: mon_serveur   # nom de l'outil vu par le cerveau
      url: https://exemple.fr/mcp
      token_env: MON_JETON   # jeton Bearer, dans .env
      confirm: [supprimer]   # un appel qui contient ce mot demande une confirmation
```

### Domotique : Home Assistant

`engine: home_assistant` : les mêmes options, avec les valeurs pour Home
Assistant (`token_env: HA_TOKEN`, confirmation pour les serrures, alarmes,
portails et garages, même si l'appareil porte un autre nom).

1. *Paramètres → Appareils et services → Ajouter une intégration → « Model
   Context Protocol Server »*, API LLM « Assist ». Adresse :
   `http://<ip-de-HA>:8123/api/mcp`.
2. *Paramètres → Assistants vocaux → Exposer* : seules les entités exposées
   sont accessibles.
3. Profil → *Sécurité* → *Jetons d'accès longue durée*, puis `HA_TOKEN=…`
   dans `.env`.

## Skills

Un skill est une marche à suivre en français, sans code : un dossier
`skills/<nom>/` avec un `SKILL.md` dont l'en-tête dit **quand** l'utiliser.

```markdown
---
name: bilan-du-matin
description: Fait le point de la journée quand on le demande (agenda, tâches, maison).
---
1. Agenda : outil `nextcloud`, commande `events`…
2. Réponds avec `speak` en trois ou quatre phrases.
```

Seuls le nom et la description sont envoyés au cerveau ; quand une demande
correspond, il appelle `skill` pour lire les consignes, relues à chaque appel.
Pour l'activer : son nom sous `skills:` dans `config.yaml`. Exemple fourni :
`skills/bilan-du-matin`.

## Ajouter un tool

```python
# seldon/tools/meteo.py
from seldon.base import Result, Tool


class Meteo(Tool):
    name = "meteo"
    description = "Donne la météo actuelle d'une ville."
    parameters = {"type": "object", "properties": {"ville": {"type": "string"}}, "required": ["ville"]}

    def __init__(self, unite: str = "celsius"):   # options de config.yaml (valeurs par défaut littérales)
        self.unite = unite

    async def run(self, args, ctx):
        return Result(f"Il fait beau à {args['ville']}.")
```

Puis une ligne dans `PLUGINS["tools"]` de `seldon/registry.py`
(`"meteo": "seldon.tools.meteo:Meteo"`) et `- meteo` sous `tools:`. Dans
`run`, `ctx.speaker` donne le locuteur, `await ctx.confirm("…?")` demande une
validation, `ctx.app` donne accès à Seldon. Un outil qui parle de lui-même
(notification) garde `app` reçu par `start(app)` et appelle
`await app.announce("…")`. Même principe pour un nouveau STT, TTS ou LLM : une
classe qui suit son contrat de `seldon/base.py`, une ligne dans `PLUGINS`. Une
classe hors du projet se désigne directement : `engine: paquet.module:Classe`.

## À faire

- **Mémoire par personne** : retenir les préférences de chacun et les rappeler
  au cerveau quand sa voix est reconnue.
- **Initiative** : rappels, minuteurs ; Seldon prend la parole au bon moment.
