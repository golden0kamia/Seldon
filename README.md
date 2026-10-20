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

Quand une action prend la main (commande terminal, caméra, Claude Code,
recherche dans les conversations), Seldon prévient aussitôt qu'il a compris
(« Compris, je m'en occupe. », « Je regarde. »…) pendant qu'elle s'exécute,
puis donne le résultat. Une seule fois par échange, et pas si le cerveau a
déjà parlé ; `acknowledge: false` dans `config.yaml` le désactive. Pour une
nouvelle action : attribut `acknowledgement` (ou méthode `acknowledge(params)`).

Quand une erreur survient pendant un échange, Seldon le dit à voix haute
plutôt que de rester muet : modèle de langage injoignable ou trop lent,
commande en échec, service indisponible, ou action échouée dont le cerveau
n'a pas parlé (« Désolé, l'action terminal a échoué. »). Le détail reste
dans le journal.

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

Trois pages, choisies en haut de la fenêtre :

- **⚙ Paramètres** (Ctrl+1) : formulaire généré depuis `config.yaml` et les
  signatures des plugins (moteurs disponibles, options connues avec leur valeur
  par défaut). Les commentaires du fichier sont conservés à l'enregistrement.
  La page **Locuteurs** liste les voix enregistrées (nom, nombre
  d'échantillons, conversations où chacun a parlé) ; la page **Mémoire** liste
  les conversations sauvegardées avec, pour chacune, un **résumé** (période,
  participants, thèmes, demandes, dernière réponse — calculé localement, sans
  le cerveau) ou la **transcription** complète. Les deux listes suivent leur
  dossier (`voices_dir`, `memory.dir`) et se mettent à jour d'elles-mêmes.
- **✎ Éditeur** (Ctrl+2) : arborescence des fichiers (dossier Seldon, `data/`,
  `cwd` des actions, dossiers récents ; création, renommage, corbeille) et
  éditeur à onglets, jamais vide (document « Sans titre » par défaut).
  Coloration Python, YAML, JSON, Markdown, TOML, shell. Raccourcis : Ctrl+N
  nouveau document, Ctrl+S / Ctrl+Maj+S enregistrer (sous), Ctrl+F / Ctrl+H
  rechercher / remplacer, Ctrl+/ commenter. Un fichier modifié ailleurs (par
  Seldon, Claude Code…) est rechargé automatiquement.
- **▤ Journal** (Ctrl+3) : menu **Seldon** › Lancer (F5, mode vocal) ou
  Lancer en mode texte (Ctrl+F5), Arrêter (Maj+F5). Le programme principal
  tourne dans un sous-processus et ses logs s'affichent en direct
  (avertissements en ambre, erreurs en rouge) ; en mode texte, le champ sous
  le journal envoie les messages à Seldon. Quand une action demande une
  confirmation (`confirm: console`), le Journal s'affiche, la question
  apparaît en ambre et les boutons **Oui** / **Non** (ou « o » / « n » dans le
  champ) y répondent, en mode vocal comme en mode texte. En mode vocal, le
  bouton **🎙 Micro** (F6) coupe le micro sans arrêter Seldon : plus rien
  n'est entendu, pas même « stop », jusqu'à ce qu'on le rétablisse (signaux
  `SIGUSR1` / `SIGUSR2` du sous-processus, Linux / macOS). Des paramètres non
  enregistrés sont proposés à l'enregistrement avant le lancement.

Thème néomorphique « cassette futurism », accent vert, en variante claire ou
sombre (Affichage › Thème, bouton ☾/☀ ou Ctrl+Maj+L ; « Automatique » suit le
thème du système). Polices Share Tech et Share Tech Mono (Google Fonts, à
installer sur le système ; IBM Plex à défaut). Couleurs et polices dans
`seldon/gui/theme.py`, avec les espacements communs (`SPACE`, `PAD`,
`SECTION`, `GUTTER`) : toute l'interface s'aligne sur ces valeurs, et aucune
sélection (état « modifié », moteur choisi, onglet actif) ne déplace les
éléments.

### Mémoire des conversations

Chaque conversation est sauvegardée dans `data/conversations/` (transcription +
historique complet du cerveau) après chaque échange. Après 10 min de silence,
une nouvelle conversation commence ; l'ancienne reste disponible. Pour reprendre
un sujet, il suffit de le demander (« on reprend notre discussion sur les
tomates ? ») : le cerveau cherche et recharge la conversation via l'outil
`conversations`. Au clavier : `/conversations tomates` puis `/reprendre 1`
(pour ajouter une commande : décorateur `@command` dans
`seldon/core/commands.py`). Dans l'interface, page Paramètres › Mémoire.

Seldon peut aussi **modifier ou supprimer** une conversation à la demande
(« renomme cette conversation Potager », « oublie ce que je t'ai dit sur mon
code », « supprime la conversation sur le vélo ») : opérations `renommer`,
`oublier` (retire les messages contenant un passage) et `supprimer` de l'outil
`conversations`. **Chacune demande une confirmation** (Oui / Non dans le
Journal de l'interface, ou `o` en console) ; sans accord, rien ne change.

**Sujet déjà abordé** : au premier message d'une nouvelle conversation, Seldon
cherche si le sujet a déjà été traité (au moins `related_min_words` mots en
commun, 2 par défaut). Si oui, il demande s'il faut poursuivre l'ancienne
conversation ; en cas d'accord, toute la nouvelle y est rattachée.
`suggest_related: false` désactive la proposition.

**Compression** : toutes les `compress_every_s` secondes (30 min ; `null` =
jamais), les conversations de plus de `compress_max_chars` caractères (20 000)
sont allégées : sauf les `compress_keep` derniers messages (12), leur début
est remplacé par un résumé rédigé par le cerveau (ou calculé localement s'il
ne sait pas résumer). La vérification attend la fin de l'échange en cours.

Commandes terminal (`mode: allowlist`), classées en trois niveaux :

- **bénigne** — lecture seule (`ls`, `grep`, `df`, `git status`…) : s'exécute
  directement, quel que soit le locuteur ;
- **standard** — peut modifier quelque chose (`mkdir`, `git commit`,
  `systemctl restart`, redirection vers un fichier…) : acceptée d'office si la
  voix reconnue fait partie des `admins`, confirmation au clavier sinon ;
- **sévère** — `sudo`, `rm`, `kill`, `find -delete`, `chmod -R`,
  `systemctl stop`, désinstallations, `git push --force`, `git reset --hard`… :
  confirmation au clavier obligatoire, même pour un admin et même en `mode: auto`
  (motifs ajoutables via `extra_severe`).

Les commandes dangereuses (`rm -rf /`, `mkfs`…) sont toujours refusées, et
`allowed_speakers` restreint les commandes standard et sévères (les bénignes
restent ouvertes à tous). Voir `config.yaml`.

### Voix inconnues

Seldon reconnaît les personnes à leur voix. Quand une voix n'est pas encore
enregistrée, il répond puis demande son prénom ; dès que la personne le donne
(ou qu'on la présente : « c'est Paul, un ami »), il enregistre sa voix à partir
des phrases qu'elle vient de prononcer, via l'action `voix`. Pas besoin de
passer par `seldon.tools.enroll`, qui reste utile pour une empreinte plus
précise (phrases lues au calme). On peut aussi lui demander qui il connaît, ou
d'oublier une voix (avec confirmation). Une voix inconnue reste
enregistrable pendant `unknown_memory_s` (10 min par défaut) ; retirer `voix`
de `actions` désactive ce comportement.

Les voix **s'affinent avec l'usage** : une phrase reconnue avec certitude
(`adapt_threshold`, et nettement devant toute autre voix) est ajoutée à
l'empreinte de la personne, au plus une par minute (`adapt_interval_s`).
Seuls les `max_samples` échantillons les plus récents sont gardés, si bien
que la reconnaissance suit un changement de micro, de pièce ou un rhume.

### Accueil

La mémoire sait quand chacun a parlé pour la dernière fois. Après une absence
de plus de `greet_after_s` (6 h par défaut, section `memory`), le cerveau
reçoit une note (« Paul te reparle après 3 jours ; votre dernier échange
portait sur … ») : Seldon salue la personne et peut proposer de reprendre le
sujet. Il accueille aussi une voix connue qui lui parle pour la première fois.

### Attention et mots d'éveil

Avec des mots d'éveil (`stt.options.wake_words`, ex. `["Seldon"]`), Seldon
répond aux phrases qui commencent par l'un d'eux (« Seldon » seul suffit pour
l'appeler), puis reste attentif `follow_up_s` secondes (20 par défaut) après
chaque échange : on peut lui répondre sans le rappeler. Il ne répond alors
que si la phrase lui est adressée. Ce qui est dit hors de ces moments n'est pas
traité, mais les dernières phrases entendues (avec leur locuteur) sont
transmises au cerveau la fois suivante : il sait de quoi on parlait et qui a
dit quoi. Sans mots d'éveil, tout est transmis au cerveau, qui ignore ce qui
ne lui est visiblement pas adressé.

### Couper la parole

Pendant que Seldon parle, le micro reste à l'écoute des sons plus forts que
son propre écho (`barge_in_db`, -35 dBFS par défaut). Un mot d'interruption
(« stop », « attends », « tais-toi », « chut »… ou un mot d'éveil, liste
`interrupt_words`) le fait taire. Un mot qu'il est lui-même en train de
prononcer est ignoré (c'est sans doute son écho). Si la phrase demande autre
chose (« attends, plutôt demain »), elle est traitée juste après. Avec des
haut-parleurs forts, monter `barge_in_db` (ex. -25) ou mettre
`barge_in: false` ; avec un casque, on peut le baisser.

## À faire

- **Mémoire par personne** : retenir les préférences et faits marquants de
  chacun (« Paul est végétarien », « Bastien préfère des réponses courtes ») et
  les rappeler au cerveau dès que sa voix est reconnue (action pour noter /
  oublier, fichier par locuteur à côté de son empreinte vocale).
- **Initiative** : rappels, minuteurs, alertes de fin de longue commande ;
  Seldon prend la parole de lui-même au bon moment (file d'événements planifiés
  injectée dans le pipeline, sans couper quelqu'un qui parle).

## Architecture

| Dossier | Rôle | Contrat |
|---|---|---|
| `seldon/stt/` | Parole → texte | `STTEngine.listen()` produit des `Utterance` |
| `seldon/speakers/` | Qui parle ? | `SpeakerIdentifier.identify(utterance)` |
| `seldon/brain/` | Décision | `Brain.handle(utterance, actions, execute)` |
| `seldon/actions/` | Outils du cerveau | `Action.run(params, ctx) -> ActionResult` |
| `seldon/tts/` | Texte → parole | `TTSEngine.speak(text)` |
| `seldon/core/` | Pipeline, registre de plugins, bus d'événements | |
| `seldon/gui/` | Interface Qt 6 (paramètres, voix et conversations enregistrées, fichiers, éditeur, thème) | lit les plugins par AST, sans les importer |

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
