# Seldon

Assistant vocal polyvalent, modulaire et local par défaut.

```
micro ─► STT (faster-whisper + VAD) ─► identification du locuteur (Resemblyzer)
                                              │
                                              ▼
          cerveau : LLM local (llama.cpp, Ollama), Claude ou OpenAI (+ secours)
                                              │ appels d'outils
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
     tools (seldon/tools)            serveurs MCP (seldon/mcp)      skills (skills/<nom>/)
  speak, run_shell_command,          Home Assistant                 consignes chargées à la
  capture_image, musique, PC,                                       demande (SKILL.md)
  Nextcloud…
```

Le cerveau reçoit « [date | locuteur : X] texte » et décide quelles actions
lancer. La réponse à l'utilisateur est elle-même une action (`speak`).
Les actions se rangent en trois familles, chacune sa section de `config.yaml` :

- **tools** (`tools:`) : code Python qui agit directement — parler, terminal,
  caméra, musique, bureau du PC, Claude Code, Nextcloud ;
- **MCP** (`mcp:`) : serveurs MCP externes dont les outils sont proposés au
  cerveau — Home Assistant ;
- **skills** (`skills:`) : des consignes sans code, qui disent au cerveau comment
  mener un type de demande avec les tools et MCP (voir [Skills](#skills)).

La gestion des conversations et la reconnaissance des voix ne sont pas des
tools : ce sont des plugins de Seldon (`memory:` et `speakers:`), qui proposent
eux-mêmes leur outil au cerveau quand ils sont activés — `conversations` pour
la mémoire, `voix` pour l'identification des locuteurs.

L'ancienne liste unique `actions:` est encore lue (et répartie entre tools et
MCP) ; l'interface la convertit au premier enregistrement.

Quand une action prend la main (commande terminal, caméra, Claude Code,
musique, recherche dans les conversations), Seldon prévient aussitôt qu'il a
compris pendant qu'elle s'exécute, puis donne le résultat. La phrase varie et
suit la demande : « Je regarde l'espace disponible. » pour un `df`, « Je lance
l'album Rumours. », « Je cherche ce qu'on s'est dit à propos des vacances. »…,
jamais deux fois de suite la même. Une seule fois par échange, et pas si le
cerveau a déjà parlé ; `acknowledge: false` dans `config.yaml` le désactive.
Pour une nouvelle action : attribut `acknowledgement` (une phrase ou une liste
de formulations tirées au hasard) ou méthode `acknowledge(params)`, qui peut
s'appuyer sur `self.vary(phrases)`.

Quand une erreur survient pendant un échange, Seldon le dit à voix haute
plutôt que de rester muet : modèle de langage injoignable ou trop lent,
commande en échec, service indisponible, ou action échouée dont le cerveau
n'a pas parlé (« Désolé, l'action terminal a échoué. »). Le détail reste
dans le journal.

Seldon commence à parler avant que le modèle ait fini d'écrire : avec le
cerveau `llamacpp`, la réponse est lue en flux (`stream: true`) et chaque phrase
est dite dès qu'elle est complète ; une coupure de parole arrête aussi la
génération. Au démarrage, une requête de préchauffage fait calculer au serveur
le prompt système et la liste des outils (la première réponse n'attend pas).
L'historique envoyé au modèle reste borné (`max_history_chars`,
`max_history_messages`) : les anciens résultats d'outils sont d'abord abrégés,
puis les échanges les plus anciens retirés, et seule la dernière photo de la
caméra est gardée (`keep_images`).

Ce que disent les gens, ce que renvoient les outils (sortie d'une commande,
fichier lu, réponse de Claude Code) : pour le cerveau, ce sont des données,
jamais des ordres. Les crochets d'une phrase dite ou tapée sont remplacés par
des parenthèses, si bien que personne ne peut imiter les lignes
`[locuteur : …]` ou `[note : …]` que Seldon ajoute lui-même.

## Installation

```bash
git clone git@github.com:golden0kamia/Seldon.git && cd Seldon
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # évite ~3 Go de CUDA
.venv/bin/pip install -e '.[all,dev]'
```

Le cerveau par défaut (`llamacpp`) se connecte à un serveur `llama serve`
(`http://127.0.0.1:9931/v1`, API compatible OpenAI, avec appels d'outils).
Le modèle chargé sur le serveur est utilisé automatiquement. S'il ne répond
pas au démarrage, Seldon lance `./llama-serve.sh` (option `server_command`,
`null` pour ne jamais le lancer), attend que le modèle soit chargé et laisse
le serveur tourner après son arrêt ; sa sortie va dans `llama-serve.log`.
`./llama-serve.sh` le lance avec les réglages de Seldon : il n'écoute que la
machine elle-même et refuse les pages web d'autres origines (`--cors-origins
localhost`) ; les options ajoutées sont transmises (`./llama-serve.sh -c 16384`).
Pour une voix réactive, privilégier un modèle qui tient entièrement en VRAM
et laisser `thinking: false`.

Cerveaux en ligne : `engine: claude` (clé `ANTHROPIC_API_KEY`, ou abonnement : voir plus bas) ou
`engine: openai` (clé `OPENAI_API_KEY`, modèle dans `options.model` ;
`base_url` vise aussi un autre fournisseur compatible OpenAI). Un **cerveau de
secours** se déclare sous `brain.backup`, au même format, ou dans l'interface
(page Cerveau, case « Cerveau de secours ») :

```yaml
brain:
  engine: claude
  options: {model: claude-opus-5-5, effort: low}
  backup:
    engine: openai
    options: {model: gpt-5.6-luna}
```

Claude avec un **abonnement claude.ai** (Pro / Max) plutôt qu'une clé API
facturée au token : `auth: abonnement`. Les requêtes passent par Claude Code
(paquet `claude-agent-sdk`, CLI `claude`), connecté une fois avec
`claude auth login` ; elles comptent dans les limites de l'abonnement. Claude
Code n'y reçoit que les actions de Seldon (aucun de ses outils, réglages ou
serveurs MCP), et `ANTHROPIC_API_KEY` lui est masquée. Usage personnel
uniquement : les conditions d'Anthropic ne permettent pas d'offrir la connexion
claude.ai dans un produit distribué à d'autres.

```yaml
brain:
  engine: claude
  options: {auth: abonnement, model: claude-opus-5-5, effort: low}
```

Si le principal ne démarre pas ou échoue avant d'avoir lancé une action
(serveur arrêté, clé refusée, quota…), la phrase passe au secours ; le principal
est réessayé 5 min plus tard. Chaque échange est recopié dans l'autre cerveau :
la conversation continue sans perte de fil.

Paquets système : `libportaudio2` (micro/haut-parleur), `espeak-ng` (seulement pour le TTS pyttsx3).

### Identifiants

Les secrets (jeton Home Assistant, mots de passe Feishin et Nextcloud, clés API
Anthropic et OpenAI) ne
s'écrivent jamais dans `config.yaml` mais dans un fichier `.env` à côté de lui,
ignoré par git :

```bash
cp .env.example .env && chmod 600 .env   # puis remplir HA_TOKEN, FEISHIN_PASSWORD…
```

Une ligne par variable (`CLÉ=valeur`, `export` et guillemets acceptés, `#` pour
les commentaires). Seldon le charge au démarrage (avec `-c`, celui du dossier de
la configuration) ; une variable déjà exportée dans le shell l'emporte. Un
avertissement signale un `.env` lisible par d'autres utilisateurs. Ces
variables ne sont pas transmises aux commandes lancées par l'action terminal
(`hidden_env`), et lire `.env` y demande une confirmation.

## Utilisation

```bash
python -m seldon                    # mode vocal complet (config.yaml) ; Ctrl+C ou SIGTERM : arrêt propre
python -m seldon --text             # clavier + console ; "Bastien: bonjour" simule un locuteur
python -m seldon.cli.enroll Bastien   # enregistre une empreinte vocale
python -m seldon.cli.wakeword         # entraîne le mot d'éveil sur votre voix
python -m pytest                    # tests (sans micro, caméra ni modèle)
```

Commandes locales (traitées sans passer par le modèle, au clavier ou à la voix) :
`exit` / `quitter`, `muet` / `coupe le micro`, `nouvelle` / `oublie tout`,
`conversations [mots-clés]`, `reprendre N|mots-clés`, `supprimer N`, `actions`,
`aide`. À la voix, le « / » initial est facultatif ; au clavier il est
obligatoire (« quitter » tapé seul part au cerveau comme une phrase ordinaire),
comme pour passer des arguments. Un numéro désigne une ligne de la dernière
liste affichée par `conversations`.

Ces commandes demandent la phrase exacte. Avec le tool `controle` (dans
`tools:`), on peut aussi le demander librement (« tu peux couper ton micro ? »,
« éteins-toi, Seldon ») : le cerveau coupe le micro ou arrête Seldon après sa
réponse. Un simple « stop » ou « tais-toi » ne fait que l'interrompre. Le micro
coupé ne se rétablit qu'avec le bouton **🎙 Micro** du Journal, puisque plus
rien n'est entendu.

La configuration est vérifiée au lancement : une option qui attend un nombre
ou `true` / `false` refuse `"700ms"` ou `"non"`, et une clé mal orthographiée
est signalée avec la clé probable.

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
  Coloration Python, YAML, JSON, Markdown, TOML, shell (désactivée au-delà
  de 1 Mo). Raccourcis : Ctrl+N nouveau document, Ctrl+S / Ctrl+Maj+S
  enregistrer (sous), Ctrl+F / Ctrl+H rechercher / remplacer, Ctrl+/ commenter.
  Encodage (UTF-8, Latin-1) et fins de ligne sont conservés et affichés dans
  la barre d'état ; l'enregistrement est atomique (jamais de fichier à moitié
  écrit). Un fichier modifié ailleurs (par Seldon, Claude Code…) est rechargé
  automatiquement. Créer ou renommer refuse les chemins absolus et `..`, et
  n'écrase jamais un fichier existant.
- **▤ Journal** (Ctrl+3) : menu **Seldon** › Lancer (F5, mode vocal) ou
  Lancer en mode texte (Ctrl+F5), Arrêter (Maj+F5), Redémarrer (Ctrl+Maj+F5).
  Le programme principal
  tourne dans un sous-processus et ses logs s'affichent en direct
  (avertissements en ambre, erreurs en rouge), filtrables par niveau (Tout,
  Sans DEBUG, Avertissements, Erreurs) ; en mode texte, le champ sous le
  journal envoie les messages à Seldon. Quand une action demande une
  confirmation (`confirm: console`), le Journal s'affiche et la question
  apparaît dans un bandeau ambre dont les boutons **Oui** / **Non** (ou « o » /
  « n » dans le champ) y répondent, en mode vocal comme en mode texte. Seules
  les questions de Seldon lui-même sont reconnues : chaque lancement reçoit un
  jeton secret (`SELDON_GUI_TOKEN`, retiré aussitôt de l'environnement), et une
  ligne qui imite la balise (sortie d'une commande, réponse du modèle) n'affiche
  ni fausse question ni faux état du micro. **Arrêter** envoie Ctrl+C à Seldon
  et à ses sous-processus, un second appui force l'arrêt, et fermer
  l'interface arrête Seldon (rien ne garde le micro). **↻ Redémarrer** fait
  le même arrêt propre puis relance Seldon dans le même mode (vocal ou texte),
  par exemple pour appliquer une configuration modifiée ; Arrêter pendant le
  redémarrage l'annule. En mode vocal, le
  bouton **🎙 Micro** (F6) coupe le micro sans arrêter Seldon : plus rien
  n'est entendu, pas même « stop », jusqu'à ce qu'on le rétablisse (signaux
  `SIGUSR1` / `SIGUSR2` du sous-processus, Linux / macOS). Des paramètres non
  enregistrés sont proposés à l'enregistrement avant le lancement. Dans les
  Paramètres, faire défiler la page ne modifie jamais un champ, et
  `config.yaml` est validé puis écrit de façon atomique ; s'il est illisible,
  l'erreur s'affiche avec sa ligne et sa colonne.
- **◈ Structure** (Ctrl+4) : schéma bloc des sous-systèmes de Seldon (micro,
  reconnaissance vocale, pipeline, cerveau, actions, synthèse vocale, mémoire…)
  et de ce qu'ils s'échangent, avec les moteurs disponibles dans chacun. Un
  clic sur un bloc éclaire ses liens et le décrit ; un double-clic ouvre son
  code.

Thème néomorphique « cassette futurism », accent vert, en variante claire ou
sombre (Affichage › Thème, bouton ☾/☀ ou Ctrl+Maj+L ; « Automatique » suit le
thème du système). Polices Share Tech et Share Tech Mono (Google Fonts, à
installer sur le système ; IBM Plex à défaut). Couleurs et polices dans
`seldon/gui/theme.py`, avec les espacements communs (`SPACE`, `PAD`,
`SECTION`, `GUTTER`) : toute l'interface s'aligne sur ces valeurs, et aucune
sélection (état « modifié », moteur choisi, onglet actif) ne déplace les
éléments. Les textes respectent un contraste d'au moins 4,5:1 (WCAG AA) dans
les deux thèmes, le focus clavier est entouré de vert, et les boutons à icône
(☾/☀, 🎙, ✕…) ont un nom lisible par les lecteurs d'écran.

### Mémoire des conversations

Chaque conversation est sauvegardée dans `data/conversations/` (transcription +
historique complet du cerveau) après chaque échange. Conversations, empreintes
vocales et photos sont des données personnelles : fichiers en `0600`, dossiers
créés en `0700`, écriture atomique. Un identifiant de conversation fourni par
le cerveau ne peut désigner qu'un fichier de ce dossier. Après 10 min de silence,
une nouvelle conversation commence ; l'ancienne reste disponible. Pour reprendre
un sujet, il suffit de le demander (« on reprend notre discussion sur les
tomates ? ») : le cerveau cherche et recharge la conversation via l'outil
`conversations`, fourni par la mémoire (désactivé avec `enabled: false`). Au clavier : `/conversations tomates` puis `/reprendre 1`
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
ne sait pas résumer). Le résumé est rédigé sans bloquer Seldon : seul son
enregistrement attend la fin de l'échange en cours.

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
restent ouvertes à tous ; la casse des noms est ignorée). Voir `config.yaml`.

Le modèle peut être manipulé (sortie d'une commande précédente, fichier lu,
autre voix dans la pièce) ; la classification ne se fie donc qu'à la ligne
elle-même :

- une commande n'est bénigne que si chaque option est sans effet : `env -S…`,
  `sort --compress-program`, `tree -o`, `ss -K`, `git -c…`, `git branch -m`,
  une variable placée devant (`GIT_EXTERNAL_DIFF=… git diff`, `PATH=… ls` ;
  seules `LC_*`, `LANG`, `TZ`… sont admises) ou un programme hors des
  dossiers système (`./ls`) la rendent au moins standard ;
- est **sévère** toute commande dont l'effet ne se lit pas dans la ligne : code
  en ligne (`bash -c`, `python -c`, `perl -e`, `sed …e`), lanceurs (`xargs`,
  `nohup`, `timeout`, `env CMD`, `find -exec`), substitutions (`$(…)`,
  `` `…` ``), noms masqués (`"r"m`, `$X`, `/???/r?`, `{rm,-rf,x}`), alias ou
  pager git, ou écriture dans un fichier lancé automatiquement (`.bashrc`,
  `authorized_keys`, `.git/hooks`, crontab) ;
- lire un secret (`~/.ssh/`, `.env`, `~/.aws/`, `/etc/shadow`,
  `/proc/*/environ`, `ps e`…, motifs `sensitive` / `extra_sensitive`) est
  sévère : n'importe quelle voix pourrait le demander, et avec un cerveau
  distant il quitterait la machine ;
- les commandes tournent sans les variables secrètes de Seldon (`hidden_env` :
  clés d'API, jetons, `BASH_ENV`…) ; l'agent SSH reste disponible ;
- la sortie est lue par morceaux : au-delà de `max_read_bytes` (16 Mo) la
  commande est arrêtée (`cat /dev/zero`), et un délai dépassé renvoie la
  sortie obtenue jusque-là.

L'analyse reste statique : pour un admin, une commande standard (`make`,
`python script.py`…) s'exécute sans confirmation ; `mode: confirm` fait tout
valider.

L'action `claude_code` (Claude Code en mode non interactif) ne travaille que
sous `allowed_dirs` (par défaut son `cwd`), reçoit la tâche comme un texte
(jamais comme une option de la ligne de commande), plafonne sa sortie
(`max_output_bytes`) et demande toujours confirmation avec
`permission_mode: bypassPermissions`. Une annonce de fin en arrière-plan
attend la fin de l'échange en cours.

### Musique : Feishin

L'action `feishin` (outil `control_music`) pilote le lecteur Feishin lancé sur
l'ordinateur : titre en cours, lecture/pause, suivant/précédent, volume,
position, aléatoire, répétition, favori, note, recherche dans la bibliothèque
et lecture d'un titre, d'un album ou d'une playlist (« mets l'album Hearts
d'America »). Elle passe par le serveur « Remote control » de Feishin
(Paramètres > Général, port 4333, extra `seldon[music]`) avec les identifiants
`FEISHIN_USERNAME` / `FEISHIN_PASSWORD` du fichier `.env` (voir
[Identifiants](#identifiants)). Sans ce serveur, les
commandes de base passent par MPRIS (D-Bus) ; la recherche, les favoris et les
notes exigent le Remote control. Pas de recherche par artiste : Seldon cherche
ses albums ou ses titres.

### Domotique : Home Assistant

Le serveur MCP `home_assistant` (section `mcp:`) pilote la maison (lumières, prises, volets,
chauffage, médias, minuteurs…) via le serveur MCP de Home Assistant. Tous les
outils d'Assist (HassTurnOn, HassLightSet, GetLiveContext…, et vos scripts
exposés) passent par un seul outil `home_assistant`.

1. **Activer le serveur MCP** : *Paramètres → Appareils et services → Ajouter
   une intégration → « Model Context Protocol Server »*, API LLM « Assist ».
   Il répond sur `http://<ip-de-HA>:8123/api/mcp`.
2. **Choisir les appareils** : *Paramètres → Assistants vocaux → Exposer*.
   Seules les entités exposées à Assist sont accessibles.
3. **Créer un jeton longue durée** : profil (en bas à gauche) → *Sécurité* →
   *Jetons d'accès longue durée* → *Créer un jeton*, avec un compte
   administrateur (exigé par défaut par l'intégration).
4. **Le donner à Seldon sans l'écrire dans la configuration** : ligne
   `HA_TOKEN=…` (ou `HOME_ASSISTANT_TOKEN`) dans le fichier `.env` (voir
   [Identifiants](#identifiants)), ou `export HA_TOKEN="…"` avant de lancer Seldon.

Si Home Assistant est injoignable au démarrage, Seldon démarre quand même et
réessaie au premier appel. Une **confirmation** est demandée avant de
déverrouiller une serrure, désarmer une alarme ou ouvrir un portail ou un
garage (`confirm_domains`, `confirm_keywords`, `confirm_tools`) ;
`allowed_tools` / `denied_tools` limitent les outils proposés au modèle.

### Nextcloud

L'outil `nextcloud` (section `tools:`) parle directement à Nextcloud en WebDAV,
sans serveur intermédiaire (`<NEXTCLOUD_HOST>/remote.php/dav`) :

- **fichiers** (`files`) : lister, lire, écrire ou compléter, déplacer,
  supprimer, chercher par nom ;
- **notes** (`notes`) : les fichiers `.md` / `.txt` du dossier de l'application
  Notes (`notes_folder`, catégories = sous-dossiers) : lister, lire, écrire ou
  compléter, chercher dans le titre et le texte ;
- **agenda** (`calendar`, CalDAV) : événements d'un ou plusieurs jours
  (répétitions comprises), création ;
- **tâches** (`tasks`, CalDAV) : tâches ouvertes (`days` : seulement celles
  dues d'ici là, retards compris), création, marquer terminée ;
- **contacts** (`contacts`, CardDAV) : recherche par nom, courriel ou téléphone.

1. **Créer un mot de passe d'application** dans Nextcloud : *Paramètres
   personnels → Sécurité → Créer un nouveau mot de passe d'application*.
2. **Le mettre dans `.env`** (voir [Identifiants](#identifiants)) :
   `NEXTCLOUD_HOST=https://cloud.example.com`, `NEXTCLOUD_USERNAME=…`,
   `NEXTCLOUD_PASSWORD=<mot de passe d'application>`.
3. **Installer httpx** si besoin : `pip install -e '.[nextcloud]'`.

`apps` choisit les applications proposées au modèle, `calendars` limite les
agendas consultés et `default_calendar` reçoit les nouveaux événements et
tâches. Les chemins restent dans les fichiers de l'utilisateur (`..` refusé).
Une **confirmation** est demandée avant de supprimer, de déplacer ou d'écraser
un fichier ou une note ; `confirm_writes: true` la demande aussi pour chaque
création ou modification.

### Contrôle du PC

L'action `desktop` (outil `control_computer`) pilote l'ordinateur hôte, pensée
pour KDE Plasma 6 sous Wayland (avec replis GNOME et génériques) :

- **notifications lues à voix haute** (`read_notifications`, activable à la
  voix) : mode ne pas déranger respecté, applications ignorées (`ignore_apps`),
  rafales regroupées, débit limité (`max_per_minute`), annonce qui attend la
  fin d'un échange ; rappel des dernières notifications reçues ;
- luminosité des écrans, moniteurs externes compris (API ScreenBrightness de
  Plasma, sinon `ddcutil`) ; mode **ne pas déranger** avec durée ;
- volume et sourdine (`wpctl`/`pactl`), lecteurs multimédia (MPRIS), batterie ;
- verrouillage de l'écran, mise en veille (toujours confirmée), ouverture d'une
  application installée ou d'une adresse web, envoi d'une notification, capture
  d'écran transmise au modèle.

Aucun shell n'est utilisé, chaque fonction se désactive proprement si l'outil
système manque, et `commands` restreint les fonctions proposées. Le texte des
notifications est transmis au modèle comme une donnée, pas comme une consigne.

### Voix inconnues

Seldon reconnaît les personnes à leur voix. Quand une voix n'est pas encore
enregistrée, il répond puis demande son prénom ; dès que la personne le donne
(ou qu'on la présente : « c'est Paul, un ami »), il enregistre sa voix à partir
des phrases qu'elle vient de prononcer, via l'action `voix`. Pas besoin de
passer par `seldon.cli.enroll`, qui reste utile pour une empreinte plus
précise (phrases lues au calme). On peut aussi lui demander qui il connaît, ou
d'oublier une voix (avec confirmation). Ajouter une voix inconnue à une voix
déjà enregistrée (« je suis Bastien ») demande une **confirmation** : sinon,
n'importe qui pourrait prendre l'identité, et les droits, d'un admin. Une voix inconnue reste
enregistrable pendant `unknown_memory_s` (10 min par défaut). L'action `voix`
est fournie par l'identification des locuteurs : elle disparaît avec
`speakers: none`.

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

**Entraîner le mot d'éveil** : Whisper écrit parfois un nom propre de travers
(« Celdon », « Sel don »…), et le mot d'éveil n'est alors pas reconnu.
`python -m seldon.cli.wakeword` fait prononcer le mot d'éveil une dizaine de
fois (seul ou en début de phrase), retient les transcriptions qui diffèrent,
puis fait lire quelques phrases sans le mot d'éveil et écarte toute variante
qui y apparaît en tête (elle réveillerait Seldon à tort). Les variantes sont
ajoutées à `stt.options.wake_variants` (`data/wake_words.json`) et lues au
démarrage ; `--list` les affiche, `--reset` les oublie. Si le mot est rarement
bien transcrit, `initial_prompt: "Seldon"` aide aussi Whisper.

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
| `seldon/speakers/` | Qui parle ? (et outil `voix`) | `SpeakerIdentifier.identify(utterance)`, `.actions()` |
| `seldon/brain/` | Décision | `Brain.handle(utterance, actions, execute)` |
| `seldon/tools/` | Tools du cerveau (Python) | `Action.run(params, ctx) -> ActionResult` (`seldon/core/action.py`) |
| `seldon/mcp/` | Client MCP et serveurs branchés (Home Assistant) | `MCPAction` : un outil par serveur |
| `seldon/skills/` | Chargement des skills (`skills/<nom>/SKILL.md`) | outil `skill` |
| `seldon/cli/` | Outils hors ligne : `enroll`, `wakeword` | `python -m seldon.cli.…` |
| `seldon/tts/` | Texte → parole | `TTSEngine.speak(text)` |
| `seldon/core/` | Pipeline, registre de plugins, bus d'événements, mémoire, écriture privée des fichiers (`storage.py`), balises lues par l'interface (`tags.py`) | |
| `seldon/gui/` | Interface Qt 6 (paramètres, voix et conversations enregistrées, fichiers, éditeur, journal, schéma des sous-systèmes, thème) | lit les plugins par AST, sans les importer |

### Skills

Un skill est une marche à suivre écrite en français, sans code : un dossier
`skills/<nom>/` avec un `SKILL.md` dont l'en-tête dit **quand** l'utiliser.

```markdown
---
name: bilan-du-matin
description: Fait le point de la journée quand on le demande (agenda, tâches, maison).
---
1. Agenda : outil `nextcloud`, commande `events`…
2. Réponds avec `speak` en trois ou quatre phrases.
```

Seuls le nom et la description sont envoyés au cerveau à chaque échange, dans
l'outil `skill` ; quand une demande correspond, il appelle `skill` pour lire les
consignes, puis les suit avec les tools et serveurs MCP activés. Les autres
fichiers du dossier (modèles, listes…) se lisent avec le paramètre `file`, sans
pouvoir sortir du dossier. Les consignes sont relues à chaque appel : une
modification s'applique sans redémarrer (pas le nom ni la description).

Pour activer un skill, ajoutez son nom (ou le chemin de son dossier) à
`skills:` dans `config.yaml`, ou cochez-le dans *Paramètres → Skills* ; le
bouton *Nouveau skill* y crée un `SKILL.md` à compléter. Exemple fourni :
`skills/bilan-du-matin`.

### Ajouter un tool

```python
# seldon/tools/meteo.py
from seldon.core.action import Action
from seldon.core.registry import register
from seldon.core.types import ActionResult

@register("tools", "meteo")
class Weather(Action):
    name = "get_weather"
    description = "Donne la météo actuelle d'une ville."
    input_schema = {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"], "additionalProperties": False}

    async def run(self, params, ctx):
        return ActionResult(text=f"Il fait beau à {params['city']}.")
```

Puis l'ajouter au dictionnaire `_BUILTINS` de `seldon/core/registry.py` et à la
liste `tools:` de `config.yaml`. Un nouveau serveur MCP dérive de `MCPAction`
(`seldon/mcp/base.py`), s'enregistre avec `@register("mcp", …)` et s'active sous
`mcp:`. Une classe hors du projet peut aussi être
référencée directement : `engine: mon_paquet.module:MaClasse`.

Le même mécanisme vaut pour les autres étages (nouveau STT, TTS, cerveau…).
Pour observer le pipeline sans le modifier (journal, interface, domotique…),
abonnez-vous aux événements : `assistant.events.on("action.end", handler)`.
Les services partagés (TTS, STT…) sont accessibles aux actions via `ctx.services`.
