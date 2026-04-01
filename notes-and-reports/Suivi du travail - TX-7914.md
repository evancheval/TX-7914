# Suivi du travail effectué - TX-7914

## Contexte
Le travail présenté dans ce dépôt s'inscrit dans la TX-7914 de l'UTC, consacrée à la réidentification de personnes dans des vidéos de réunion à partir de détections de type YOLO. Le besoin fonctionnel, formulé dans le sujet, est de conserver un identifiant cohérent pour une même personne tout au long d'une séquence vidéo, même lorsque la détection fluctue d'une frame à l'autre. Le contexte d'usage vise notamment des situations de réunion ou de coaching en temps réel, avec l'objectif de faire remonter des informations comportementales (par exemple la répartition de la parole).

Les étudiants travaillant sur la TX sont Killian Debruyne et Evan Chevalerias, sous la supervision de Claude Moulin, Damien Mortelecq et Thierry Gidel. Le travail a débuté début mars 2026, avec une première version de script de test et une veille technologique sur les détecteurs d'objets. Depuis, le projet a évolué vers une structuration plus formelle des scripts, une préparation rigoureuse des données vidéo, et une annotation manuelle des comportements pour une fenêtre temporelle cible (12m-20m).

## Démarche générale et progression dans le temps
Le dépôt montre une progression en étapes courtes et pragmatiques. Le commit initial du 05/03/2026 crée la base documentaire du projet. Le même jour, un premier bloc technique est ajouté avec un script Python de test, les poids YOLO (version 26n et variante classification), ainsi qu'un document d'état de l'art. À ce stade, la priorité est clairement de vérifier rapidement la faisabilité sur un cas réel avant d'industrialiser le pipeline.

Dans cette première version, le script principal charge YOLO et lance une prédiction sur une vidéo de réunion. Le choix est volontairement direct : valider que la chaîne inférentielle fonctionne bout en bout (chargement du modèle, lecture de la vidéo, affichage des détections) avant de complexifier la logique de suivi d'identité. Cette logique de validation rapide est cohérente avec le sujet, qui nécessite d'abord une détection robuste avant d'aborder la réidentification.

Le 07/03/2026, la veille technologique est enrichie après réunion, avec des pistes alternatives à YOLO (RT-DETR, RF-DETR, Detectron) et des notes de comparaison speed/precision. Cette mise à jour montre que l'équipe n'a pas figé trop tôt le choix de modèle : elle garde ouverte la possibilité de pivoter si les compromis précision/temps réel de YOLO deviennent insuffisants pour la suite.

Le 19/03/2026, une étape importante est franchie avec l'ajout d'outils de préparation vidéo (ffmpeg-cmds) et l'évolution de main.py vers un script paramétré en ligne de commande. Le commit introduit aussi une adaptation de .gitignore pour exclure les données volumineuses et les artefacts de travail local (dossiers data, annotation, fichiers de configuration locaux), ce qui traduit un choix explicite de conserver un dépôt léger et reproductible sans embarquer de médias lourds.

En parallèle des commits, le dépôt contient des notes de réunion et des artefacts d'annotation qui documentent la phase expérimentale : définition de la fenêtre temporelle analysée (12m à 20m), extraction d'images, attribution manuelle d'IDs visibles, puis annotation comportementale par intervalles temporels dans un CSV.

## Travail technique réalisé sur les scripts Python
Le fichier main.py est le coeur exécutable du dépôt. Son évolution est lisible entre les versions du 05/03 et du 19/03. Initialement, le chemin de la vidéo était fixé en dur dans le code, ce qui est pratique pour un premier test mais limite la réutilisation. La version actuelle introduit une interface CLI minimale via sys.argv. Le script vérifie la présence d'un argument, affiche un message d'usage si le chemin media n'est pas fourni, puis termine proprement avec un code d'erreur. Ce comportement évite les exécutions silencieusement incorrectes et facilite l'usage par d'autres membres de l'équipe.

Le script charge ensuite le modèle YOLO depuis models/yolo26n.pt, puis fixe model.overrides['classes'] = 0 avant prédiction. Techniquement, cette surcharge restreint l'inférence à la classe personne (classe indexée 0 dans les modèles COCO utilisés classiquement par Ultralytics), ce qui réduit le bruit de détection sur des objets non pertinents. La prédiction est lancée avec show=True, donc avec visualisation immédiate des bounding boxes, ce qui est utile en phase de diagnostic qualitatif du tracking. Un bloc try/except KeyboardInterrupt a été ajouté pour rendre l'arrêt manuel propre lors de tests interactifs longs.

Ce script reste volontairement simple : il ne calcule pas encore de réidentification explicite à partir d'embeddings ni de logique d'association personnalisée. En revanche, il constitue une base stable pour injecter la prochaine couche algorithmique (association d'IDs persistants), car il formalise déjà l'entrée utilisateur, le chargement modèle et la contrainte de classe.

### Évolution vers le tracking (fin mars 2026)

Une évolution significative a été apportée au script entre le 19/03 et fin mars. Le mode `predict` a été remplacé par `model.track`, ce qui active nativement le mécanisme de suivi d'identité intégré à Ultralytics. Concrètement, là où `predict` se contente de détecter des objets frame par frame de façon indépendante, `track` maintient un algorithme d'association entre frames pour tenter d'attribuer un identifiant persistant à chaque détection dans le temps — par défaut via le tracker BoT-SORT ou ByteTrack selon la configuration.

Deux constantes ont été ajoutées en tête de script : `SHOW = False` et `SAVE = True`. Le premier paramètre désactive la fenêtre d'affichage temps réel, ce qui est utile pour lancer des traitements en batch sans interface graphique. Le second force la sauvegarde des résultats dans le dossier `runs/detect/`, ce qui permet de conserver les vidéos annotées pour analyse différée. La verbosité est aussi coupée (`verbose=False`) pour alléger la sortie console lors des traitements longs.

Par ailleurs, le script accepte désormais un second argument optionnel en ligne de commande pour spécifier le chemin du modèle, ce qui permet de basculer facilement entre `yolo26n.pt` (modèle léger, rapide) et `yolo26x.pt` (modèle plus lourd, plus précis) sans modifier le code. Le modèle `yolo26x.pt` a d'ailleurs été ajouté au dépôt pour permettre cette comparaison.

Quatre runs de tracking sont présents dans `runs/detect/` (track, track2, track3, track4), dont deux ont produit des vidéos annotées conservées : `track3/output-12m-to-20m.avi` et `track4/output-12m-to-20m-5fps.avi`. Ces résultats correspondent à des tests sur la fenêtre expérimentale de référence (12m–20m) avec des cadences d'entrée différentes, ce qui permet de comparer visuellement la stabilité des IDs selon le sous-échantillonnage temporel appliqué en amont.

## Travail technique réalisé sur la préparation des vidéos
Le document ffmpeg-cmds formalise plusieurs commandes utilitaires qui ont servi aux tests. Leur logique est importante pour la reprise car elle conditionne la qualité des données d'entrée.

Une première commande accélère une vidéo avec le filtre setpts=0.5*PTS. Sur le plan technique, PTS représente les Presentation Time Stamps ; multiplier par 0.5 contracte l'axe temporel et double la vitesse de lecture. L'option -an supprime l'audio, ce qui simplifie les fichiers de travail quand seul le flux image est utile.

Les commandes de sous-échantillonnage temporel utilisent -vf fps=1 (ou fps=5 selon les cas) pour ne conserver qu'un nombre contrôlé d'images par seconde. Ce choix répond à deux besoins : réduire les coûts de calcul en phase exploratoire et rendre l'annotation manuelle praticable. Dans les essais, l'équipe a privilégié la fenêtre 12:00 à 20:00 de la vidéo source, en appliquant -ss et -to pour un découpage par timestamps, puis un échantillonnage à 5 fps. Cette méthode est plus lisible et plus sûre que le filtrage direct par indices de frames lorsqu'on veut communiquer un protocole reproductible à des non-spécialistes.

Une autre séquence de commandes illustre une stratégie en deux étapes : extraction d'un segment selon un intervalle de frames via select='between(n,18000,30000)', création d'une vidéo temporaire, puis ré-échantillonnage de cette vidéo temporaire à 0.5 fps avant export des images. Cette décomposition est utile lorsque l'on souhaite dissocier strictement "selection de plage" et "cadence de sortie" pour vérifier chaque transformation intermédiaire.

## Tests de soustraction de fond

En parallèle du travail sur le tracking, une piste de pré-traitement a été explorée : la soustraction du fond vidéo. L'idée est de comparer chaque frame à un état de référence capturé en début de vidéo, quand aucune personne n'est présente dans la salle, afin d'isoler uniquement les zones de mouvement et de réduire le bruit de détection (éléments statiques comme le sol, la table, le tableau).

Deux approches ont été testées. La première, implémentée manuellement à partir d'une différence pixel par pixel entre la frame courante et la frame de référence, s'est révélée insuffisante : les artefacts d'ombres portées, les halos (aura) autour des personnes et les reflets sur les surfaces dégradent fortement le masque résultant et induisent des fausses détections ou des pertes de contour.

La seconde, basée sur l'algorithme MOG2 (Mixture of Gaussians version 2, disponible dans OpenCV via `cv2.createBackgroundSubtractorMOG2`), modélise le fond de façon adaptative en apprenant une distribution gaussienne par pixel sur les premières frames. Elle offre une meilleure robustesse aux variations d'éclairage que la méthode manuelle, mais les résultats obtenus en l'état ne sont pas encore satisfaisants : les paramètres `history`, `varThreshold` et `detectShadows` nécessitent un réglage spécifique à la scène pour obtenir un masque propre. Cette piste reste donc ouverte et sera reprise avec une exploration systématique des hyperparamètres.

## Annotations, tests menés et enseignements
Le fichier annotation/timerange.md verrouille le cadrage expérimental principal : vidéo à 25 fps, plage 00:12:00 à 00:20:00, correspondant aux frames 18000 à 30000. Cette équivalence temps/frame est structurante pour reproduire exactement les mêmes expériences et pour aligner les annotations manuelles avec les sorties d'inférence.

Le fichier annotation/12m-20m-annotation.csv montre une annotation temporelle fine des comportements par personne, avec quatre colonnes : person_id, start_sec, end_sec, annotation. On y trouve des transitions détaillées (standing, walking, sitting down, sitting at the table, leaving the room, entering the room, etc.). Ce format met en évidence un choix méthodologique important : l'équipe n'a pas seulement annoté des boîtes englobantes, elle a aussi capturé des états comportementaux dans le temps, ce qui ouvre la voie à des analyses plus riches que la simple détection.

Les images annotation/ids-1.png et annotation/ids-2.png confirment un travail manuel d'attribution d'identifiants visuels à plusieurs intervenants dans des scènes distinctes (phase debout puis phase assise). Ce matériel sert de vérité terrain légère pour vérifier la cohérence inter-frames et observer les cas où un détecteur/tracker peut perdre ou changer un ID.

Les comptes rendus de réunion indiquent plusieurs tests et hypothèses : comparer plusieurs familles de détecteurs (YOLO, RT-DETR, RF-DETR), comprendre la logique d'association d'IDs de YOLO dans la littérature, expérimenter une exécution YOLO sur une frame sur n pour réduire la charge, et envisager une approche d'embeddings moyens sur l'historique pour stabiliser l'identité. Ces pistes montrent une stratégie itérative qui alterne expérimentation pratique et appui bibliographique.

## Choix techniques et arbitrages observés
Le principal arbitrage à ce stade est un arbitrage de maturité : conserver YOLO comme base opérationnelle immédiate, tout en documentant des alternatives plus performantes en précision potentielle. Ce choix est cohérent avec la contrainte de prototypage rapide et avec la nécessité de disposer d'un pipeline exécutable en continu.

Un second arbitrage concerne les données : les fichiers volumineux de type vidéos, annotations brutes et dossiers locaux sont ignorés par Git. Cela facilite le versionnement des scripts et documents, mais impose pour la reprise de bien transmettre les chemins, formats et procédures d'extraction. Les documents déjà présents couvrent une partie de ce besoin, notamment via les commandes ffmpeg et la note timerange.

Enfin, la trajectoire de travail montre un positionnement clair : la détection seule n'est pas considérée comme suffisante, et le coeur de la valeur attendue se situe dans la stabilisation d'identité lorsque les bounding boxes disparaissent ou se réassignent. Les notes du 20/03 insistent d'ailleurs sur le fait que, hors zone de sortie (porte), une personne ne devrait pas "disparaître" logiquement de la scène ; cette contrainte de bon sens est transformée en piste algorithmique de ré-association.

## Pistes suivies et pistes encore envisagées

La piste du tracking natif YOLO via `model.track` est désormais opérationnelle et constitue le principal fil de travail : il s'agit à présent d'évaluer la qualité de la persistance des IDs sur la fenêtre de référence et de comprendre dans quels cas le tracker perd ou réattribue un identifiant à une personne déjà connue.

La piste de pré-traitement par soustraction de fond a été initiée avec deux algorithmes (approche manuelle et MOG2) et reste ouverte, avec un réglage fin de MOG2 comme prochain objectif court terme. L'idée d'afficher les IDs au centre géométrique des personnes plutôt qu'au coin de la bounding box est également retenue comme amélioration d'affichage à intégrer.

Parmi les pistes encore non tranchées, on trouve l'utilisation d'indices de couleur pour enrichir la ré-identification en cas de perte de tracking, la comparaison entre modèle léger (`yolo26n`) et modèle plus lourd (`yolo26x`) sur la stabilité des IDs, et l'étude plus poussée de la littérature sur la ré-identification de personnes. Le projet conserve une ouverture méthodologique délibérée, sans figer prématurément l'architecture de la solution.

## État actuel pour une équipe de reprise

À la date de ce suivi (début avril 2026), le projet dispose d'un pipeline de tracking fonctionnel basé sur `model.track` d'Ultralytics, de plusieurs vidéos annotées produites par les runs de test, et d'une fenêtre expérimentale de référence bien définie (12m–20m, vidéo `TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4`). Des annotations comportementales manuelles sont disponibles pour servir de vérité terrain partielle.

Le prochain travail critique est double : d'une part, évaluer quantitativement la stabilité des IDs produits par le tracker sur la fenêtre de référence (nombre de changements d'ID par personne, taux de perte de tracking) ; d'autre part, affiner le pré-traitement par soustraction de fond pour améliorer la qualité des détections en entrée du tracker. Pour une reprise efficace, il est recommandé de commencer par visionner les vidéos dans `runs/detect/track3/` et `runs/detect/track4/`, de les comparer aux annotations du fichier `annotation/12m-20m-annotation.csv`, puis de procéder à une exploration des paramètres MOG2 avant d'envisager des modifications plus profondes de l'architecture.
