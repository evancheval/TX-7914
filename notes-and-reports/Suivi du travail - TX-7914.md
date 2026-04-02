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

### Interpolation de boîtes manquantes et raffinement du tracking (avril 2026)

Une nouvelle itération en cours (changements non-committés au 02/04) porte le script main.py vers une logique beaucoup plus sophistiquée. L'enjeu central abordé est la stabilisation des IDs lorsqu'une personne disparaît temporairement du cadre de détection (occultation, pose ambiguë, etc.). Le script implémente désormais une stratégie d'interpolation temporelle sur une fenêtre glissante de frames (`gap_frame`).

La nouvelle interface CLI utilise argparse avec quatre paramètres majeurs :
- `input` : chemin du fichier média à traiter,
- `--show` : affiche deux vidéos côte à côte (détections originales vs résultat traité),
- `--save` : enregistre la vidéo annotée dans `runs/detect/`,
- `--model_path` : permet de basculer entre modèles légers et lourds,
- `--gap_frame` : nombre maximal de frames conservées dans l'historique pour interpolation (par défaut 10),
- `--max-box-shift` : seuil en pixels pour considérer deux boîtes comme étant la même personne entre deux frames (par défaut 10, calibré pour résolution 360×640).

Le cœur du traitement repose sur plusieurs fonctions complémentaires :

1. **draw_boxes()** : affiche les boîtes englobantes et les identifiants au centre géométrique de chaque personne détectée (amélioration visuelle par rapport à l'affichage au coin).

2. **find_near_boxes()** et **find_nearest_box()** : implémentent la logique d'interpolation. Lorsqu'une boîte disparaît d'une frame à une autre, ces fonctions cherchent la boîte la plus proche dans les frames suivantes en utilisant une distance euclidienne sur les quatre coins (x1, y1, x2, y2). Le seuil `max-box-shift` limite la recherche pour éviter les fausses associations.

3. **lost_id()** : détecte si un ID (identifiant de personne) a été "perdu" entre deux frames — c'est-à-dire si une personne s'est volontairement retirée de la scène ou s'est occultée au-delà du seuil d'interpolation. Cette information est affichée en superposition sur la vidéo de sortie ("LOST ID").

Le traitement utilise `model.track()` en mode streaming (`stream=True`), ce qui permet de traiter des vidéos de grande taille sans charger la totalité en mémoire. À chaque nouvelle frame, le script accumule les résultats dans un historique (`results_trough_time`) limité à `gap_frame` éléments. Une fois l'historique suffisamment rempli, il applique une interpolation en prenant la frame au centre temporel de la fenêtre (index `median_index`), ce qui lisse les apparus/disparitions ponctuelles tout en conservant une latence acceptable. L'affichage côte à côte (avec cv2.hconcat) permet de comparer en temps réel les boîtes originales du modèle avec les boîtes interpolées.

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
Le principal arbitrage observé dans l'évolution du projet reflète une transition : passage d'une approche purement basée sur la détection indépendante (predict) vers une approche hybride alliant tracking natif et interpolation spatiale. Ce pivot est motivé par l'observation que la persistance d'identité dépend moins de la précision brute de détection que de la cohérence géométrique d'une frame à la suivante, surtout dans un contexte de vidéo statique (réunion dans une salle).

Un arbitrage de maturité maintient YOLO comme base opérationnelle, avec une alternative (modèle 26x) disponible pour comparaison, plutôt que de chercher à changer complètement de famille de détecteur.

Un second arbitrage concerne les données : les fichiers volumineux de type vidéos, annotations brutes et dossiers locaux sont ignorés par Git. Cela facilite le versionnement des scripts et documents, mais impose pour la reprise de bien transmettre les chemins, formats et procédures d'extraction. Les documents déjà présents couvrent une partie de ce besoin, notamment via les commandes ffmpeg et la note timerange.

Enfin, la trajectoire de travail montre une orientation claire : la détection seule n'est pas considérée comme suffisante, et le cœur de la valeur attendue se situe dans la stabilisation d'identité lorsque les bounding boxes disparaissent ou se réassignent. Les notes du 20/03 insistent d'ailleurs sur le fait que, hors zone de sortie (porte), une personne ne devrait pas "disparaître" logiquement de la scène ; cette contrainte de bon sens est maintenant transformée en piste algorithmique de ré-association via interpolation.

## Pistes suivies et pistes encore envisagées

Le fil principal de travail est maintenant clairement l'**interpolation temporelle de boîtes manquantes** couplée au tracking natif. L'intuition clé est que dans une scène de réunion, les pertes ponctuelles de détection (1 à quelques frames) ne devraient pas entraîner un nouvel identifiant pour une personne : il suffit de faire correspondre géométriquement la boîte de la frame suivante avec la dernière boîte connue. Cette approche hybride (tracking YOLO + interpolation spatiale) vise à réduire le nombre de changements d'ID tout en gardant une latence minimale.

La piste de **pré-traitement par soustraction de fond** (MOG2) reste ouverte mais moins prioritaire en ce moment ; elle pourrait être intégrée en amont du tracking si les tests d'interpolation révèlent que le problème principal est le bruit de détection plutôt que la perte de tracking.

L'**affichage visuel des identifiants au centre des bounding boxes** a été implémenté, ce qui améliore la lisibilité en cas de résolution variable.

Une piste **détection de perte d'ID** (`lost_id()`) a été ajoutée pour mettre en évidence dans la vidéo de sortie les moments où une personne a logiquement quitté la scène plutôt que d'être simplement occultée temporairement.

Parmi les pistes encore envisagées mais non tranchées, on trouve l'affinement des paramètres `gap_frame` et `max-box-shift` via une validation sur le corpus d'annotations existant, l'exploration d'autres trackers (Kalman, Deep SORT) en remplacement de ByteTrack, et l'éventuel ajout d'une couche d'embeddings pour enrichir la ré-identification en cas de vrai changement d'identité plutôt que simple perte.

## État actuel pour une équipe de reprise

À la date de ce suivi (début avril 2026), le projet dispose d'un **pipeline de tracking avec interpolation temporelle** en cours de développement (code non-commis au 02/04). Le code formalise une approche hybride : utilisation du tracker natif YOLO pour l'association d'identités entre frames successives, couplée à une logique d'interpolation géométrique pour récupérer les IDs lorsqu'une personne disparaît temporairement du champ de détection.

Les artefacts produits incluent :
- Plusieurs vidéos annotées dans `runs/detect/track*/` issues des tests précédents,
- Annotations comportementales manuelles (`annotation/12m-20m-annotation.csv`) servant de vérité terrain,
- Une fenêtre expérimentale de référence bien définie (12m–20m, vidéo `TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4` à 25 fps),
- Paramètres de préparation vidéo documentés (extraction via ffmpeg, sous-échantillonnage temporel).

Le prochain travail critique comprend :
1. **Tester et valider** les paramètres d'interpolation (`gap_frame`, `max-box-shift`) sur la fenêtre de référence en comparaison avec les annotations manuelles,
2. **Mesurer quantitativement** la réduction du bruit d'ID (nombre de changements d'identifiant par personne et par vidéo),
3. **Affiner ou explorer d'autres trackers** si l'interpolation seule ne suffit pas,
4. **Committer et documenter** les changements une fois validés.

Pour une reprise efficace, il est recommandé de :
- Relancer le script déjà amélioré avec `--show` pour observer visuellement l'effet de l'interpolation sur la fenêtre 12m–20m,
- Comparer les vidéos générées avec le fichier d'annotations pour quantifier les gains en termes de stabilité d'ID,
- Explorer les valeurs de `gap_frame` et `max-box-shift` adaptées à la résolution réelle des vidéos testées (pas nécessairement 360×640),
- Intégrer le pré-traitement MOG2 seulement si les résultats d'interpolation seule restent insuffisants.
