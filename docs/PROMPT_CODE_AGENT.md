# Cahier des charges technique et prompt pour agent de code — SciLit

> **Dépôt cible :** [OmarNaceur2001/scilit-](https://github.com/OmarNaceur2001/scilit-)  
> **But :** faire évoluer le prototype actuel vers un assistant de revue de littérature traçable et évalué, sans réécrire inutilement les composants qui fonctionnent.  
> **Usage :** ce document peut être fourni tel quel à GitHub Copilot Coding Agent, Codex ou à un autre agent de programmation capable de lire et modifier le dépôt.

---

# PROMPT À DONNER À L’AGENT DE CODE

## 1. Rôle et mission

Agis comme un **ingénieur logiciel senior spécialisé en NLP scientifique, retrieval, évaluation expérimentale et systèmes fiables avec LLM**. Tu travailles directement dans le dépôt existant `OmarNaceur2001/scilit-`.

Ta mission n’est pas de produire uniquement des recommandations ou un plan. Tu dois **inspecter le code, implémenter les changements nécessaires, écrire des tests, exécuter les vérifications disponibles, documenter les résultats et laisser une application fonctionnelle**.

Le produit visé est :

**SciLit — assistant de revue de littérature scientifique : recherche d’articles, extraction d’affirmations (claims), liaison des claims aux preuves, comparaison des travaux, rédaction assistée et vérification des citations phrase par phrase.**

La contribution principale doit être la **traçabilité vérifiable entre chaque phrase générée, le claim qui la soutient et le passage exact de l’article source**. Un simple chatbot RAG, un résumé de PDF ou une génération de citations sans contrôle ne répond pas à cette mission.

## 2. Règles impératives de travail

1. **Commence par auditer le dépôt et exécuter les tests existants.** Lis `README.md`, `pyproject.toml`, les schémas, les pipelines de collecte/parsing/indexation/retrieval, l’extraction des claims, la génération, la vérification NLI, l’API et les tests. Vérifie ce qui fonctionne réellement avant de le modifier.
2. **Ne réécris pas le projet de zéro.** Préserve les modules réutilisables, les interfaces publiques, les identifiants stables et la compatibilité des données existantes autant que raisonnable.
3. **Travaille par petits changements cohérents.** Après chaque étape, exécute les tests pertinents. Ne termine pas avec une grande réécriture non testée.
4. **N’invente jamais de résultats, de données, de références, de métriques ou d’annotations.** Les données de démonstration doivent être clairement identifiées comme fixtures synthétiques.
5. **Ne marque jamais une citation “verified” simplement parce que le DOI, le titre ou l’article existe.** Il faut aussi contrôler le soutien apporté à la phrase par le passage source.
6. **Ne confonds pas “non vérifiable” avec “halluciné”.** Un article absent du corpus peut exister ailleurs. Utilise des statuts distincts pour article introuvable, preuve insuffisante, contradiction et vérification impossible.
7. **Aucune API payante ne doit être nécessaire pour le système évalué.** La génération déterministe doit continuer à fonctionner sans clé API. Les modèles externes éventuels doivent être optionnels, configurables et documentés. N’expose ni ne commit de secrets.
8. **Pas de téléchargement de modèle ou d’accès réseau imposé lors des tests unitaires.** Mocke les clients externes et les modèles coûteux. Les tests d’intégration dépendant d’un corpus ou d’un modèle doivent être identifiés explicitement.
9. **Respecte les licences et l’accès ouvert.** Stocke la provenance, l’URL, le DOI, la licence déclarée et la date de collecte. Ne commit pas un corpus de PDF volumineux ni des données dont la redistribution n’est pas autorisée.
10. Si une fonctionnalité ne peut pas être implémentée ou évaluée dans l’environnement disponible, explique précisément le blocage et laisse un comportement sûr — jamais un faux succès ou une métrique fabriquée.

## 3. Contexte technique existant à confirmer

Le dépôt contient déjà une base Python organisée sous `scilit/src/scilit/`, avec Pydantic v2, FastAPI et pytest. Les modules existants incluent notamment :

- `corpus/collect.py` : collecte via OpenAlex et sauvegarde de métadonnées en JSONL ;
- `corpus/parse.py` : parsing d’abstracts et mode PDF avec PyMuPDF, découpage en sections/chunks ;
- `corpus/index.py` : index lexical BM25, index dense FAISS et fusion RRF ;
- `retrieval/pipeline.py` : retrieval et enrichissement des résultats ;
- `extraction/claims.py` : extraction de claims principalement fondée sur règles/spaCy ;
- `extraction/fetch_refs.py` et `extraction/citations.py` : récupération de références et graphe de citations à partir des métadonnées disponibles ;
- `generation/template.py` : génération déterministe d’un brouillon de revue ;
- `generation/nli_verify.py` : vérification NLI d’un brouillon ;
- `schemas.py` : modèles de données et statuts de vérification ;
- `evaluation/bronze_eval.py` : base d’évaluation ;
- `api/main.py` et `api/ui.html` : API et interface ;
- tests de régression et tests d’intégration.

**Ne suppose pas que tous ces éléments sont complets ou corrects.** Par exemple, le graphe de références basé sur les métadonnées n’équivaut pas forcément à une extraction des citations dans le texte intégral, et une validation NLI automatique n’est pas une preuve infaillible. Confirme ces limites dans le code et rends-les visibles dans la documentation et l’interface.

## 4. Objectif produit et parcours utilisateur

Un utilisateur saisit une question de recherche, par exemple :

> “How do LoRA and full fine-tuning compare for low-resource NLP?”

SciLit doit effectuer la chaîne suivante :

1. valider et normaliser la requête ;
2. rechercher les articles et passages pertinents ;
3. présenter les métadonnées vérifiables des articles ;
4. extraire des informations structurées et des claims depuis les textes réellement disponibles ;
5. relier chaque claim à son article, sa section et son passage source ;
6. comparer les méthodes uniquement lorsque les critères sont comparables ;
7. générer une revue de littérature phrase par phrase à partir des preuves retrouvées ;
8. vérifier les références et le soutien de chaque phrase ;
9. distinguer les phrases soutenues, non soutenues, contredites et non vérifiables ;
10. permettre à l’utilisateur de consulter les preuves sources et d’exporter la revue/BibTeX.

Le système doit clairement distinguer **article trouvé**, **claim extrait**, **preuve liée** et **claim vérifié**. Le retrieval seul ne prouve pas la véracité d’un claim.

## 5. Architecture cible

Conserve une architecture modulaire ; évite une unique fonction monolithique.

```text
Question de recherche
        |
        v
Validation / compréhension de la requête
        |
        v
Recherche d'articles et de passages
(BM25 + dense optionnel + fusion/reranking)
        |
        v
Articles et chunks avec provenance
        |
        v
Parsing / sections / chunks
        |
        v
Extraction structurée
(question, contribution, méthode, données,
résultats, limites, métriques, hedging)
        |
        v
Claims avec offsets/passages sources
        |
        v
Liaison Claim <-> Evidence <-> Article
        |
        v
Tableau comparatif traçable
        |
        v
Génération contrainte de la revue
        |
        v
Vérification par phrase et par citation
        |
        +------> Soutenue / VERIFIED
        +------> Article présent mais soutien insuffisant
        +------> Contradiction détectée
        +------> UNVERIFIABLE
        |
        v
UI + API + export BibTeX + audit
```

Le système doit rester utilisable sous une forme Bronze déterministe, sans LLM, et permettre l’ajout progressif de composants appris.

## 6. Priorité 0 — Audit, stabilité et démarrage reproductible

Avant d’ajouter des fonctionnalités :

- lancer `python -m pytest` ou `pytest`, puis `ruff check .` si disponible ;
- noter les tests qui échouent au départ, sans les masquer ;
- inspecter `pyproject.toml`, la configuration, les chemins de données et le chargement de `.env` ;
- repérer les imports inutilisés, exceptions trop larges, chemins codés en dur, écritures non atomiques et configurations incohérentes ;
- s’assurer que le dépôt s’installe avec `pip install -e ".[dev]"` ;
- ajouter/compléter un workflow CI exécutant tests, lint et vérification de types si elle peut être rendue fiable ;
- documenter les commandes reproductibles dans le README.

Préserve le comportement historique des variables d’environnement déjà documentées, sauf raison explicite et tests de compatibilité.

### Critères d’acceptation

- le paquet s’installe dans un environnement propre ;
- l’API démarre sans corpus, retourne un health check utile et explique les données manquantes au lieu de renvoyer une erreur opaque ;
- les tests unitaires n’ont pas besoin d’OpenAlex, d’Internet ou de télécharger un modèle ;
- aucun secret ou fichier local `.env` n’est ajouté au dépôt.

## 7. Priorité 1 — Corpus et provenance

Renforce la collecte et le stockage d’articles :

- conserver l’identifiant source stable (OpenAlex, DOI, arXiv ou ACL) et un identifiant interne stable ;
- dédupliquer en priorité par identifiant source/DOI, et gérer les doublons de titre comme cas de vérification, pas comme une règle aveugle ;
- valider les réponses réseau et les schémas ; ajouter timeout, retry avec backoff pour les erreurs transitoires, limites de débit et messages de diagnostic ;
- enregistrer la source, l’URL de page, le PDF s’il est légalement accessible, la licence connue ou inconnue, la date de collecte et les raisons de rejet ;
- rendre la collecte reprenable après interruption sans dupliquer les données ;
- ne pas prétendre que tous les textes intégraux sont disponibles quand seule la métadonnée ou l’abstract est accessible ;
- ajouter des commandes de statistiques du corpus (nombre d’articles, années, venues, langue, abstracts manquants, PDF disponibles, doublons).

Les premiers tests peuvent utiliser 20 à 100 articles de fixtures. Pour l’objectif de cours, prévoir une montée contrôlée vers 500–2 000 articles en accès ouvert. Ne télécharge pas ce volume à chaque lancement de test.

### Critères d’acceptation

- une collecte de test peut être exécutée, reprise et reproduite ;
- les données non valides sont rapportées et ne provoquent pas la perte silencieuse du reste de la collecte ;
- la provenance d’un article peut être retrouvée depuis tout claim ou toute phrase générée à partir de cet article.

## 8. Priorité 2 — Parsing et conservation des preuves

Améliore le pipeline `Article -> Section -> Chunk` :

- garder les identifiants d’article/section/chunk ainsi que les positions source ; les offsets doivent correspondre au texte enregistré ;
- conserver le type de section et distinguer abstract, introduction, method, experiments, results, discussion, limitations et references lorsque le texte le permet ;
- normaliser le texte sans effacer les valeurs numériques, noms de modèles, acronymes, négations, signes de pourcentage ou marqueurs d’incertitude ;
- exclure le contenu de la bibliographie des passages de preuve ordinaires, sauf tâche explicite portant sur les références ;
- séparer les faits extraits des inférences du système ;
- signaler explicitement les articles dont seul l’abstract est analysé ;
- tester les PDF mal formés, pages vides, URL invalides, fichiers trop volumineux et erreurs de téléchargement ;
- sauvegarder les sorties de façon sûre pour éviter une corruption en cas d’arrêt.

Ne promets pas une précision de parsing PDF parfaite. Ajoute des tests d’erreur et décris les limites.

## 9. Priorité 3 — Recherche et retrieval

Préserve et vérifie les modes `bm25`, `dense` et `rrf` existants.

- BM25 constitue la baseline lexicale.
- Dense retrieval peut utiliser le modèle scientifique/multilingue configuré ; le nom et la version du modèle doivent être traçables.
- RRF fusionne des rangs, non des scores bruts incompatibles.
- Le retrieval doit renvoyer les identifiants, rangs/scores, texte du chunk, article et section.
- Filtrer les doublons dans les résultats finaux ou, au minimum, éviter que trop de chunks d’un même article monopolisent les résultats.
- Les résultats doivent pouvoir être justifiés : pourquoi ce passage est-il présenté, de quel article provient-il ?
- Les modèles et index doivent avoir une version/configuration ou empreinte associée pour éviter d’utiliser silencieusement un index devenu incompatible.

Les requêtes en français ou en arabe sur des articles en anglais sont une exigence du produit à tester. Ne prétends pas qu’elles fonctionnent tant que le modèle choisi et le protocole ne l’ont pas démontré. Si la qualité est insuffisante, expose cette limite et propose une solution locale/multilingue sans API payante.

### Métriques

Prévoir une évaluation sur des requêtes et articles annotés manuellement : Recall@10 et MRR au minimum. Comparer BM25, dense et hybride sur le même jeu de test figé.

## 10. Priorité 4 — Extraction structurée et claims

Les règles/spaCy existantes restent la baseline Bronze. Renforce-les sans les présenter comme une extraction parfaite.

Chaque claim devrait porter au minimum :

- `id` stable ;
- `article_id`, `section_id` et `chunk_id` ;
- texte du claim ;
- extrait/verbatim source exact ;
- offsets de début et de fin dans le texte source lorsque disponibles ;
- type : finding, method, dataset, limitation, comparison ou hypothesis ;
- niveau de hedging/incertitude ;
- métrique, valeur, unité, dataset et modèle si réellement détectés ;
- méthode d’extraction et version du composant ;
- état de validation et score optionnel.

Exigences :

- conserver les négations, les conditions et le hedging (“may”, “might”, “suggests”, etc.) ;
- ne jamais convertir une hypothèse ou une phrase spéculative en résultat certain ;
- les résultats numériques doivent garder la métrique, l’unité/échelle, le dataset et le contexte si présents ;
- ne pas inventer un champ manquant ; utiliser `null`/“non rapporté” ;
- éviter de construire des phrases artificielles à partir d’extractions SVO manifestement incorrectes ;
- dédupliquer les claims quasi identiques de chunks avec chevauchement sans supprimer les différences de contexte ;
- ajouter une analyse d’erreurs pour les faux positifs, faux négatifs, résultats chiffrés, négations et hedging.

Le développement Silver (par exemple classifieur scientifique fine-tuné) ne doit être entrepris qu’après la constitution d’un jeu d’annotations acceptable. Si aucun modèle n’est entraîné, indique honnêtement que l’extraction utilisée est la baseline par règles.

## 11. Priorité 5 — Evidence linking et graphe de traçabilité

C’est le cœur du projet. Introduis ou complète une représentation explicite de l’évidence, réutilisant les schémas existants si possible.

Chaque lien claim-preuve doit contenir :

- l’identifiant du claim ;
- l’identifiant de l’article et du passage source ;
- le texte exact du passage source (ou une référence non ambiguë vers lui) ;
- la section et les offsets ;
- le type de relation : support, contradiction, contexte ou incertain ;
- le mode de création du lien : direct/extrait, retrieval ou inférence ;
- le score du vérificateur lorsqu’il existe ;
- la version du modèle et les seuils utilisés.

Une phrase de revue ne peut citer un article que si les identifiants de cet article existent dans le corpus/la liste de références associée et que le système sait quel claim ou passage justifie cette citation. Toute référence doit être navigable depuis l’interface jusqu’au passage d’origine.

**Ne confonds pas le graphe bibliographique** (un article cite un autre article selon les métadonnées) avec **une preuve textuelle** qu’un passage soutient une affirmation.

## 12. Priorité 6 — Comparaison des articles

Produis un tableau comparatif à partir de claims et métadonnées structurés :

- méthode/modèle ;
- tâche et question de recherche ;
- dataset ;
- métrique et valeur ;
- conditions expérimentales, si extraites ;
- limites déclarées ;
- source et claim IDs.

Règles impératives :

- comparer des résultats numériques uniquement lorsque la métrique, l’échelle et le contexte sont suffisamment comparables ;
- ne pas classer un modèle comme “meilleur” sur la base de nombres incomparables ;
- afficher “non rapporté” plutôt que combler les cellules par hallucination ;
- chaque ligne doit ouvrir l’article et la preuve associée ;
- signaler les comparaisons ambiguës ou contradictoires, sans les résoudre arbitrairement.

## 13. Priorité 7 — Génération de la revue

Garde le générateur de gabarits comme baseline déterministe. Ajoute une interface de génération optionnelle seulement si elle respecte les mêmes contrats de preuve.

La génération doit :

- écrire un paragraphe à partir de claims/evidences disponibles, pas directement à partir de la seule question utilisateur ;
- associer à chaque phrase un ou plusieurs `supporting_claim_ids` et des identifiants d’articles cités ;
- inclure des citations uniquement si les identifiants sont connus ;
- distinguer explicitement consensus, résultats divergents et limites ;
- ne pas fabriquer de taille d’effet, score, date ou détail expérimental absent des sources ;
- maintenir le hedging original ;
- éviter de présenter le résumé d’abstract comme une lecture des résultats du PDF complet ;
- fournir un brouillon même si l’évidence est limitée, mais indiquer les trous et s’abstenir des affirmations non soutenues ;
- versionner le prompt et la configuration pour la reproductibilité.

Si un LLM local ou externe est utilisé, impose une sortie structurée validée par Pydantic. Une sortie invalide doit être rejetée/réparée avec un nombre limité de tentatives, jamais acceptée silencieusement.

## 14. Priorité 8 — Vérification des citations et des affirmations

Vérifie chaque couple **phrase générée × citation** séparément.

Étapes :

1. vérifier que l’identifiant d’article correspond à une entrée source réelle et cohérente (titre, auteurs, année, DOI/URL) ;
2. vérifier que le claim lié appartient bien à l’article cité ;
3. retrouver le passage source exact ;
4. évaluer si le passage soutient la proposition de la phrase ;
5. détecter si la phrase extrapole au-delà du passage, inverse la négation, supprime une condition ou transforme un résultat limité en affirmation générale ;
6. stocker le score brut, le label, le modèle et les seuils ;
7. permettre un audit humain et marquer les cas incertains.

Utilise des statuts non ambigus, compatibles avec les modèles existants :
- `VERIFIED` : preuve suffisante de soutien, dans le protocole et les seuils validés ;
- `EXISTS_NO_SUPPORT` : référence existante, mais le passage ne soutient pas la phrase ;
- `CONTRADICTED` : le passage apporte un élément contradictoire ;
- `HALLUCINATED` : référence/identité citée est fabriquée ou ne peut être résolue après les vérifications prévues et la décision est justifiée ;
- `UNVERIFIABLE` : manque de texte, preuve insuffisante, accès externe impossible ou confiance insuffisante.

Si les enums existants doivent changer, prévoir compatibilité/migration, tests et mise à jour de tous les consommateurs. Ne change pas un contrat sans vérifier tous ses usages.

**Une sortie NLI est un signal automatique, pas une preuve absolue.** Valide les seuils sur un petit jeu annoté par des humains avant de présenter un taux de vérification comme résultat scientifique. Si le modèle ne peut pas charger, les phrases restent non vérifiables ; ne remplace pas l’échec par des valeurs artificielles.

## 15. Priorité 9 — API et interface

Préserve les routes existantes lorsqu’elles sont utiles : `/`, `/search`, `/articles/{article_id}`, `/generate`, `/review/{slug}`, `/export/bibtex` et `/claims`. Lis leur comportement actuel avant d’ajouter ou de changer un contrat.

Complète si nécessaire :
- une route de vérification/audit d’une revue ;
- une route pour consulter les claims et leurs passages sources ;
- une route de comparaison exportable ;
- un statut de corpus/index/modèles plus informatif ;
- une route d’export BibTeX avec clés uniques et métadonnées échappées correctement.

Dans l’interface `/ui`, permettre à l’utilisateur de :
1. soumettre une requête ;
2. voir les articles et le statut d’accès (abstract seulement / texte intégral) ;
3. ouvrir les claims et les passages source ;
4. lire le tableau comparatif ;
5. lire la revue phrase par phrase ;
6. ouvrir les détails de vérification et les scores ;
7. distinguer les états vérifié/non soutenu/contradictoire/non vérifiable ;
8. exporter BibTeX et sauvegarder/télécharger la revue.

Le frontend ne doit pas afficher un indicateur vert “verified” si seul l’article a été retrouvé. Utilise une légende explicite et accessible. Les erreurs réseau, corpus vide et modèle non disponible doivent être expliqués à l’utilisateur.

## 16. Priorité 10 — Évaluation expérimentale requise

Construis des évaluations reproductibles, pas seulement une démonstration visuelle.

### Jeux de données et annotation

Prévoir progressivement :
- un petit corpus de développement pour l’intégration ;
- un ensemble de requêtes/articles annotés pour le retrieval ;
- une annotation d’articles couvrant question, contribution, méthode, données, résultats et limites ;
- des citations/claims annotés par fonction et relation de soutien ;
- un sous-ensemble doublement annoté pour mesurer l’accord inter-annotateurs ;
- des revues de référence validées par l’enseignant lorsque disponibles.

Le cahier de cours cible 500–2 000 articles, 200 articles annotés, 500 citations annotées et 20 revues de référence validées. Traite ces nombres comme des **objectifs de livraison**, pas comme des données déjà existantes. Documente ce qui est réellement collecté/annoté.

### Protocoles

- établir un split train/dev/test et figer le test dès sa première définition ;
- ne jamais utiliser le test pour régler des hyperparamètres/seuils ;
- pour les composants appris, lancer trois graines fixes (par exemple 42, 123, 999) et présenter moyenne ± écart-type ;
- comparer les systèmes sur les mêmes requêtes et les mêmes exemples ;
- effectuer un bootstrap apparié pour le résultat principal lorsque l’échantillon s’y prête ;
- conserver les configurations et versions dans `experiments.md` ou un journal JSONL machine-lisible ;
- produire une analyse d’au moins 30 erreurs réelles, catégorisées, sans fabriquer d’exemples présentés comme erreurs observées.

### Métriques minimales

- Retrieval : Recall@10, MRR.
- Extraction de sections/claims/fonctions de citation : précision, rappel, Macro-F1 selon la tâche.
- Support de citation : taux de références existantes et soutenant réellement la phrase ; taux de soutien, taux de contradiction, taux non vérifiable et couverture.
- Review : exactitude/couverture évaluées par des humains à partir d’une grille ; accord entre évaluateurs lorsque possible.
- Robustesse : reformulations, négations, nombres, hedging, articles dont l’abstract et le résultat divergent.
- Biais de couverture : répartition des sources, années et venues.
- Latence et consommation mémoire documentées séparément de la qualité.

Ne donne jamais une métrique sans définir son numérateur, son dénominateur et le jeu d’évaluation concerné.

### Question de recherche principale

Évaluer :

**La vérification phrase par phrase des citations réduit-elle les citations non soutenues ou inventées, et à quel coût en couverture des affirmations ?**

Comparer au minimum :
- **Baseline A :** retrieval + génération sans vérification systématique ;
- **Système B :** retrieval + claims/evidence liés + génération contrainte + vérification phrase par phrase.

Mesurer le soutien effectif, les citations invalides/non soutenues, la proportion de phrases laissées non vérifiables, la couverture et la qualité humaine. Ne présume pas que le système B est meilleur avant d’avoir exécuté l’expérience.

## 17. Priorité 11 — Baseline Bronze, composants Silver et audit Gold

Livrer par paliers :

### Bronze — démo robuste de bout en bout
- collecte limitée et documentée ;
- parsing abstract + PDF lorsqu’il est disponible ;
- baseline BM25 et si possible dense/RRF ;
- extraction des claims par règles ;
- evidence linking explicite ;
- génération déterministe ;
- API/UI utilisable ;
- premiers tests et métriques ;
- au moins 30 erreurs d’exemples annotés avant le rapport lorsqu’elles sont disponibles.

### Silver — améliorer la recherche et l’extraction
- retrieval dense/hybride mesuré contre BM25 ;
- modèle d’extraction/classification scientifique uniquement avec données d’annotation ;
- questions FR/AR sur les articles EN évaluées séparément ;
- contrôle de la qualité de génération ;
- trois graines et ablations pour les composants appris.

### Gold — audit de fiabilité
- audit des références inventées/mal attribuées ;
- audit du soutien des claims ;
- vérification des incohérences entre abstract, résultats et limites ;
- analyse de couverture et de biais ;
- seuil NLI validé contre un ensemble annoté par humains ;
- sections Limitations quantifiées.

Ne sacrifie pas la complétude Bronze pour commencer prématurément un fine-tuning coûteux.

## 18. Tests à ajouter ou compléter

Les tests doivent utiliser un corpus synthétique/fixé de petite taille et des modèles stubbés.

Couvrir au minimum :

1. parsing de métadonnées et reconstruction des abstracts ;
2. collecte reprenable, déduplication et réponses réseau invalides ;
3. génération de sections/chunks avec identifiants et offsets déterministes ;
4. persistance/rechargement d’articles, chunks, claims, index et revues ;
5. extraction des valeurs, métriques, négations et hedging ;
6. liaison claim -> passage source -> article ;
7. retrieval BM25/dense/RRF, résultats vides et identifiants inconnus ;
8. génération sans claim pertinent (abstention ou couverture explicitement insuffisante) ;
9. attribution correcte de chaque citation à l’article qui fournit le passage ;
10. cas NLI de soutien, neutralité, contradiction, référence manquante et modèle indisponible ;
11. un article absent du corpus ne doit pas être automatiquement qualifié d’hallucination sans justification ;
12. export BibTeX : caractères spéciaux, métadonnées manquantes et clés dupliquées ;
13. validation des entrées API, erreurs HTTP, sécurité des noms de fichiers/slug et chemins de sortie ;
14. test end-to-end sans Internet à partir de fixtures ;
15. compatibilité des anciens schémas/fichiers JSON de revues si les modèles évoluent.

Les tests ne doivent pas dépendre d’un modèle téléchargé ni d’un serveur externe. Utilise des mocks et fixtures pour les tests rapides.

## 19. Documentation et livrables

Maintiens à jour :

- `README.md` : objectif, installation, configuration, commandes et limites ;
- un diagramme d’architecture ;
- `docs/DATA_CARD.md` : sources, couverture, licences, dates, biais, splits ;
- `docs/MODEL_CARD.md` ou une carte par composant appris ;
- `docs/ANNOTATION_GUIDE.md` : règles pour claims/citations/relations de soutien ;
- `docs/EVALUATION.md` : jeux, protocoles, métriques et limites ;
- `scilit/experiments.md` : journal d’expériences réellement effectuées ;
- instructions pour reconstruire le corpus et les index ;
- exemples d’API et exemple de revue avec liens vers les preuves ;
- limitations connues et fonctionnalités non encore réalisées.

Ne prétends pas qu’un modèle a été entraîné, qu’un résultat publié a été reproduit ou que les objectifs d’annotation sont atteints si aucune trace vérifiable ne l’atteste.

## 20. Sécurité, fiabilité et performance

- valider les paramètres d’API et limiter les valeurs de `k` ;
- ne pas incorporer directement un chemin fourni par l’utilisateur dans un chemin de fichier ;
- normaliser/valider les slugs et empêcher les path traversal ;
- limiter taille/timeout des téléchargements et valider les types de contenu avant parsing ;
- protéger les écritures concurrentes ou utiliser des écritures atomiques pour les fichiers d’index/revues ;
- éviter de charger des modèles lourds à chaque requête ; utiliser un cache paresseux sûr ;
- renvoyer des messages d’erreur explicites sans exposer les clés, variables sensibles ou stack traces à l’utilisateur ;
- documenter les coûts en mémoire/latence ; prévoir le mode CPU ;
- ne pas exiger un GPU pour l’exécution Bronze.

## 21. Plan d’exécution que tu dois suivre

Travaille dans cet ordre et garde le dépôt fonctionnel à chaque étape :

1. **Audit initial** : état des tests, inventaire des modules, bugs prioritaires et plan bref.
2. **Stabilité Bronze** : corriger les régressions, configurations, erreurs de chemins et contrats de schéma.
3. **Traçabilité** : garantir les identifiants et offsets de la source jusqu’au claim et à la phrase.
4. **Vérification fiable** : statuts distincts, attribution par citation, preuves exactes, gestion de l’absence de modèle.
5. **API/UI** : rendre visibles preuves, limites et statuts sans faux labels “verified”.
6. **Évaluation** : jeu test figé, métriques, comparaison avant/après vérification, scripts reproductibles.
7. **Silver/Gold** : ne passer aux modèles appris et audits plus avancés qu’une fois la baseline mesurable.
8. **Documentation finale** : commandes testées, limitations réelles, résultats disponibles et tâches restantes.

Pour chaque étape :
- modifier le code ;
- ajouter/mettre à jour les tests ;
- exécuter les tests pertinents ;
- expliquer les fichiers modifiés et les résultats ;
- noter les échecs restants et la prochaine priorité.

N’arrête pas ton travail après avoir seulement présenté un plan. Implémente au moins une tranche fonctionnelle de bout en bout dans le contexte de travail disponible. Si le scope est trop grand pour une seule passe, livre la tranche la plus prioritaire, laisse le dépôt cohérent et indique précisément les prochaines étapes.

## 22. Définition de “terminé”

Le projet n’est acceptable que si :

- l’installation et les commandes principales sont documentées et reproductibles ;
- l’utilisateur peut effectuer une recherche, obtenir des articles/passages et générer un brouillon ;
- chaque phrase déclarée vérifiée dispose d’un article et d’un passage source récupérables ;
- la vérification distingue soutien, absence de soutien, contradiction et impossibilité de vérifier ;
- l’absence de preuves déclenche l’abstention ou un état explicite, jamais une affirmation fabriquée ;
- BibTeX et les liens de provenance fonctionnent ;
- les tests automatisés couvrent le pipeline et les cas d’échec ;
- les métriques sont calculées sur des données identifiées, avec protocole et limitations ;
- la démonstration ne dépend pas d’une API payante ;
- README, cartes de données/modèles et journal d’expériences reflètent honnêtement ce qui a été réalisé ;
- aucun résultat de test, annotation humaine ou référence n’a été inventé.

## 23. Exigences obligatoires du module NLP 2026-2027 — matrice de conformité

Cette annexe transforme la fiche pédagogique du projet **P6 — SciLit** en contraintes vérifiables. Ces objectifs sont des exigences à atteindre, et non des affirmations selon lesquelles les données, annotations ou expériences existent déjà. Commence par mesurer l'état réel, puis consigne l'écart.

### 23.1 Périmètre exact de SciLit

Le parcours de bout en bout doit couvrir :

1. Recherche d'articles pertinents, en priorité dans les sources ouvertes ACL Anthology et arXiv, sur une requête de recherche.
2. Affichage clair des métadonnées et de la disponibilité du contenu : abstract seul, texte intégral analysé, ou accès/parse indisponible.
3. Extraction structurée de la question de recherche, contribution, méthode, données, résultats et limites, avec provenance et extrait source.
4. Classification de la fonction des citations et construction d'un graphe bibliographique lorsque les données sources le permettent. Ne présente jamais ce graphe comme une preuve textuelle de soutien.
5. Tableau comparatif traçable des études et de leurs critères.
6. État de l'art rédigé phrase par phrase, avec références vérifiées et accès au claim puis au passage justificatif.
7. Questions posées en français ou en arabe sur des articles en anglais. Cette capacité doit être mesurée séparément par langue et ne doit pas être déclarée opérationnelle sur la seule base de l'existence d'un champ de langue.
8. Application, API et export BibTeX fonctionnels.

Le corpus cible du sujet est de **500 à 2 000 articles en accès ouvert**. Toute indisponibilité du texte intégral, licence inconnue ou extraction partielle doit être visible dans les résultats et les cartes de données.

### 23.2 Contraintes académiques, modèles et assistance IA

- Le système évalué ne doit dépendre d'aucune API payante. Une clé ou un service externe peut rester une intégration facultative, désactivée par défaut, sans casser le parcours Bronze.
- Respecter les plafonds du module : modèles entraînés jusqu'à **1 milliard de paramètres** et modèles jusqu'à **8 milliards de paramètres quantifiés** pour l'inférence. Consigner taille, précision/quantification, source, licence et empreinte/version de chaque modèle.
- Aucun test unitaire ne doit exiger Internet, une clé API ou le téléchargement d'un modèle. Utiliser des mocks et des fixtures reproductibles.
- L'assistance IA pour le code, la relecture ou la reformulation doit être déclarée dans le rapport. Elle ne doit jamais servir à fabriquer des résultats, des références non lues ou des annotations présentées comme humaines.
- Respecter la licence et les conditions de redistribution de chaque corpus. Ne publier que les éléments partageables et documenter les exclusions.

### 23.3 Les cinq lots et leurs livrables

Organise les tickets, les tests et la documentation selon les lots communs suivants. Dans un projet individuel ou une petite équipe, ces lots restent des axes de responsabilité et d'audit.

| Lot | Périmètre SciLit | Livrables minimum |
|---|---|---|
| A — Données & langue | Collecte, provenance, parsing, normalisation, couverture linguistique, corpus et splits | Carte de données, guide de collecte/annotation, manifeste reproductible, statistiques et limites de couverture |
| B — Analyse linguistique & sens | Sections, syntaxe utile, sémantique, hedging, ontologie des méthodes, coréférence si implémentée | Baselines par règles, exemples annotés, métriques d'extraction et analyse d'erreurs |
| C — Modèles appris | Représentations, retrieval dense, extraction/classification apprise, transfert | Carte de modèle, expériences reproductibles, comparaison baseline classique/neuronale, trois graines si composant appris |
| D — Génération & recherche | Retrieval, RAG, claims, evidence linking, revue et citations | Métriques de retrieval, tableau comparatif, état de l'art lié aux preuves, export BibTeX |
| E — Évaluation, audit & intégration | Tests, API/UI, audit des citations, robustesse, biais, rapport | Tableau de bord ou rapport d'évaluation, audit, tests end-to-end et démonstration intégrée |

Ne coche un livrable comme terminé que si un fichier, une commande, un test ou une preuve de démonstration le confirme.

### 23.4 Protocole expérimental commun obligatoire

Applique ces règles à SciLit ; explique explicitement toute impossibilité au lieu de simuler la conformité.

- Geler le jeu de test lors de sa première constitution et sauvegarder les identifiants, critères d'inclusion, version et empreinte. Ne jamais l'utiliser pour choisir un hyperparamètre ou un seuil.
- Pour chaque composant appris, comparer au moins une baseline classique (règles, TF-IDF, BM25 ou FastText + modèle linéaire selon la tâche) et une baseline neuronale pertinente. Pour le retrieval, BM25 est la baseline lexicale à conserver.
- Pour tout composant appris, réaliser trois exécutions avec graines fixées et publier moyenne ± écart-type. Les scores doivent être calculés sur les mêmes exemples.
- Utiliser un bootstrap apparié contre la baseline pour le résultat principal lorsque le volume de test le permet ; documenter la méthode, le nombre de réplications et l'intervalle rapporté.
- Réaliser au moins une ablation par lot : retirer ou varier un composant identifiable (par exemple normalisation, type d'index, fusion, evidence linking ou contrôle de citation).
- Fournir une analyse d'au moins 30 erreurs catégorisées par composant appris, à partir de sorties réellement observées. Les exemples synthétiques doivent être identifiés comme tels et ne remplacent pas des erreurs mesurées.
- Rapporter séparément les résultats par langue et par script pour toute fonctionnalité multilingue, notamment les requêtes FR/AR sur des articles EN.
- Reproduire un résultat publié avant la revue de mi-parcours : choisir un résultat réalisable, préciser la source, les données, les différences de protocole et les résultats effectivement obtenus.
- Produire une carte de données et une carte de modèle pour chaque corpus/composant concerné.
- Inclure une section « Limitations » couvrant cinq dimensions explicitement nommées et chiffrées lorsque c'est possible. La fiche du sujet ne nomme pas ces cinq dimensions : ne prétends pas qu'une liste arbitraire vient de l'enseignant ; vérifie la grille du module ou définis et justifie clairement les dimensions retenues.

### 23.5 Cibles d'annotation pour la version académique

Les chiffres ci-dessous sont les objectifs du sujet, pas des statistiques acquises du dépôt :

- 200 articles annotés selon la grille : question, contribution, méthode, résultats et limites ;
- 500 citations annotées selon leur fonction ;
- 20 états de l'art de référence rédigés par les étudiants et validés par l'enseignant ;
- accord inter-annotateurs mesuré sur un sous-ensemble doublement annoté, avec guide d'annotation et kappa adapté à la tâche.

Si ces cibles ne sont pas atteintes, publier l'état réel, les écarts et le plan de réduction du risque. N'invente ni annotation, ni validation enseignante, ni valeur de kappa. Documenter les annotateurs, la procédure d'adjudication, les catégories, les désaccords et la version du dataset. Éviter les fuites entre train/dev/test au niveau article, doublon et quasi-doublon.

### 23.6 Métriques minimales et question de recherche

Les scripts et rapports d'évaluation doivent définir chaque métrique, son numérateur/dénominateur et le jeu précis auquel elle s'applique.

- Retrieval : Recall@10 et MRR au minimum, avec comparaison BM25, dense et hybride sur un test identique.
- Sections, claims et fonction de citation : précision/rappel et macro-F1 adaptés aux labels ; rapporter les erreurs de chaque classe.
- Soutien des citations : séparer existence de la référence, attribution au bon article, soutien textuel, contradiction, non-vérifiabilité et couverture. Ne pas publier un taux « 0 % hallucination » lorsque le vérificateur n'a pas évalué les éléments.
- Revue de littérature : grille humaine portant au minimum sur l'exactitude et la couverture, avec accord entre évaluateurs lorsqu'il est disponible.
- Robustesse : tester reformulations, négations, nombres, hedging, données manquantes et divergences abstract/résultats.
- Biais de couverture : distribution des sources, années, venues et articles effectivement accessibles.
- Documenter latence et mémoire séparément de la qualité.

Question de recherche recommandée, conforme au sujet : **la vérification des citations phrase par phrase réduit-elle les affirmations non soutenues ou les références inventées, et à quel coût en couverture ?** Comparer une baseline sans vérification systématique à la chaîne avec claims/preuves liés et audit par citation. Ne présuppose pas le résultat avant l'expérience.

### 23.7 Jalons et critères de sortie

- **Bronze — W7 :** pipeline de bout en bout opérationnel avec composants classiques, BM25, génération déterministe, tests de base, métriques par composant et démonstration.
- **Silver — W10 :** retrieval dense/hybride et composants appris pertinents, questions FR/AR sur articles EN évaluées, trois graines, ablations et reproduction d'un résultat publié.
- **Gold — W12 :** audit des références inventées/mal attribuées, couverture, biais, robustesse, contradictions, abstention et seuils de vérification validés par rapport à des annotations humaines.
- **Platinum — W14 :** ressource publiable (jeu annoté, suite de tests ou charte), calibration justifiée, étude d'utilité réalisable et rapport au niveau d'un atelier, sans revendiquer de validation absente.

À chaque jalon, la sortie obligatoire est une démo sur des entrées non préparées, un état des métriques réellement disponibles, les échecs connus et les prochaines priorités.

### 23.8 Livrables finaux et barrière de validation

Avant de déclarer SciLit prêt pour la soutenance, vérifier au minimum : application + API + export BibTeX ; corpus et licences documentés ; guide d'annotation, cartes de données/modèles et manifeste reproductible ; baseline classique et résultats des composants appris réellement exécutés ; tableau comparatif et revue traçables ; protocole et test gelé ; trois graines, ablations et bootstrap lorsque requis ; analyse d'erreurs ; journal `experiments.md` ; section Limitations ; rapport de 12–15 pages et vidéo de démonstration.

Le barème du sujet doit guider la priorité : problématique/carte du système (3 points), données (3), composants classiques (4), composants appris et génératifs (5), multilingue/audit (4), intégration/démonstration (3), reproductibilité/rédaction (3). La soutenance valorise la démo de bout en bout, les résultats honnêtes, l'audit et la capacité à expliquer les limites. La conformité au barème ne se remplace pas par une liste de fonctionnalités annoncées.

## 24. Rapport final que tu dois fournir

À la fin, présente un rapport bref et factuel avec :

1. fonctionnalités terminées ;
2. fichiers principaux modifiés ;
3. commandes lancées et résultat de chacune ;
4. tests réussis/échoués/ignorés et pourquoi ;
5. flux utilisateur réellement fonctionnel ;
6. métriques réellement mesurées avec leurs jeux de données ;
7. limites restantes et risques ;
8. commandes exactes pour lancer SciLit depuis un environnement propre ;
9. prochaines tâches classées par priorité.

**Règle de conclusion : affirme uniquement ce que le dépôt et les tests permettent de démontrer.**
