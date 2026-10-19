# SciLit — Journal d'expériences

## exp_001 — Setup initial
- **Date** : 2026-10-08
- **Composant** : infrastructure
- **Hypothèse** : la structure du projet est correcte
- **Config** : Python 3.12, pydantic 2.x
- **Résultat** : schemas.py validé, tous les tests passent
- **Notes** : corpus et annotations à construire

---
# Format pour chaque expérience :
# exp_NNN — nom
# Date / Composant / Hypothèse / Config
# Résultat : metric ± std (3 seeds)
# Ablation / Analyse d'erreurs
# Reproduit depuis : (référence publiée)copy

## Évaluation retrieval gelée

Le benchmark scientifique de retrieval attend `data/evaluation/frozen_relevance_test.jsonl`.
Chaque ligne contient `query_id`, `query` et `relevant_chunk_ids` annotés par des
humains. Les mêmes requêtes et jugements servent à BM25, dense et RRF. Les métriques
rapportées sont Recall@10 et MRR ; le SHA-256 du jeu de test est enregistré avec
l'expérience. Les observations par requête sont conservées séparément des graines
aléatoires utilisées pour les modèles appris.

Tant que le jeu annoté n'existe pas, aucune comparaison officielle de retrieval,
aucune ablation statistique et aucune conclusion scientifique ne sont publiées.
Les tests fondés sur le chevauchement de mots-clés sont uniquement des smoke tests.
Les sorties NLI automatiques ne remplacent pas un audit humain des citations.
