"""
SciLit — Schémas Pydantic v2
═══════════════════════════════════════════════════════════════════════════════
Pipeline couvert :
  Article → Section → Chunk → Claim / Citation → CitationGraph
  Query → BM25 + Dense → RRF → Reranker → ComparativeTable
  ComparativeTable + LLM → ReviewDraft (phrases vérifiées) → BibTeX

Sections du fichier :
  1. Enums / vocabulaires contrôlés
  2. Corpus          (Article, Section, Chunk)
  3. Extraction      (Claim, InTextCitation, CitationGraph)
  4. Retrieval       (Query, BM25Result, DenseResult, RankedResult)
  5. Génération      (ComparativeTable, VerifiedSentence, ReviewDraft, BibTeX)
  6. Évaluation      (ComponentMetrics, ExperimentRecord, AblationResult,
                      HallucinationAudit)
  7. Annotation      (section, citation, grille article, SoTA de référence,
                      kappa inter-annotateurs)
  8. Cartes          (DataCard, ModelCard)

Contraintes du module rappelées dans les docstrings :
  § 2.3.1  Test gelé à la première exécution → SplitType.TEST
  § 2.3.3  3 graines, moyenne ± écart-type → ComponentMetrics.seeds (validé)
  § 2.3.4  1 ablation par lot → AblationResult
  § 2.3.6  Résultat publié reproduit → ExperimentRecord.reproduced_from
  § 2.3.7  Carte de données + carte de modèle → DataCard, ModelCard
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import statistics
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENUMS / VOCABULAIRES CONTRÔLÉS
# ═══════════════════════════════════════════════════════════════════════════


class Language(str, Enum):
    EN = "en"
    FR = "fr"
    # Extension prévue : AR = "ar", TN = "tn"


class Venue(str, Enum):
    ACL = "acl"
    EMNLP = "emnlp"
    NAACL = "naacl"
    COLING = "coling"
    ICLR = "iclr"
    NEURIPS = "neurips"
    ICML = "icml"
    ARXIV = "arxiv"
    OTHER = "other"


class SectionType(str, Enum):
    """
    Types de sections académiques détectés par SciBERT (Lot C, Ch7).
    Le classifieur prédit predicted_type ; la vérité terrain est
    section_type dans SectionAnnotation.
    """
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    LIMITATIONS = "limitations"
    ACKNOWLEDGEMENTS = "acknowledgements"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"


class ClaimType(str, Enum):
    """
    Taxonomie des affirmations extraites (Lot B règles + Lot C encodeur).
    Chaque type a des patrons syntaxiques distincts (Ch3-Ch4).
    """
    FINDING = "finding"          # résultat empirique chiffré
    METHOD = "method"            # contribution méthodologique
    DATASET = "dataset"          # ressource publiée
    LIMITATION = "limitation"    # limite reconnue par les auteurs
    COMPARISON = "comparison"    # comparaison directe entre systèmes/modèles
    HYPOTHESIS = "hypothesis"    # affirmation non encore prouvée dans l'article


class HedgingLevel(str, Enum):
    """
    Niveau d'incertitude linguistique — pragmatique (Ch4).
    Utilisé pour filtrer les affirmations trop spéculatives dans le SoTA.
    """
    CERTAIN = "certain"          # "we show", "results demonstrate"
    HEDGED = "hedged"            # "we suggest", "may indicate"
    SPECULATIVE = "speculative"  # "it is possible", "future work could"


class CitationFunction(str, Enum):
    """
    Fonction rhétorique de la citation (Ch4 + Lot C classifieur).
    Schème de Teufel et al. simplifié pour SciLit.
    """
    BACKGROUND = "background"    # fournit le contexte ou définit le problème
    USE = "use"                  # utilise directement la méthode ou les données
    COMPARE = "compare"          # compare les résultats numériquement
    CONTRAST = "contrast"        # résultats opposés ou contradictoires
    EXTEND = "extend"            # construit sur le travail, l'améliore
    FUTURE = "future"            # piste future proposée par les auteurs
    UNDEFINED = "undefined"      # non classifiable


class VerificationStatus(str, Enum):
    """
    Statut après vérification NLI de chaque référence dans le SoTA généré.
    Métrique principale du projet : taux VERIFIED / total.
    """
    VERIFIED = "verified"                   # article existe ET soutient la phrase
    EXISTS_NO_SUPPORT = "exists_no_support"  # article existe mais ne soutient pas
    HALLUCINATED = "hallucinated"            # article introuvable dans le corpus
    UNVERIFIABLE = "unverifiable"            # hors corpus ou accès impossible


class SplitType(str, Enum):
    """
    § 2.3.1 : le jeu TEST est gelé à la première exécution.
    Il n'est jamais utilisé pour choisir des hyperparamètres.
    """
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"   # ← gelé, lecture seule après la première exécution


# ═══════════════════════════════════════════════════════════════════════════
# 2. CORPUS
# ═══════════════════════════════════════════════════════════════════════════


class ArticleSource(BaseModel):
    """
    Provenance, accès et licence d'un article.
    Lot A collecte ACL Anthology (acl_id) et arXiv (arxiv_id).
    Au moins un des deux identifiants est obligatoire.
    """
    venue: Venue
    acl_id: str | None = None       # ex. "2023.acl-long.42"
    arxiv_id: str | None = None     # ex. "2304.01234"
    url: HttpUrl
    pdf_url: HttpUrl | None = None
    license: str                    # "CC BY 4.0", "ACL anthology open", etc.
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def at_least_one_id(self) -> ArticleSource:
        if self.acl_id is None and self.arxiv_id is None:
            raise ValueError("acl_id ou arxiv_id requis pour la traçabilité")
        return self


class Article(BaseModel):
    """
    Unité de base du corpus (500–2000 articles, accès ouvert).
    Produit par Lot A après parsing GROBID ou PyMuPDF.

    corpus_id est un hash stable utilisé dans experiments.md pour
    référencer sans ambiguïté la version du corpus (§ 2.3.6).
    """
    id: UUID = Field(default_factory=uuid4)
    title: str
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1990, le=2030)
    abstract: str
    language: Language = Language.EN
    source: ArticleSource
    keywords: list[str] = Field(default_factory=list)
    raw_text: str | None = None     # texte brut complet (conservé pour audit)
    split: SplitType = SplitType.TRAIN

    @computed_field
    @property
    def corpus_id(self) -> str:
        """Identifiant court reproductible pour experiments.md."""
        base = self.source.acl_id or self.source.arxiv_id or str(self.id)
        return hashlib.md5(base.encode()).hexdigest()[:8]


class Section(BaseModel):
    """
    Segment structurel d'un article.
    - Lot A : détection initiale par GROBID ou règles heuristiques.
    - Lot C : SciBERT fine-tuné prédit predicted_type (Ch7).
    Les offsets char_start / char_end permettent de relier le chunk au
    verbatim source pour la vérification (§ VerifiedSentence.verbatim).
    """
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    section_type: SectionType           # vérité terrain annotée ou GROBID
    title: str | None = None            # titre du paragraphe si présent
    text: str
    char_start: int                     # offset dans Article.raw_text
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    section_index: int                  # ordre dans l'article (0-based)
    # Champs prédits par SciBERT
    predicted_type: SectionType | None = None
    classifier_confidence: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("char_end")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        start = info.data.get("char_start", 0)
        if v <= start:
            raise ValueError("char_end doit être > char_start")
        return v


class Chunk(BaseModel):
    """
    Sous-unité d'indexation et de retrieval, consciente de la section.
    Lot A : chunking avec overlap (taille et overlap configurables).
    L'embedding (SPECTER2 ou BGE-M3) est exclu de la sérialisation JSON
    mais stocké en base vectorielle (FAISS ou ChromaDB).
    """
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    section_id: UUID
    section_type: SectionType           # dénormalisé pour le retrieval rapide
    text: str
    char_start: int                     # offset dans Section.text
    char_end: int
    chunk_index: int                    # ordre dans la section (0-based)
    token_count: int | None = None
    # Exclu de la sérialisation JSON — persisté séparément en base vectorielle
    embedding: list[float] | None = Field(None, exclude=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════


class Claim(BaseModel):
    """
    Affirmation extraite d'un chunk.

    Pipeline d'extraction (trois niveaux, ablation Lot B/C) :
      Bronze : patrons syntaxiques SVO (Ch3-Ch4) → text, subject/predicate/object_
      Silver : encodeur fine-tuné SciBERT → claim_type, hedging_level, confidence
      Gold   : LLM à sortie structurée (Ch8) → metric_name/value, dataset_name

    verbatim : texte source exact conservé pour la vérification NLI aval.
    Les champs metric_value / dataset_name / model_name alimentent ComparativeRow.
    """
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    chunk_id: UUID
    section_type: SectionType

    # Texte normalisé (pour la déduplication et la comparaison)
    text: str
    # Texte source verbatim (pour NLI et audit d'hallucination)
    verbatim: str

    claim_type: ClaimType
    hedging_level: HedgingLevel

    # Analyse syntaxique SVO (Ch3 — Lot B)
    subject: str | None = None
    predicate: str | None = None
    object_: str | None = Field(None, alias="object")

    # Valeurs structurées (Ch4 / LLM — Lot D)
    metric_name: str | None = None      # ex. "BLEU", "F1 macro", "accuracy"
    metric_value: float | None = None
    dataset_name: str | None = None     # ex. "SQuAD 2.0", "FEVER"
    model_name: str | None = None       # ex. "BERT-base-uncased"

    # Scores de confiance pour l'ablation
    extraction_confidence: float | None = Field(None, ge=0.0, le=1.0)

    # Embedding exclu de la sérialisation (FAISS)
    embedding: list[float] | None = Field(None, exclude=True)

    class Config:
        populate_by_name = True


class InTextCitation(BaseModel):
    """
    Citation in-text repérée dans un article (Lot B — règles de dépendances).
    target_article_id est None si l'article cité est hors corpus.
    function est prédit par le classifieur de Lot C (Ch7, § 2.3.3).
    """
    id: UUID = Field(default_factory=uuid4)
    source_article_id: UUID             # article qui cite
    target_article_id: UUID | None = None   # article cité (None = hors corpus)
    target_raw_ref: str                 # ex. "[Smith et al., 2022]"
    context_sentence: str               # phrase contenant la citation
    function: CitationFunction = CitationFunction.UNDEFINED
    function_confidence: float | None = Field(None, ge=0.0, le=1.0)
    chunk_id: UUID | None = None        # chunk source


class CitationGraphEdge(BaseModel):
    """Arête orientée du graphe de citations (Lot A + Lot B)."""
    source_id: UUID                     # article citant
    target_id: UUID                     # article cité
    function: CitationFunction
    count: int = Field(default=1, ge=1) # nb de mentions dans le même article


class CitationGraph(BaseModel):
    """
    Graphe de citations du corpus complet.
    Construit après extraction de toutes les InTextCitation.
    density mesure la connectivité du corpus.
    """
    corpus_snapshot_date: datetime
    nodes: list[UUID]                   # tous les article IDs du corpus
    edges: list[CitationGraphEdge]

    @computed_field
    @property
    def density(self) -> float:
        """Densité = |arêtes| / (|nœuds| × (|nœuds| - 1))."""
        n = len(self.nodes)
        if n < 2:
            return 0.0
        return len(self.edges) / (n * (n - 1))


# ═══════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════


class RetrievalQuery(BaseModel):
    """
    Requête utilisateur, langue détectée, type inféré.
    benchmark_id pointe vers une des 20 requêtes du benchmark gelé (§ 2.3.1).
    """
    id: UUID = Field(default_factory=uuid4)
    text: str
    language: Language = Language.EN
    query_type: str = "free_text"   # "comparison", "method", "dataset", "survey"
    benchmark_id: str | None = None # ex. "bq_015" — requête de benchmark gelée


class BM25Result(BaseModel):
    """Résultat lexical BM25 (baseline Bronze — Lot A/C)."""
    chunk_id: UUID
    article_id: UUID
    score: float
    rank: int = Field(ge=1)


class DenseResult(BaseModel):
    """Résultat de recherche dense (encodeur bi-encoder — Lot C Silver)."""
    chunk_id: UUID
    article_id: UUID
    cosine_score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)


class RankedResult(BaseModel):
    """
    Résultat final après fusion RRF et reranking cross-encoder.
    is_relevant sert à calculer recall@10 et MRR sur le benchmark gelé.

    Ablation clé (Lot C/D) :
      - rrf_score sans reranker → comparer avec reranker_score activé
    """
    query_id: UUID
    chunk_id: UUID
    article_id: UUID
    section_type: SectionType
    chunk_text: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None     # cross-encoder (Silver)
    final_rank: int = Field(ge=1)
    is_relevant: bool | None = None         # None = non annoté


# ═══════════════════════════════════════════════════════════════════════════
# 5. GÉNÉRATION
# ═══════════════════════════════════════════════════════════════════════════


class ComparativeRow(BaseModel):
    """
    Ligne du tableau comparatif généré par LLM à sortie structurée (Lot D).
    source_claim_id permet de remonter au verbatim source pour l'audit.
    verbatim_snippet ≤ 15 mots : respecte la règle de citation du module.
    """
    model_or_method: str
    dataset: str | None = None
    metric_name: str
    metric_value: str               # str : supporte "92.3 ± 0.4" ou "—"
    source_claim_id: UUID
    source_article_id: UUID
    verbatim_snippet: str           # extrait source ≤ 15 mots (traçabilité)

    @field_validator("verbatim_snippet")
    @classmethod
    def snippet_length(cls, v: str) -> str:
        words = v.split()
        if len(words) > 20:
            raise ValueError(
                f"verbatim_snippet trop long ({len(words)} mots) — risque copyright"
            )
        return v


class ComparativeTable(BaseModel):
    """Tableau comparatif complet pour une requête."""
    query_id: UUID
    rows: list[ComparativeRow]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_model: str


class VerifiedSentence(BaseModel):
    """
    Phrase du SoTA généré avec ses références vérifiées phrase par phrase.

    C'est la brique de la métrique principale du projet :
      « taux de références existantes ET soutenantes »
    Calculé via NLI (Ch7) entre la phrase et le verbatim source du claim.

    hallucination_rate = nb HALLUCINATED / total citations de cette phrase.
    """
    text: str
    citations: list[UUID]                            # IDs des articles cités
    verification_statuses: list[VerificationStatus]  # 1 statut par citation
    supporting_claim_ids: list[UUID]                 # claims qui soutiennent la phrase
    nli_score: float | None = Field(None, ge=0.0, le=1.0)  # entailment NLI

    @model_validator(mode="after")
    def lengths_match(self) -> VerifiedSentence:
        if len(self.citations) != len(self.verification_statuses):
            raise ValueError(
                "citations et verification_statuses doivent avoir la même longueur"
            )
        return self

    @computed_field
    @property
    def hallucination_rate(self) -> float:
        if not self.verification_statuses:
            return 0.0
        return sum(
            1 for s in self.verification_statuses
            if s == VerificationStatus.HALLUCINATED
        ) / len(self.verification_statuses)

    @computed_field
    @property
    def support_rate(self) -> float:
        if not self.verification_statuses:
            return 0.0
        return sum(
            1 for s in self.verification_statuses
            if s == VerificationStatus.VERIFIED
        ) / len(self.verification_statuses)


class ReviewDraft(BaseModel):
    """
    Revue de littérature générée avec traçabilité complète sentence-level.

    overall_hallucination_rate agrège toutes les VerifiedSentence :
      = nb_HALLUCINATED / nb_total_citations dans tout le draft.
    C'est la valeur rapportée dans les tableaux du rapport.

    prompt_version permet de rejouer exactement la même génération
    (reproductibilité § 2.3.1 / § 4 rapport final).
    """
    id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    title: str
    sentences: list[VerifiedSentence]
    language: Language = Language.EN
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_model: str                  # ex. "mistral-7b-instruct-q4_K_M"
    prompt_version: str             # ex. "v3.2" — lié à experiments.md

    @computed_field
    @property
    def overall_hallucination_rate(self) -> float:
        all_statuses = [
            s
            for sent in self.sentences
            for s in sent.verification_statuses
        ]
        if not all_statuses:
            return 0.0
        return sum(
            1 for s in all_statuses if s == VerificationStatus.HALLUCINATED
        ) / len(all_statuses)

    @computed_field
    @property
    def overall_support_rate(self) -> float:
        all_statuses = [
            s
            for sent in self.sentences
            for s in sent.verification_statuses
        ]
        if not all_statuses:
            return 0.0
        return sum(
            1 for s in all_statuses if s == VerificationStatus.VERIFIED
        ) / len(all_statuses)


class BibTeXEntry(BaseModel):
    """
    Entrée BibTeX exportable pour un article du corpus.
    Utilisé par l'endpoint /export/bibtex de l'API FastAPI.
    """
    article_id: UUID
    bibtex_key: str                 # ex. "Smith2023LoRA"
    entry_type: str = "inproceedings"
    title: str
    authors: list[str]
    year: int
    venue: str
    url: str | None = None

    def to_bibtex(self) -> str:
        authors_str = " and ".join(self.authors)
        lines = [
            f"@{self.entry_type}{{{self.bibtex_key},",
            f"  title     = {{{self.title}}},",
            f"  author    = {{{authors_str}}},",
            f"  year      = {{{self.year}}},",
            f"  booktitle = {{{self.venue}}},",
        ]
        if self.url:
            lines.append(f"  url       = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ÉVALUATION  (experiments.md + métriques)
# ═══════════════════════════════════════════════════════════════════════════


class SeedResult(BaseModel):
    """Résultat pour une graine unique d'un composant."""
    seed: int
    metric_name: str
    value: float


class ComponentMetrics(BaseModel):
    """
    Métriques pour un composant avec exactement 3 graines.
    § 2.3.3 : tableaux en moyenne ± écart-type.
    Le validateur refuse < 3 ou > 3 graines pour éviter les oublis.

    Exemple d'usage :
        metrics = ComponentMetrics(
            component_name="citation_function_classifier",
            metric_name="macro_F1",
            seeds=[
                SeedResult(seed=42,  metric_name="macro_F1", value=0.831),
                SeedResult(seed=123, metric_name="macro_F1", value=0.844),
                SeedResult(seed=999, metric_name="macro_F1", value=0.827),
            ]
        )
        print(f"{metrics.mean:.3f} ± {metrics.std:.3f}")
    """
    component_name: str
    metric_name: str
    seeds: list[SeedResult]

    @field_validator("seeds")
    @classmethod
    def exactly_three_seeds(cls, v: list[SeedResult]) -> list[SeedResult]:
        """§ 2.3.3 : exactement 3 graines, pas plus, pas moins."""
        if len(v) != 3:
            raise ValueError(
                f"§ 2.3.3 exige exactement 3 graines — {len(v)} fournies."
            )
        return v

    @computed_field
    @property
    def mean(self) -> float:
        return statistics.mean(r.value for r in self.seeds)

    @computed_field
    @property
    def std(self) -> float:
        return statistics.stdev(r.value for r in self.seeds)

    def summary(self) -> str:
        return f"{self.component_name} | {self.metric_name}: {self.mean:.3f} ± {self.std:.3f}"


class AblationResult(BaseModel):
    """
    Résultat d'une ablation (§ 2.3.4 : 1 ablation minimale par lot).
    delta = ablated_score − full_system_score (négatif ⟹ la variante dégrade).

    Exemples d'ablations SciLit :
      Lot A : suppression de la normalisation des abréviations
      Lot B : suppression du filtre hedging_level SPECULATIVE
      Lot C : BM25 seul vs hybride (RRF)
      Lot D : sans reranker cross-encoder
    """
    component: str
    ablated_element: str            # ex. "reranker cross-encoder retiré"
    full_system_metric: float
    ablated_metric: float
    metric_name: str
    split: SplitType = SplitType.TEST

    @computed_field
    @property
    def delta(self) -> float:
        return self.ablated_metric - self.full_system_metric


class ExperimentRecord(BaseModel):
    """
    Enregistrement d'expérience pour experiments.md (§ 2.1 obligatoire).

    À sauvegarder en JSONL dans experiments.md à chaque run.
    reproduced_from documente le résultat publié reproduit (§ 2.3.6).
    error_analysis_count doit atteindre ≥ 30 pour le rapport (§ 2.3.4).
    """
    experiment_id: str              # ex. "exp_042_scibert_sections"
    date: datetime = Field(default_factory=datetime.utcnow)
    component: str
    hypothesis: str                 # "Pourquoi cette expérience ?"
    config: dict[str, Any]          # lr, batch_size, max_seq_len, ...
    dataset_split: SplitType
    results: list[ComponentMetrics]
    ablations: list[AblationResult] = Field(default_factory=list)
    error_analysis_count: int = Field(default=0, ge=0)  # catégoriser ≥ 30
    notes: str = ""
    reproduced_from: str | None = None  # ex. "Cohan et al. 2019 (Table 3 F1=88.3)"


class HallucinationAuditEntry(BaseModel):
    """
    Entrée d'audit pour une référence dans un ReviewDraft.
    auditor = "auto_nli" pour l'audit automatique,
              identifiant humain pour l'audit manuel Gold.
    """
    review_id: UUID
    sentence_text: str
    cited_key: str                  # clé BibTeX ou ID mentionné dans le texte
    status: VerificationStatus
    auditor: str = "auto_nli"
    note: str = ""


class HallucinationAudit(BaseModel):
    """
    Rapport d'audit complet pour un ReviewDraft.
    invented_rate = taux HALLUCINATED / total.
    support_rate  = taux VERIFIED / total.
    Ces deux valeurs apparaissent dans les tableaux du rapport final.
    """
    review_id: UUID
    entries: list[HallucinationAuditEntry]
    audited_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def invented_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(
            1 for e in self.entries if e.status == VerificationStatus.HALLUCINATED
        ) / len(self.entries)

    @computed_field
    @property
    def support_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(
            1 for e in self.entries if e.status == VerificationStatus.VERIFIED
        ) / len(self.entries)

    def summary(self) -> str:
        return (
            f"Audit review {self.review_id} | "
            f"inventé: {self.invented_rate:.1%} | "
            f"soutenu: {self.support_rate:.1%} | "
            f"n={len(self.entries)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. ANNOTATION  (vérité terrain)
# ═══════════════════════════════════════════════════════════════════════════


class SectionAnnotation(BaseModel):
    """
    Annotation humaine du type de section (vérité terrain pour SciBERT).
    Double annotation sur un sous-ensemble → InterAnnotatorKappa.
    """
    section_id: UUID
    annotator_id: str
    assigned_type: SectionType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_ambiguous: bool = False
    annotation_date: datetime = Field(default_factory=datetime.utcnow)


class CitationFunctionAnnotation(BaseModel):
    """
    Annotation humaine de la fonction de citation.
    Corpus cible : 500 citations annotées (exigence corpus § 6).
    Double annotation sur 20 % → kappa.
    """
    citation_id: UUID
    annotator_id: str
    function: CitationFunction
    is_ambiguous: bool = False
    annotation_date: datetime = Field(default_factory=datetime.utcnow)


class ArticleGridAnnotation(BaseModel):
    """
    Grille de lecture complète d'un article (§ 6 SciLit corpus).
    200 articles à annoter : question / contribution / méthode / résultats / limites.
    kappa_round : 1 = première passe, 2 = révision post-kappa.
    """
    article_id: UUID
    annotator_id: str
    research_question: str
    contribution: str
    method_summary: str
    main_results: str                   # résultats chiffrés principaux
    limitations: str
    relevant_claim_ids: list[UUID] = Field(default_factory=list)
    kappa_round: int = Field(default=1, ge=1, le=3)
    annotation_date: datetime = Field(default_factory=datetime.utcnow)


class ReferenceSoTAAnnotation(BaseModel):
    """
    Revue de littérature de référence rédigée par un étudiant / enseignant.
    20 états de l'art de référence requis (§ 6 SciLit).
    Sert de vérité terrain pour évaluer la couverture et l'exactitude
    du ReviewDraft généré (grille humaine, kappa juge).
    """
    id: UUID = Field(default_factory=uuid4)
    query_text: str
    annotator_id: str
    validated_by_teacher: bool = False
    review_text: str
    cited_article_ids: list[UUID]       # articles réellement cités (vérifiables)
    annotation_date: datetime = Field(default_factory=datetime.utcnow)


class InterAnnotatorKappa(BaseModel):
    """
    Cohen's kappa entre deux annotateurs sur un sous-ensemble.
    Requis pour : sections, fonctions de citation, grilles d'articles (§ 2.3).
    kappa ≥ 0.60 = accord substantiel (seuil conseillé).
    """
    task: str                   # "section_type", "citation_function", "article_grid"
    annotator_a: str
    annotator_b: str
    n_items: int
    kappa: float = Field(ge=-1.0, le=1.0)
    agreement_pct: float = Field(ge=0.0, le=1.0)
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    def is_acceptable(self, threshold: float = 0.60) -> bool:
        return self.kappa >= threshold


# ═══════════════════════════════════════════════════════════════════════════
# 8. CARTES  (§ 2.3.7 — obligatoires pour chaque composant)
# ═══════════════════════════════════════════════════════════════════════════


class DataCard(BaseModel):
    """
    Carte de données (§ 2.3.7).
    Une par composant de données : corpus articles, annotations sections,
    annotations citations, grilles articles, revues SoTA de référence.
    known_biases doit lister les biais de couverture source (Gold Ch11).
    """
    dataset_name: str
    description: str
    sources: list[str]              # ex. ["ACL Anthology", "arXiv cs.CL"]
    size: int
    languages: list[Language]
    split_sizes: dict[SplitType, int]
    annotation_schema_path: str     # chemin vers le guide d'annotation dans le dépôt
    kappa: float | None = Field(None, ge=-1.0, le=1.0)
    license: str
    known_biases: list[str] = Field(default_factory=list)
    collection_date: datetime


class ModelCard(BaseModel):
    """
    Carte de modèle (§ 2.3.7).
    Une par composant appris : SciBERT sections, classifieur citations,
    bi-encodeur retrieval, cross-encodeur reranker.
    limitations doit avoir ≥ 5 dimensions (§ 2.3.7).
    """
    model_name: str
    task: str                       # ex. "section classification"
    base_model: str                 # ex. "allenai/scibert_scivocab_uncased"
    training_data_card: str         # nom de la DataCard associée
    hyperparameters: dict[str, Any] # lr, batch_size, epochs, seed...
    evaluation: list[ComponentMetrics]
    ablations: list[AblationResult] = Field(default_factory=list)
    limitations: list[str] = Field(min_length=5)  # § 2.3.7 — 5 dimensions minimales
    carbon_footprint_kgco2: float | None = None
    training_date: datetime
    checkpoint_path: str | None = None  # chemin dans le dépôt Git

    @field_validator("limitations")
    @classmethod
    def five_dimensions(cls, v: list[str]) -> list[str]:
        if len(v) < 5:
            raise ValueError(
                f"§ 2.3.7 exige ≥ 5 dimensions de limitations — {len(v)} fournies."
            )
        return v