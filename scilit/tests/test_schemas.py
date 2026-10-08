from uuid import uuid4
import pytest
from pydantic import ValidationError
from scilit.schemas import (
    ComponentMetrics, SeedResult, VerifiedSentence,
    VerificationStatus, ReviewDraft, Language
)

def make_seeds(vals):
    return [SeedResult(seed=s, metric_name="F1", value=v)
            for s, v in zip([42, 123, 999], vals)]

def test_three_seeds_ok():
    m = ComponentMetrics(
        component_name="test", metric_name="F1",
        seeds=make_seeds([0.83, 0.84, 0.82])
    )
    assert abs(m.mean - 0.83) < 0.01
    assert m.std > 0

def test_two_seeds_rejected():
    with pytest.raises(ValidationError):
        ComponentMetrics(
            component_name="x", metric_name="F1",
            seeds=make_seeds([0.8, 0.9])[:2]
        )

def test_verified_sentence_hallucination():
    vs = VerifiedSentence(
        text="Model X achieves 92 F1.",
        citations=[uuid4(), uuid4()],
        verification_statuses=[
            VerificationStatus.VERIFIED,
            VerificationStatus.HALLUCINATED,
        ],
        supporting_claim_ids=[]
    )
    assert vs.hallucination_rate == 0.5
    assert vs.support_rate == 0.5
