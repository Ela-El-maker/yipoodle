from src.apps.kb_confidence import apply_confidence_decay, canonical_hash, canonicalize_claim, compute_claim_confidence


def test_canonicalization_and_hash_stable() -> None:
    a = canonicalize_claim("A Price is 10.0!")
    b = canonicalize_claim("a price is 10")
    assert a == b
    assert canonical_hash("A Price is 10.0!") == canonical_hash("a price is 10")


def test_confidence_and_decay() -> None:
    c = compute_claim_confidence(mean_support=0.7, citation_quality=0.9, recency=1.0, validation_signal=1.0)
    assert 0.0 <= c <= 1.0
    d = apply_confidence_decay(c, days_since_confirmed=10, decay_per_day=0.98)
    assert d <= c
