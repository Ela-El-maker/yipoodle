from src.apps.vector_index import _shard_bucket


def test_shard_bucket_stable_for_same_id() -> None:
    sid = "paper123:S10"
    got = [_shard_bucket(sid, 8) for _ in range(5)]
    assert len(set(got)) == 1
    assert 0 <= got[0] < 8


def test_shard_bucket_varies_across_ids() -> None:
    buckets = {_shard_bucket(f"p{i}:s1", 8) for i in range(50)}
    # We don't need perfect balance, just non-degenerate assignment.
    assert len(buckets) > 1
