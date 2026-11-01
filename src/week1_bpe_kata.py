"""
Week 1, Kata 1 (No-AI Hand-Written Gate): Byte-Level BPE Tokenizer from Scratch
Canonical Reference: Andrej Karpathy's `minbpe` / "Let's Build the GPT Tokenizer"

ANTI-VIBE-CODING PROTOCOL:
1. Hand-write the 3 core functions below (`get_pair_counts`, `merge_pair`, `train_bpe`)
   without AI autocomplete (~25-30 lines total).
2. Run `.venv/bin/python src/week1_bpe_kata.py` to execute the deterministic test harness.
3. Pass the Explain-Back Gate with your FDE Mentor before we mark Week 1 Tokenizer Deep Dive complete.
"""

from __future__ import annotations


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequencies of all adjacent pairs `(ids[i], ids[i + 1])` in the sequence.

    Args:
        ids: Sequence of integer token IDs.

    Returns:
        Dictionary mapping each adjacent `(left_id, right_id)` tuple to its count.
    """
    # sample_ids = [1, 2, 3, 1, 2]

    commons = {}
    sample = ()
    for i, id in enumerate(ids[:-1]):
        sample = (ids[i], ids[i+1])
        commons[sample] = commons.get(sample, 0) + 1

    # print(commons)
    return commons
    # TODO (Hand-Write ~6 lines): Iterate over adjacent pairs in `ids` and tally counts.
    raise NotImplementedError(
        "Hand-write get_pair_counts() to unlock the kata gate!")


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of `pair` in `ids` with `new_id`.

    Important Edge Case:
        Walk `ids` with an index `i = 0`. When `(ids[i], ids[i + 1]) == pair`,
        append `new_id` and advance `i += 2` (so `[1, 1, 1]` merging `(1, 1) -> 256`
        becomes `[256, 1]`). Otherwise append `ids[i]` and advance `i += 1`.

    Args:
        ids: Current list of integer token IDs.
        pair: The `(left_id, right_id)` tuple to merge.
        new_id: The newly minted token ID replacing `pair`.

    Returns:
        New list of token IDs with `pair` merged into `new_id`.
    """
    new_list = []
    i = 0
    # for i, id in enumerate(ids[:-1]):
    # for i in range(len(ids[:-1])):
    while i < len(ids):
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            new_list.append(new_id)
            i += 2
        else:
            new_list.append(ids[i])
            i += 1

    return new_list

    # TODO (Hand-Write ~12 lines): Implement greedy left-to-right non-overlapping merge.
    raise NotImplementedError(
        "Hand-write merge_pair() to unlock the kata gate!")


def train_bpe(
    hltext: str, vocab_size: int
) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Train a Byte-Level BPE tokenizer starting from raw UTF-8 bytes (0..255).

    Algorithm:
        1. Convert `text` into raw UTF-8 byte integers: `ids = list(text.encode("utf-8"))`.
        2. Initialize base vocabulary: `vocab = {idx: bytes([idx]) for idx in range(256)}`
           and `merges: dict[tuple[int, int], int] = {}`.
        3. Loop `num_merges = vocab_size - 256` times (`i` from `0` to `num_merges - 1`):
           - Compute pair frequencies via `get_pair_counts(ids)`. If empty, break.
           - Select `best_pair = max(counts, key=counts.get)`.
           - Mint `new_id = 256 + i`.
           - Update `ids = merge_pair(ids, best_pair, new_id)`.
           - Record `merges[best_pair] = new_id` and
             `vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]`.

    Args:
        text: Training corpus string.
        vocab_size: Target vocabulary size (must be >= 256).

    Returns:
        Tuple of `(merges, vocab)`.
    """
    num_merges = vocab_size - 256

    # hltext = "Kubernetes"

    # 1. The list of numbers to shrink
    ids = list(hltext.encode("utf-8"))
    # 2. Decoder ring (starts with 0..255)
    print("\n lengths :", len(ids), " and list is", ids, "\n\n")
    vocab = {idx: bytes([idx]) for idx in range(256)}
    # vocab[256] = 100
    merges = {}                                         # 3. Empty notebook for rules

    for i in range(num_merges):
        max_repeat = max(get_pair_counts(ids), key=get_pair_counts(ids).get)

        merges[max_repeat] = len(vocab)+1
        vocab[len(vocab)+1] = vocab[max_repeat[0]]+vocab[max_repeat[1]]

        print(max_repeat)
        ids = merge_pair(ids, max_repeat, len(vocab))

    print("\n lengths :", len(vocab), " and list is", vocab, "\n\n")
    print("\n lengths :", len(ids), " and list is", ids, "\n\n")

    assert vocab_size >= 256, "Byte-level BPE base vocabulary requires at least 256 tokens."
    return merges, vocab
    # TODO (Hand-Write ~15 lines): Implement the iterative BPE training loop.
    raise NotImplementedError(
        "Hand-write train_bpe() to unlock the kata gate!")


# ==============================================================================
# TEST HARNESS & INFERENCE CODEC (Scaffolded — Do Not Modify Below)
# ==============================================================================


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode arbitrary string into token IDs by applying learned merges in priority order."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        # Pick the pair in `ids` that was learned earliest (lowest minted token ID in `merges`)
        best_pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if best_pair not in merges:
            break
        ids = merge_pair(ids, best_pair, merges[best_pair])
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode token IDs back into a UTF-8 string via byte concatenation."""
    raw_bytes = b"".join(vocab[idx] for idx in ids)
    return raw_bytes.decode("utf-8", errors="replace")


def run_verification_suite() -> None:
    print("=" * 78)
    print("WEEK 1 BPE KATA VERIFICATION SUITE")
    print("=" * 78)

    # 1. Unit test get_pair_counts
    sample_ids = [1, 2, 3, 1, 2]
    counts = get_pair_counts(sample_ids)
    assert counts[(1, 2)] == 2 and counts[(2, 3)] == 1 and counts[(3, 1)] == 1, (
        f"get_pair_counts failed: got {counts}"
    )
    print("[PASS] 1/4 get_pair_counts() accurately counts adjacent pairs.")

    # 2. Unit test merge_pair (including overlapping triplet edge case)
    merged_overlap = merge_pair([1, 1, 1], (1, 1), 256)
    assert merged_overlap == [256, 1], (
        f"merge_pair failed overlap test: expected [256, 1], got {merged_overlap}"
    )
    print("[PASS] 2/4 merge_pair() handles non-overlapping left-to-right merges.")

    # # 3. Train BPE on a realistic Kubernetes / FinOps telemetry corpus
    corpus = (
        "Kubernetes PodCrashLoopBackOff in namespace prod-inference. "
        "PodCrashLoopBackOff caused by CUDA OOM on Qwen3-4B KV-Cache growth. "
        "Reduce max_new_tokens or enable PagedAttention in vLLM for prod-inference."
    )
    # print("\n", corpus.encode('utf-8'))
    raw_byte_len = len(corpus.encode("utf-8"))

    target_vocab_size = 276  # 256 base UTF-8 bytes + 20 learned subword merges

    merges, vocab = train_bpe(corpus, vocab_size=target_vocab_size)

    assert len(merges) == 20 and len(vocab) == 276, (
        f"Expected 20 merges and 276 vocab entries, got {len(merges)} and {len(vocab)}"
    )
    print(f"[PASS] 3/4 train_bpe() learned {len(merges)} subword merges.")

    # Show top 5 learned merges
    print("\nFirst 5 Learned Subword Merges:")
    for (p0, p1), minted_id in list(merges.items())[:5]:
        token_str = vocab[minted_id].decode("utf-8", errors="replace")
        print(f"  ({p0:3d}, {p1:3d}) -> ID {minted_id} : {token_str!r}")

    # # 4. Verify lossless UTF-8 round-trip (including unseen multi-byte emoji!)
    test_text = "PodCrashLoopBackOff in prod-inference 🚀🔥"
    encoded_ids = encode(test_text, merges)
    decoded_text = decode(encoded_ids, vocab)
    assert decoded_text == test_text, (
        f"Round-trip mismatch: expected {test_text!r}, got {decoded_text!r}"
    )

    corpus_encoded = encode(corpus, merges)
    compression_ratio = raw_byte_len / len(corpus_encoded)
    print(
        f"\n[PASS] 4/4 Lossless UTF-8 Round-Trip verified (including unseen emoji 🚀🔥)!"
    )
    print(
        f"Corpus Compression: {raw_byte_len} raw UTF-8 bytes -> {len(corpus_encoded)} BPE tokens "
        f"({compression_ratio:.2f}x compression)"
    )
    print("=" * 78)


if __name__ == "__main__":
    run_verification_suite()
