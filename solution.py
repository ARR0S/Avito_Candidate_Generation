"""Candidate Generation для поиска услуг Авито.

Полностью воспроизводимый финальный pipeline, который был использован для
получения benchmark Recall@50 = 0.818206.

Схема решения:
    1. grouped split train по нормализованному тексту запроса;
    2. полевой BM25 + мягкая geography до candidate cutoff;
    3. multilingual-e5-small: global и geography-conditioned semantic retrieval;
    4. объединение четырех retrieval-каналов;
    5. LightGBM LambdaRank на 39 признаках;
    6. одна итерация hard-negative mining;
    7. top-50 и строгая проверка answer.csv.

Первый полный запуск строит индексы, embeddings и обе версии ранкера, поэтому
занимает заметное время. Все промежуточные результаты кэшируются в work-dir.
Повторный запуск переиспользует их.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Конфигурация
# =============================================================================

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

SEARCH_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Заголовок — самый короткий и точный текстовый источник, поэтому его вес выше.
FIELDS = {
    "item_title_raw": (4.0, None),
    "item_infm_params_text": (1.0, 1000),
    "item_description_raw": (0.4, 2000),
}

BM25_K1 = 1.2
BM25_B = 0.75
FILTER_WEIGHT = 0.2
BASE_LOCATION_MULTIPLIER = 1.2
EXACT_LOCATION_MULTIPLIER = 3.0
CATEGORY_MULTIPLIER = 1.05
TRANSITION_STRENGTH = 0.5

GLOBAL_SEMANTIC_K = 1000
EXACT_SEMANTIC_K = 200
TRANSITION_SEMANTIC_K = 100
TRANSITION_LOCATIONS = 3
MIN_TRANSITION_SIGNAL = 0.25

SEED = 20260928
TRAIN_QUERIES = 10_000
INTERNAL_QUERIES = 500
SAMPLED_CANDIDATES = 250
CHANNEL_HEAD = 50
HARD_NEGATIVES = 200

MODEL_CONFIG = dict(
    objective="lambdarank",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=100,
    reg_lambda=5.0,
    min_split_gain=0.01,
    max_bin=127,
    lambdarank_truncation_level=60,
    n_jobs=4,
    random_state=SEED,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
    importance_type="gain",
)

CHANNELS = ("bm25", "global", "exact", "transition")
FEATURES = [
    "bm25_score", "bm25_relative", "lexical_score", "cosine",
    "cosine_gap_to_best", "cosine_range_normalized",
]
FEATURES += [f"{channel}_{name}" for channel in CHANNELS for name in ("rank", "rr", "present")]
FEATURES += [
    "channel_count", "exact_location", "transition_signal", "category_match",
    "query_title_overlap", "query_title_coverage", "title_query_coverage",
    "query_params_overlap", "query_params_coverage", "query_description_coverage",
    "filter_params_coverage", "query_tokens", "query_chars", "filter_tokens",
    "title_tokens", "params_tokens", "item_rating", "log_reviews",
    "phone_hidden", "message_forbidden", "log_price",
]
assert len(FEATURES) == 39

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


# =============================================================================
# Общие утилиты
# =============================================================================

def normalize(text: str | None) -> str:
    """Нормализация, использованная во всех этапах решения."""
    return " ".join((text or "").lower().replace("ё", "е").split())


def tokenize(text: str | None) -> list[str]:
    """Unicode-токены без стемминга — одинаково для индекса и overlap features."""
    return TOKEN_RE.findall(normalize(text))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Детерминированный top-k: при равных score раньше идет меньший doc id."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    threshold = np.partition(scores, len(scores) - k)[-k]
    better = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[: k - len(better)]
    selected = np.concatenate([better, tied])
    return selected[np.lexsort((selected, -scores[selected]))].astype(np.int32, copy=False)


def key_id(key: tuple) -> str:
    """Стабильный ID полного поискового ключа; используется только при split."""
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def query_record(key: tuple, positives: set[str]) -> dict:
    return dict(query_id=key_id(key), **dict(zip(SEARCH_COLUMNS, key)), positives=sorted(positives))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# 1. Grouped split и training query groups
# =============================================================================

def build_validation_split(train_path: Path, split_dir: Path) -> tuple[Path, Path]:
    """Воспроизводит исходный grouped split по normalized search_query.

    Для финального ранкера нужен fit без holdout-текстов. Secondary-ключи также
    исключаются ровно так же, как в экспериментальном pipeline, чтобы training
    data совпадали с теми, на которых была получена финальная модель.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    fit_path = split_dir / "fit.jsonl"
    holdout_path = split_dir / "holdout_texts.json"
    if fit_path.exists() and holdout_path.exists():
        return fit_path, holdout_path

    groups: dict[tuple, set[str]] = {}
    columns = SEARCH_COLUMNS + ["item_id"]
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            key = tuple(normalize(row[c]) if c == "search_query" else row[c] for c in SEARCH_COLUMNS)
            groups.setdefault(key, set()).add(row["item_id"])

    texts = sorted({key[0] for key in groups})
    permutation = np.random.RandomState(42).permutation(len(texts))
    holdout = {texts[i] for i in permutation[: math.ceil(0.2 * len(texts))]}

    by_text: dict[str, list[tuple]] = defaultdict(list)
    for key in groups:
        by_text[key[0]].append(key)
    for keys in by_text.values():
        keys.sort(key=key_id)

    # Secondary нужен только для точного воспроизведения состава fit.
    warm_texts = sorted(text for text in texts if text not in holdout and len(by_text[text]) > 1)
    warm_order = np.random.RandomState(43).permutation(len(warm_texts))[:500]
    secondary = {by_text[warm_texts[i]][0] for i in warm_order}

    fit_rows = [
        query_record(key, groups[key])
        for key in groups
        if key[0] not in holdout and key not in secondary
    ]
    write_jsonl(fit_path, fit_rows)
    save_json(holdout_path, sorted(holdout))
    return fit_path, holdout_path


def select_ranker_queries(fit_path: Path, output_dir: Path) -> tuple[list[dict], list[dict], set[str]]:
    """Детерминированно выбираем 10k train + 500 internal query texts."""
    train_file = output_dir / "ranker_train.jsonl"
    internal_file = output_dir / "ranker_internal.jsonl"
    if train_file.exists() and internal_file.exists():
        train_rows, internal_rows = read_jsonl(train_file), read_jsonl(internal_file)
        return train_rows, internal_rows, {q["search_query"] for q in train_rows + internal_rows}

    best: dict[str, dict] = {}
    fit_rows = read_jsonl(fit_path)
    for query in fit_rows:
        text = query["search_query"]
        if text not in best or query["query_id"] < best[text]["query_id"]:
            best[text] = query

    texts = sorted(best)
    order = np.random.RandomState(SEED).permutation(len(texts))
    selected = [best[texts[i]] for i in order[: TRAIN_QUERIES + INTERNAL_QUERIES]]
    train_rows = selected[:TRAIN_QUERIES]
    internal_rows = selected[TRAIN_QUERIES:]
    write_jsonl(train_file, train_rows)
    write_jsonl(internal_file, internal_rows)
    return train_rows, internal_rows, {q["search_query"] for q in selected}


# =============================================================================
# 2. Корпус и полевой BM25
# =============================================================================

CORPUS_COLUMNS = [
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_location_id", "item_category_id", "item_microcat_id", "item_rating",
    "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden",
    "item_price",
]


def make_unique_corpus(source: Path, target: Path) -> int:
    """Сохраняет по одной строке на item_id в исходном порядке."""
    if target.exists():
        return pq.ParquetFile(target).metadata.num_rows

    seen: set[str] = set()
    writer = None
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=CORPUS_COLUMNS):
            positions = []
            for i, item_id in enumerate(batch.column(0).to_pylist()):
                if item_id not in seen:
                    seen.add(item_id)
                    positions.append(i)
            if not positions:
                continue
            table = pa.Table.from_batches([batch]).take(pa.array(positions, type=pa.int32()))
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return len(seen)


def field_tokens(value: str | None, limit: int | None) -> list[str]:
    text = normalize(value)
    if limit is not None:
        text = text[:limit]
    return tokenize(text)


def iter_field(corpus_path: Path, field: str):
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=512, columns=[field]):
        yield from batch.column(0).to_pylist()


def build_bm25_field(corpus_path: Path, field_dir: Path, field: str, limit: int | None, n_docs: int) -> None:
    """Двухпроходный inverted index: DF/lengths -> BM25 postings."""
    field_dir.mkdir(parents=True, exist_ok=True)
    ready = field_dir / "vocabulary.pkl"
    if ready.exists():
        return

    vocabulary: dict[str, int] = {}
    frequencies: list[int] = []
    lengths = np.zeros(n_docs, dtype=np.float32)

    for doc, value in enumerate(iter_field(corpus_path, field)):
        tokens = field_tokens(value, limit)
        lengths[doc] = len(tokens)
        for token in set(tokens):
            term = vocabulary.get(token)
            if term is None:
                term = len(vocabulary)
                vocabulary[token] = term
                frequencies.append(0)
            frequencies[term] += 1

    df = np.asarray(frequencies, dtype=np.int64)
    offsets = np.zeros(len(df) + 1, dtype=np.int64)
    np.cumsum(df, out=offsets[1:])
    nnz = int(offsets[-1])

    docs = np.lib.format.open_memmap(field_dir / "docs.npy", mode="w+", dtype="int32", shape=(nnz,))
    weights = np.lib.format.open_memmap(field_dir / "weights.npy", mode="w+", dtype="float32", shape=(nnz,))
    cursor = offsets[:-1].copy()

    average = float(lengths.mean()) or 1.0
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
    norms = BM25_K1 * (1 - BM25_B + BM25_B * lengths / average)

    for doc, value in enumerate(iter_field(corpus_path, field)):
        counts = Counter(field_tokens(value, limit))
        if not counts:
            continue
        terms = np.fromiter((vocabulary[token] for token in counts), dtype=np.int64)
        tf = np.fromiter(counts.values(), dtype=np.float32)
        positions = cursor[terms]
        docs[positions] = doc
        weights[positions] = idf[terms] * tf * (BM25_K1 + 1) / (tf + norms[doc])
        cursor[terms] += 1

    docs.flush()
    weights.flush()
    np.save(field_dir / "offsets.npy", offsets)
    with ready.open("wb") as stream:
        pickle.dump(vocabulary, stream, protocol=5)


def build_bm25_index(corpus_path: Path, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    n_docs = pq.ParquetFile(corpus_path).metadata.num_rows
    for field, (_, limit) in FIELDS.items():
        build_bm25_field(corpus_path, index_dir / field, field, limit, n_docs)
        gc.collect()


class BM25:
    """Полевой BM25, который хранит postings на диске и читает их через mmap."""

    def __init__(self, corpus_path: Path, index_dir: Path):
        meta = pq.read_table(
            corpus_path,
            columns=["item_id", "item_location_id", "item_category_id", "item_microcat_id"],
        )
        self.ids = np.array(meta["item_id"].to_pylist(), dtype="U16")
        self.locations = meta["item_location_id"].to_numpy()
        self.categories = meta["item_category_id"].to_numpy()
        self.microcats = meta["item_microcat_id"].to_numpy()
        self.fields = []
        for field, (weight, _) in FIELDS.items():
            directory = index_dir / field
            with (directory / "vocabulary.pkl").open("rb") as stream:
                vocabulary = pickle.load(stream)
            self.fields.append((
                weight,
                vocabulary,
                np.load(directory / "offsets.npy", mmap_mode="r"),
                np.load(directory / "docs.npy", mmap_mode="r"),
                np.load(directory / "weights.npy", mmap_mode="r"),
            ))

    def scores(self, query: dict) -> np.ndarray:
        # Query tokens имеют вес 1, фильтры — небольшой дополнительный вес.
        query_weights = {token: 1.0 for token in tokenize(query["search_query"])}
        for token in sorted(set(tokenize(query["search_infm_params_text"]))):
            query_weights[token] = query_weights.get(token, 0.0) + FILTER_WEIGHT

        scores = np.zeros(len(self.ids), dtype=np.float32)
        for field_weight, vocabulary, offsets, docs, weights in self.fields:
            for token, query_weight in query_weights.items():
                term = vocabulary.get(token)
                if term is None:
                    continue
                section = slice(offsets[term], offsets[term + 1])
                scores[docs[section]] += field_weight * query_weight * weights[section]

        # Эти базовые множители были частью исходного BM25 baseline.
        scores *= 1 + 0.20 * (self.locations == query["search_location_id"])
        if query["search_category"] not in (None, 0):
            scores *= 1 + 0.05 * (self.categories == query["search_category"])
        return scores


def geography_scores(model: BM25, query: dict, signals: dict[int, list[tuple[int, float]]]) -> np.ndarray:
    """Мягкая география применяется ДО top-k, чтобы не потерять локальные items."""
    scores = model.scores(query).astype(np.float64)

    # BM25.scores уже содержит ×1.2; доводим exact location до итогового ×3.
    exact = model.locations == query["search_location_id"]
    scores[exact] *= EXACT_LOCATION_MULTIPLIER / BASE_LOCATION_MULTIPLIER

    for target, signal in signals.get(query["search_location_id"], []):
        scores[model.locations == target] *= 1 + TRANSITION_STRENGTH * signal
    return scores


# =============================================================================
# 3. Сглаженные переходы между локациями
# =============================================================================

def transition_rows_from_queries(queries: list[dict], item_location: dict[str, int], excluded_texts: set[str] | None = None) -> list[dict]:
    """Строит leakage-safe transition statistics по агрегированным query groups."""
    excluded_texts = excluded_texts or set()
    pairs, sources, targets = Counter(), Counter(), Counter()

    for query in queries:
        if query["search_query"] in excluded_texts:
            continue
        source = query["search_location_id"]
        for item_id in query["positives"]:
            target = item_location[item_id]
            pairs[source, target] += 1
            sources[source] += 1
            targets[target] += 1

    return _transition_rows(pairs, sources, targets)


def transition_rows_from_train(train_path: Path) -> list[dict]:
    """Финальные benchmark transitions по всем уникальным train query–item парам."""
    pairs, sources, targets = Counter(), Counter(), Counter()
    seen: set[tuple] = set()
    columns = SEARCH_COLUMNS + ["item_id", "item_location_id"]

    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            key = tuple(normalize(row[c]) if c == "search_query" else row[c] for c in SEARCH_COLUMNS)
            unique_pair = key + (row["item_id"],)
            if unique_pair in seen:
                continue
            seen.add(unique_pair)
            source, target = row["search_location_id"], row["item_location_id"]
            pairs[source, target] += 1
            sources[source] += 1
            targets[target] += 1

    return _transition_rows(pairs, sources, targets)


def _transition_rows(pairs: Counter, sources: Counter, targets: Counter) -> list[dict]:
    total = sum(targets.values())
    rows = []
    for (source, target), count in sorted(pairs.items()):
        if source == target or sources[source] < 30 or count < 5:
            continue
        prior = targets[target] / total
        conditional = (count + 100 * prior) / (sources[source] + 100)
        signal = max(0.0, (conditional - prior) / (conditional + prior))
        signal *= count / (count + 20)
        if signal > 0:
            rows.append(dict(source=source, target=target, count=count, signal=signal))
    return rows


def transition_maps(rows: list[dict]) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[int]]]:
    """Возвращает все signals для BM25 и top-3 locations для semantic retrieval."""
    signals: dict[int, list[tuple[int, float]]] = defaultdict(list)
    supported: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        signals[row["source"]].append((row["target"], row["signal"]))
        if row["signal"] >= MIN_TRANSITION_SIGNAL:
            supported[row["source"]].append(row)

    selected = {
        source: [row["target"] for row in sorted(values, key=lambda r: (-r["signal"], -r["count"], r["target"]))[:TRANSITION_LOCATIONS]]
        for source, values in supported.items()
    }
    return dict(signals), selected


# =============================================================================
# 4. multilingual-e5-small
# =============================================================================

def document_text(row: dict) -> str:
    return "passage: " + row["item_title_raw"] + ". " + (row["item_infm_params_text"] or "")[:600]


def query_text(row: dict) -> str:
    return "query: " + row["search_query"] + ". " + (row["search_infm_params_text"] or "")[:200]


class E5Encoder:
    """Mean pooling + L2 normalization — тот же encoding, что в эксперименте."""

    def __init__(self, model_path: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(4)

        source = model_path if model_path else MODEL_ID
        kwargs = {"local_files_only": True} if model_path else {"revision": MODEL_REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.model = AutoModel.from_pretrained(source, use_safetensors=True, **kwargs).eval().to(self.device)

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32)


def encode_corpus(
    encoder: E5Encoder,
    corpus_path: Path,
    output_path: Path,
    reuse_ids: dict[str, int] | None = None,
    reuse_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Кодирует corpus; для пересекающихся benchmark items можно переиспользовать local vectors."""
    if output_path.exists():
        return np.load(output_path, mmap_mode="r")

    count = pq.ParquetFile(corpus_path).metadata.num_rows
    embeddings = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(count, 384))
    offset = 0

    for batch in pq.ParquetFile(corpus_path).iter_batches(
        batch_size=1024,
        columns=["item_id", "item_title_raw", "item_infm_params_text"],
    ):
        pending, texts = [], []
        for row in batch.to_pylist():
            if reuse_ids is not None and row["item_id"] in reuse_ids:
                embeddings[offset] = reuse_embeddings[reuse_ids[row["item_id"]]]
            else:
                pending.append(offset)
                texts.append(document_text(row))
            offset += 1

        for begin in range(0, len(texts), 32):
            embeddings[pending[begin:begin + 32]] = encoder.encode(texts[begin:begin + 32])
        embeddings.flush()

    return embeddings


def encode_queries(encoder: E5Encoder, queries: list[dict], output_path: Path) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path)
    values = np.concatenate([
        encoder.encode([query_text(q) for q in queries[i:i + 32]])
        for i in range(0, len(queries), 32)
    ])
    np.save(output_path, values)
    return values


def global_semantic_retrieve(corpus_embeddings: np.ndarray, query_embeddings: np.ndarray, output_path: Path) -> np.ndarray:
    """Top-1000 по cosine, без полной query×item матрицы."""
    if output_path.exists():
        return np.load(output_path)["candidates"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = torch.from_numpy(np.asarray(corpus_embeddings)).to(device)
    candidates = np.empty((len(query_embeddings), GLOBAL_SEMANTIC_K), dtype=np.int32)

    with torch.inference_mode():
        for begin in range(0, len(query_embeddings), 32):
            values = (torch.from_numpy(query_embeddings[begin:begin + 32]).to(device) @ docs.T).cpu().numpy()
            for j, row in enumerate(values):
                candidates[begin + j] = top_k(row, GLOBAL_SEMANTIC_K)

    del docs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.savez_compressed(output_path, candidates=candidates)
    return candidates


def geography_semantic_retrieve(
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_locations: list[int],
    item_locations: np.ndarray,
    selected_transitions: dict[int, list[int]],
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Semantic top-k считается ВНУТРИ локации до объединения каналов."""
    if output_path.exists():
        data = np.load(output_path)
        return data["exact"], data["transition"]

    groups: dict[int, list[int]] = defaultdict(list)
    for i, location in enumerate(query_locations):
        groups[location].append(i)

    exact = np.full((len(query_locations), EXACT_SEMANTIC_K), -1, dtype=np.int32)
    transition = np.full((len(query_locations), TRANSITION_SEMANTIC_K), -1, dtype=np.int32)

    for channel, output in (("exact", exact), ("transition", transition)):
        for location, query_indices in groups.items():
            targets = [location] if channel == "exact" else selected_transitions.get(location, [])
            docs = np.flatnonzero(np.isin(item_locations, targets))
            if not len(docs):
                continue

            for begin in range(0, len(query_indices), 32):
                rows = query_indices[begin:begin + 32]
                best_docs = [np.empty(0, dtype=np.int32) for _ in rows]
                best_scores = [np.empty(0, dtype=np.float32) for _ in rows]

                # Обрабатываем большие location subsets блоками, чтобы не создавать
                # большую матрицу similarities целиком.
                for offset in range(0, len(docs), 8192):
                    subset = docs[offset:offset + 8192]
                    scores = query_embeddings[rows] @ embeddings[subset].T
                    for j, values in enumerate(scores):
                        merged_docs = np.concatenate((best_docs[j], subset))
                        merged_scores = np.concatenate((best_scores[j], values))
                        order = np.argsort(merged_docs)
                        merged_docs, merged_scores = merged_docs[order], merged_scores[order]
                        k = output.shape[1]
                        selected = top_k(merged_scores, min(k, len(merged_docs)))
                        best_docs[j], best_scores[j] = merged_docs[selected], merged_scores[selected]

                for j, row in enumerate(rows):
                    output[row, : len(best_docs[j])] = best_docs[j]

    np.savez_compressed(output_path, exact=exact, transition=transition)
    return exact, transition


# =============================================================================
# 5. 39 признаков reranker
# =============================================================================

class FeatureBuilder:
    """Признаки retrieval + geography + lexical overlaps + item attributes."""

    def __init__(
        self,
        model: BM25,
        signals: dict[int, list[tuple[int, float]]],
        embeddings: np.ndarray,
        corpus_path: Path,
    ):
        self.model = model
        self.signals = signals
        # Для training corpus embeddings читаются многократно; RAM быстрее HDD mmap.
        self.embeddings = np.asarray(embeddings)
        n = len(model.ids)

        self.title_len = np.zeros(n, dtype=np.float32)
        self.params_len = np.zeros(n, dtype=np.float32)
        self.item = np.full((n, 5), np.nan, dtype=np.float32)

        columns = [
            "item_title_raw", "item_infm_params_text", "item_rating",
            "item_rating_reviews_count", "item_is_phone_hidden",
            "item_is_message_forbidden", "item_price",
        ]
        offset = 0
        for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                self.title_len[offset] = len(set(tokenize(row["item_title_raw"])))
                self.params_len[offset] = len(set(tokenize(normalize(row["item_infm_params_text"])[:1000])))

                values = [
                    row["item_rating"], row["item_rating_reviews_count"],
                    row["item_is_phone_hidden"], row["item_is_message_forbidden"],
                    row["item_price"],
                ]
                values = [float(v) if v is not None else np.nan for v in values]
                values[1] = np.log1p(max(0, values[1])) if np.isfinite(values[1]) else np.nan
                values[4] = np.log1p(max(0, values[4])) if np.isfinite(values[4]) else np.nan
                self.item[offset] = values
                offset += 1
        assert offset == n

    def overlap(self, field_index: int, tokens: set[str], docs: np.ndarray) -> np.ndarray:
        """Считает количество query tokens, встречающихся в поле каждого candidate."""
        _, vocabulary, offsets, postings, _ = self.model.fields[field_index]
        counts = np.zeros(len(self.model.ids), dtype=np.float32)
        for token in tokens:
            term = vocabulary.get(token)
            if term is not None:
                counts[postings[offsets[term]:offsets[term + 1]]] += 1
        return counts[docs]

    def make(
        self,
        query: dict,
        query_embedding: np.ndarray,
        docs: np.ndarray,
        channels: list[np.ndarray],
        bm25_scores: np.ndarray,
    ) -> np.ndarray:
        qtokens = set(tokenize(query["search_query"]))
        ftokens = set(tokenize(query["search_infm_params_text"]))
        n = len(docs)

        values = bm25_scores[docs].astype(np.float32)
        signal_map = dict(self.signals.get(query["search_location_id"], []))
        signal = np.asarray([signal_map.get(int(v), 0.0) for v in self.model.locations[docs]], dtype=np.float32)
        exact = self.model.locations[docs] == query["search_location_id"]
        category = (
            (self.model.categories[docs] == query["search_category"])
            & (query["search_category"] not in (None, 0))
        )

        # Убираем multiplicative geography/category из итогового BM25 score и
        # предоставляем модели как raw, так и de-biased lexical score.
        lexical = values / np.where(exact, 3, 1) / (1 + 0.5 * signal) / np.where(category, 1.05, 1)
        cosine = self.embeddings[docs] @ query_embedding

        result = [
            values,
            values / max(float(values.max()), 1e-6),
            lexical,
            cosine,
            cosine - cosine.max(),
            (cosine - cosine.min()) / max(float(np.ptp(cosine)), 1e-6),
        ]

        # Для каждого retrieval channel добавляем rank, reciprocal rank и presence.
        presence = np.zeros(n, dtype=np.float32)
        for channel in channels:
            rank = np.full(n, 2001, dtype=np.float32)
            valid = channel[channel >= 0]
            positions = np.searchsorted(docs, valid)
            rank[positions] = np.arange(1, len(valid) + 1)
            present = rank < 2001
            presence += present
            result += [rank, np.where(present, 1 / rank, 0), present]

        title = self.overlap(0, qtokens, docs)
        params = self.overlap(1, qtokens, docs)
        description = self.overlap(2, qtokens, docs)
        filters = self.overlap(1, ftokens, docs)
        qlen = max(len(qtokens), 1)

        result += [
            presence,
            exact,
            signal,
            category,
            title,
            title / qlen,
            title / np.maximum(self.title_len[docs], 1),
            params,
            params / qlen,
            description / qlen,
            filters / max(len(ftokens), 1),
            np.full(n, len(qtokens)),
            np.full(n, len(query["search_query"])),
            np.full(n, len(ftokens)),
            self.title_len[docs],
            self.params_len[docs],
        ]
        result += [self.item[docs, i] for i in range(5)]

        matrix = np.column_stack(result).astype(np.float32)
        assert matrix.shape == (len(docs), len(FEATURES))
        assert not np.isinf(matrix).any()
        return matrix


# =============================================================================
# 6. Candidate channels для ranker training
# =============================================================================

def build_query_channels(
    name: str,
    queries: list[dict],
    model: BM25,
    signals: dict[int, list[tuple[int, float]]],
    selected_transitions: dict[int, list[int]],
    corpus_embeddings: np.ndarray,
    encoder: E5Encoder,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает BM25-500, global-1000, exact-200, transition-100 и query embeddings."""
    query_embeddings = encode_queries(encoder, queries, cache_dir / f"{name}_query_embeddings.npy")
    global_docs = global_semantic_retrieve(
        corpus_embeddings,
        query_embeddings,
        cache_dir / f"{name}_global_semantic.npz",
    )
    exact, transition = geography_semantic_retrieve(
        corpus_embeddings,
        query_embeddings,
        [q["search_location_id"] for q in queries],
        model.locations,
        selected_transitions,
        cache_dir / f"{name}_geo_semantic.npz",
    )

    bm_path = cache_dir / f"{name}_bm25.npy"
    if bm_path.exists():
        bm = np.load(bm_path)
    else:
        bm = np.empty((len(queries), 500), dtype=np.int32)
        for i, query in enumerate(queries):
            bm[i] = top_k(geography_scores(model, query, signals), 500)
        np.save(bm_path, bm)
    return bm, global_docs, exact, transition, query_embeddings


def sample_v1(docs: np.ndarray, channels: list[np.ndarray], positives: set[int]) -> np.ndarray:
    """V1: positives + top-50 каждого канала + RRF-дополнение до 250."""
    required = set(positives) & set(docs)
    for channel in channels:
        required.update(int(doc) for doc in channel[:CHANNEL_HEAD] if doc >= 0)

    rrf = np.zeros(len(docs))
    for channel in channels:
        valid = channel[channel >= 0]
        rrf[np.searchsorted(docs, valid)] += 1 / (60 + np.arange(1, len(valid) + 1))

    for index in np.lexsort((docs, -rrf)):
        if len(required) >= SAMPLED_CANDIDATES:
            break
        required.add(int(docs[index]))
    return np.searchsorted(docs, sorted(required))


def sample_hard_negative_v2(
    docs: np.ndarray,
    channels: list[np.ndarray],
    positives: set[int],
    model_scores: np.ndarray,
) -> np.ndarray:
    """V2: positives + 200 ошибок v1 с максимальным score + RRF до 250."""
    labels = np.isin(docs, list(positives))
    order = np.lexsort((docs, -model_scores))
    hard = order[~labels[order]][:HARD_NEGATIVES]
    selected = set(np.flatnonzero(labels)) | set(hard)

    rrf = np.zeros(len(docs))
    for channel in channels:
        valid = channel[channel >= 0]
        rrf[np.searchsorted(docs, valid)] += 1 / (60 + np.arange(1, len(valid) + 1))
    for index in np.lexsort((docs, -rrf)):
        if len(selected) >= SAMPLED_CANDIDATES:
            break
        selected.add(int(index))
    return np.array(sorted(selected), dtype=np.int32)


def build_training_table(
    name: str,
    queries: list[dict],
    model: BM25,
    builder: FeatureBuilder,
    signals: dict[int, list[tuple[int, float]]],
    channels_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
    mode: str,
    v1_model: lgb.Booster | None = None,
) -> tuple[Path, Path, np.ndarray]:
    """Строит memmap X/y и group sizes для LambdaRank."""
    x_path = output_dir / f"{name}_{mode}_X.npy"
    y_path = output_dir / f"{name}_{mode}_y.npy"
    groups_path = output_dir / f"{name}_{mode}_groups.npy"
    if x_path.exists() and y_path.exists() and groups_path.exists():
        return x_path, y_path, np.load(groups_path)

    bm, global_docs, exact, transition, query_embeddings = channels_data
    id_to_doc = {item: i for i, item in enumerate(model.ids)}

    # Train хранит около 250 rows/query, internal — полный candidate pool (~1800).
    rows_per_query = 250 if name == "train" else 1800
    max_rows = len(queries) * rows_per_query + sum(len(q["positives"]) for q in queries)
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype="float32", shape=(max_rows, len(FEATURES)))
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype="uint8", shape=(max_rows,))

    groups = []
    offset = 0
    for i, query in enumerate(queries):
        scores = geography_scores(model, query, signals)
        channels = [bm[i], global_docs[i], exact[i], transition[i]]
        docs = np.unique(np.concatenate(channels))
        docs = docs[docs >= 0]

        positives = {id_to_doc[item] for item in query["positives"]}
        found = positives & set(docs)
        if name == "train" and not found:
            # LambdaRank group без positive не дает полезного ranking signal.
            continue

        features = builder.make(query, query_embeddings[i], docs, channels, scores)
        if name == "internal":
            selected = np.arange(len(docs))
        elif mode == "v1":
            selected = sample_v1(docs, channels, positives)
        else:
            assert v1_model is not None
            selected = sample_hard_negative_v2(
                docs,
                channels,
                positives,
                v1_model.predict(features, num_threads=4),
            )

        size = len(selected)
        x[offset:offset + size] = features[selected]
        y[offset:offset + size] = np.isin(docs[selected], list(positives))
        groups.append(size)
        offset += size

        if (i + 1) % 500 == 0:
            print(f"{mode.upper()} TABLE {name}: {i + 1}/{len(queries)}, rows={offset}", flush=True)

    x.flush()
    y.flush()
    groups_array = np.asarray(groups, dtype=np.int32)
    np.save(groups_path, groups_array)
    save_json(output_dir / f"{name}_{mode}_shape.json", {"rows": int(offset), "groups": len(groups)})
    return x_path, y_path, groups_array


def train_ranker(
    train_x_path: Path,
    train_y_path: Path,
    train_groups: np.ndarray,
    internal_x_path: Path,
    internal_y_path: Path,
    internal_groups: np.ndarray,
    model_path: Path,
) -> lgb.Booster:
    """Обучение с early stopping только на отдельном fit-only internal set."""
    if model_path.exists():
        return lgb.Booster(model_file=str(model_path))

    x_train = np.load(train_x_path, mmap_mode="r")[: train_groups.sum()]
    y_train = np.load(train_y_path, mmap_mode="r")[: train_groups.sum()]
    x_internal = np.load(internal_x_path, mmap_mode="r")[: internal_groups.sum()]
    y_internal = np.load(internal_y_path, mmap_mode="r")[: internal_groups.sum()]

    ranker = lgb.LGBMRanker(**MODEL_CONFIG)
    ranker.fit(
        x_train,
        y_train,
        group=train_groups,
        eval_set=[(x_internal, y_internal)],
        eval_group=[internal_groups],
        eval_at=[50],
        feature_name=FEATURES,
        callbacks=[
            lgb.early_stopping(40, first_metric_only=True),
            lgb.log_evaluation(20),
        ],
    )
    ranker.booster_.save_model(str(model_path))
    return ranker.booster_


# =============================================================================
# 7. Полное обучение v1 -> hard negatives -> v2
# =============================================================================

def prepare_ranker(
    train_path: Path,
    local_corpus: Path,
    local_index: Path,
    local_embeddings: np.ndarray,
    encoder: E5Encoder,
    work_dir: Path,
) -> Path:
    """Воспроизводит обучение финального LGBMRanker v2."""
    ranker_dir = work_dir / "ranker"
    ranker_dir.mkdir(exist_ok=True)
    final_model = ranker_dir / "model_v2.txt"
    if final_model.exists():
        return final_model

    fit_path, _ = build_validation_split(train_path, work_dir / "split")
    train_queries, internal_queries, excluded_texts = select_ranker_queries(fit_path, ranker_dir)
    fit_queries = read_jsonl(fit_path)

    local_model = BM25(local_corpus, local_index)
    item_location = dict(zip(local_model.ids.tolist(), local_model.locations.tolist()))

    # Очень важно: в training transition statistics полностью исключаются все
    # 10 500 query texts, используемых train/internal ranker groups.
    safe_rows = transition_rows_from_queries(fit_queries, item_location, excluded_texts)
    signals, selected_transitions = transition_maps(safe_rows)
    save_json(ranker_dir / "safe_transitions.json", safe_rows)

    train_channels = build_query_channels(
        "train", train_queries, local_model, signals, selected_transitions,
        local_embeddings, encoder, ranker_dir,
    )
    internal_channels = build_query_channels(
        "internal", internal_queries, local_model, signals, selected_transitions,
        local_embeddings, encoder, ranker_dir,
    )

    builder = FeatureBuilder(local_model, signals, local_embeddings, local_corpus)

    # V1: обычные hard retrieval candidates.
    train_v1_x, train_v1_y, train_v1_groups = build_training_table(
        "train", train_queries, local_model, builder, signals, train_channels,
        ranker_dir, mode="v1",
    )
    internal_x, internal_y, internal_groups = build_training_table(
        "internal", internal_queries, local_model, builder, signals, internal_channels,
        ranker_dir, mode="internal",
    )
    model_v1_path = ranker_dir / "model_v1.txt"
    model_v1 = train_ranker(
        train_v1_x, train_v1_y, train_v1_groups,
        internal_x, internal_y, internal_groups,
        model_v1_path,
    )

    # V2: те же признаки и candidate pool, но negatives выбирает уже frozen v1.
    train_v2_x, train_v2_y, train_v2_groups = build_training_table(
        "train", train_queries, local_model, builder, signals, train_channels,
        ranker_dir, mode="v2", v1_model=model_v1,
    )
    train_ranker(
        train_v2_x, train_v2_y, train_v2_groups,
        internal_x, internal_y, internal_groups,
        final_model,
    )
    return final_model


# =============================================================================
# 8. Benchmark inference и answer.csv
# =============================================================================

def validate_submission(rows: list[dict], queries: list[dict], item_ids: list[str]) -> dict:
    expected = [query["query_id"] for query in queries]
    assert len(expected) == len(set(expected))
    assert len(rows) == len(expected)

    actual = [row["query_id"] for row in rows]
    assert len(actual) == len(set(actual))
    assert set(actual) == set(expected)

    allowed = set(item_ids)
    counts = []
    for row in rows:
        assert set(row) == {"query_id", "answer"}
        ids = row["answer"].split() if row["answer"] else []
        assert len(ids) == min(50, len(allowed))
        assert len(ids) == len(set(ids))
        assert set(ids) <= allowed
        counts.append(len(ids))

    return {
        "queries": len(rows),
        "mean_candidates": float(np.mean(counts)),
        "min_candidates": min(counts),
        "max_candidates": max(counts),
        "validation": "PASS",
    }


def benchmark_inference(
    train_path: Path,
    queries_path: Path,
    items_path: Path,
    local_model: BM25,
    local_embeddings: np.ndarray,
    encoder: E5Encoder,
    final_model_path: Path,
    work_dir: Path,
    output_path: Path,
) -> dict:
    bench_dir = work_dir / "benchmark"
    bench_dir.mkdir(exist_ok=True)

    corpus_path = bench_dir / "corpus.parquet"
    make_unique_corpus(items_path, corpus_path)
    index_dir = bench_dir / "bm25_index"
    build_bm25_index(corpus_path, index_dir)
    retrieval = BM25(corpus_path, index_dir)

    queries = pq.read_table(queries_path).to_pylist()

    # Benchmark embeddings переиспользуют vectors тех item_id, которые уже были
    # закодированы в local train corpus. Это ускоряет запуск и повторяет эксперимент.
    local_id_to_doc = {item: i for i, item in enumerate(local_model.ids)}
    benchmark_embeddings = encode_corpus(
        encoder,
        corpus_path,
        bench_dir / "embeddings.npy",
        reuse_ids=local_id_to_doc,
        reuse_embeddings=local_embeddings,
    )
    query_embeddings = encode_queries(encoder, queries, bench_dir / "query_embeddings.npy")
    global_docs = global_semantic_retrieve(
        benchmark_embeddings,
        query_embeddings,
        bench_dir / "global_semantic.npz",
    )

    # После model selection разрешено использовать весь train для fit-dependent
    # geography statistics: benchmark labels при этом не используются.
    transition_file = bench_dir / "transitions.json"
    if transition_file.exists():
        rows = json.loads(transition_file.read_text(encoding="utf-8"))
    else:
        rows = transition_rows_from_train(train_path)
        save_json(transition_file, rows)
    signals, selected_transitions = transition_maps(rows)

    exact, transition = geography_semantic_retrieve(
        benchmark_embeddings,
        query_embeddings,
        [q["search_location_id"] for q in queries],
        retrieval.locations,
        selected_transitions,
        bench_dir / "geo_semantic.npz",
    )

    builder = FeatureBuilder(retrieval, signals, benchmark_embeddings, corpus_path)
    ranker = lgb.Booster(model_file=str(final_model_path))

    rows_out = []
    for i, query in enumerate(queries):
        scores = geography_scores(retrieval, query, signals)
        bm_docs = top_k(scores, 500)
        channels = [bm_docs, global_docs[i], exact[i], transition[i]]

        # np.unique одновременно удаляет дубликаты между retrieval-каналами и
        # сортирует doc ids — это нужно для searchsorted в feature builder.
        docs = np.unique(np.concatenate(channels))
        docs = docs[docs >= 0]

        features = builder.make(query, query_embeddings[i], docs, channels, scores)
        prediction = ranker.predict(features, num_threads=4)
        selected = docs[np.lexsort((docs, -prediction))[:50]]

        rows_out.append({
            "query_id": query["query_id"],
            "answer": " ".join(retrieval.ids[selected]),
        })

        if (i + 1) % 250 == 0:
            print(f"BENCHMARK: {i + 1}/{len(queries)}", flush=True)

    result = validate_submission(rows_out, queries, retrieval.ids.tolist())
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "answer"])
        writer.writeheader()
        writer.writerows(rows_out)

    # Повторно читаем уже записанный файл: это ловит ошибки serialization/columns.
    with output_path.open(encoding="utf-8", newline="") as stream:
        reread = list(csv.DictReader(stream))
    result = validate_submission(reread, queries, retrieval.ids.tolist())
    result["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return result


# =============================================================================
# 9. End-to-end entry point
# =============================================================================

def run_pipeline(
    data_dir: Path = Path("data"),
    work_dir: Path = Path("artifacts/repro"),
    output_path: Path = Path("answer.csv"),
    model_path: str | None = None,
) -> dict:
    """Полный запуск от исходных parquet до answer.csv."""
    started = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    queries_path = data_dir / "benchmark_queries.parquet"
    items_path = data_dir / "benchmark_items.parquet"
    for path in (train_path, queries_path, items_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # --- Local corpus нужен только для обучения reranker. ---
    local_dir = work_dir / "local"
    local_dir.mkdir(exist_ok=True)
    local_corpus = local_dir / "corpus.parquet"
    make_unique_corpus(train_path, local_corpus)
    local_index = local_dir / "bm25_index"
    build_bm25_index(local_corpus, local_index)
    local_model = BM25(local_corpus, local_index)

    # Encoder создается один раз и используется и для local, и для benchmark.
    encoder = E5Encoder(model_path=model_path)
    local_embeddings = encode_corpus(encoder, local_corpus, local_dir / "embeddings.npy")

    final_model_path = prepare_ranker(
        train_path,
        local_corpus,
        local_index,
        local_embeddings,
        encoder,
        work_dir,
    )

    result = benchmark_inference(
        train_path,
        queries_path,
        items_path,
        local_model,
        local_embeddings,
        encoder,
        final_model_path,
        work_dir,
        output_path,
    )

    result["total_seconds"] = time.perf_counter() - started
    memory = psutil.Process().memory_info()
    result["working_set_mib"] = memory.rss / 2**20
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Avito candidate generation: hybrid retrieval + LightGBM")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/repro"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Локальная директория multilingual-e5-small. Если не указана, "
            "Transformers загрузит pinned revision модели из Hugging Face."
        ),
    )
    args = parser.parse_args()

    with threadpool_limits(limits=4):
        run_pipeline(args.data_dir, args.work_dir, args.output, args.model_path)


if __name__ == "__main__":
    main()
