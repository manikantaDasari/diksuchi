"""
centroid_classifier.py — Embedding-based prompt capability classifier for Diksuchi.

How it works:
  1. SETUP (once at startup): For each capability bucket defined in config.yaml,
     embed all example prompts using all-MiniLM-L6-v2 (22 MB) and average the
     vectors → one "centroid" per bucket. Cache to disk so restarts are instant.

  2. RUNTIME (~8ms per request): Embed the incoming prompt → compute cosine
     similarity against every centroid → assign to highest-similarity bucket.

No training, no GPU, no retraining when new models release.
To add a bucket: add examples to config.yaml → restart → done.

Graceful degradation: if sentence-transformers is not installed, falls back
to keyword scoring automatically (no crash, just less accurate).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("diksuchi")

# ── Optional dependency ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

CENTROIDS_CACHE = Path(__file__).parent / ".centroid_cache.json"
MODEL_NAME = "all-MiniLM-L6-v2"   # 22 MB, ~5ms CPU inference


# ─── Classifier ──────────────────────────────────────────────────────────────

class CentroidClassifier:
    """
    Classifies prompts into capability buckets via cosine similarity.

    Config example (config.yaml → routing.centroid_buckets):
        simple:
          tier: 0
          examples: ["what is X", "define this", "translate to French"]
        coding:
          tier: 0
          examples: ["write a function", "fix this bug", "add type hints"]
        reasoning:
          tier: 1
          examples: ["step by step", "explain why", "analyze this"]
        creative:
          tier: 1
          examples: ["write a blog post", "draft an email"]
        critical:
          tier: 2
          examples: ["security audit", "production code review"]
    """

    def __init__(self, buckets: dict):
        """
        buckets: dict parsed from config.yaml centroid_buckets section.
        """
        self.buckets = buckets
        self._model: Optional[object] = None
        self._centroids: Optional[dict] = None
        self._ready = False

        if _ST_AVAILABLE and buckets:
            self._init()
        elif not _ST_AVAILABLE:
            log.warning(
                "⚠️  sentence-transformers not installed. "
                "Run: pip install sentence-transformers  "
                "Using keyword fallback in the meantime."
            )

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        t0 = time.perf_counter()
        try:
            self._model = SentenceTransformer(MODEL_NAME)
            self._centroids = self._load_or_compute_centroids()
            self._ready = True
            ms = (time.perf_counter() - t0) * 1000
            log.info(
                "🧠  Centroid classifier ready — %d buckets loaded in %.0fms",
                len(self.buckets), ms,
            )
        except Exception as exc:
            log.warning("⚠️  Centroid classifier init failed: %s — using keyword fallback", exc)

    def _cache_key(self) -> str:
        """Fingerprint of current bucket config — used to detect changes."""
        return str(sorted(
            (k, str(sorted(v.get("examples", []))))
            for k, v in self.buckets.items()
        ))

    def _compute_centroids(self) -> dict:
        """
        Embed all example prompts per bucket, average → centroid, normalize.
        Returns {bucket_name: unit_vector_as_list}.
        """
        import numpy as np
        centroids = {}
        for name, cfg in self.buckets.items():
            examples = cfg.get("examples", [])
            if not examples:
                log.warning("  Bucket %r has no examples — skipping", name)
                continue
            vecs = self._model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[name] = (centroid / norm if norm > 0 else centroid).tolist()
            log.debug("  Bucket %r: %d examples embedded", name, len(examples))
        return centroids

    def _load_or_compute_centroids(self) -> dict:
        """Load cached centroids if config unchanged, else recompute."""
        import numpy as np
        key = self._cache_key()

        if CENTROIDS_CACHE.exists():
            try:
                cached = json.loads(CENTROIDS_CACHE.read_text())
                if cached.get("cache_key") == key:
                    log.debug("📦  Centroid cache hit — skipping recompute")
                    return {k: np.array(v) for k, v in cached["centroids"].items()}
            except Exception:
                pass   # corrupt cache — just recompute

        log.info("⚙️   Computing centroid embeddings (first run or config changed)…")
        raw = self._compute_centroids()

        try:
            CENTROIDS_CACHE.write_text(json.dumps({"cache_key": key, "centroids": raw}))
        except Exception as exc:
            log.debug("Could not persist centroid cache: %s", exc)

        return {k: np.array(v) for k, v in raw.items()}

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, prompt: str) -> tuple[str, float]:
        """
        Returns (bucket_name, confidence_score ∈ [0, 1]).
        Falls back to keyword scoring if model isn't available.
        """
        if self._ready and self._centroids:
            return self._embed_classify(prompt)
        return self._keyword_fallback(prompt)

    def get_tier(self, bucket: str) -> int:
        """Return the routing tier for a bucket: 0=local, 1=openrouter_free, 2=paid."""
        return self.buckets.get(bucket, {}).get("tier", 0)

    def recompute(self) -> None:
        """Force centroid recomputation — call after editing bucket examples."""
        if CENTROIDS_CACHE.exists():
            CENTROIDS_CACHE.unlink()
        if _ST_AVAILABLE and self.buckets:
            self._init()

    # ── Internal classifiers ──────────────────────────────────────────────────

    def _embed_classify(self, prompt: str) -> tuple[str, float]:
        """Embed prompt → cosine similarity against all centroids → argmax."""
        import numpy as np
        vec = self._model.encode([prompt], convert_to_numpy=True, show_progress_bar=False)[0]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return "simple", 0.0
        vec = vec / norm

        scores: dict[str, float] = {
            name: float(np.dot(vec, centroid))
            for name, centroid in self._centroids.items()
        }
        best = max(scores, key=scores.__getitem__)
        return best, round(scores[best], 3)

    def _keyword_fallback(self, prompt: str) -> tuple[str, float]:
        """
        Keyword-based fallback when sentence-transformers is unavailable.
        Ordered from most-specific to least-specific so critical wins over coding.
        """
        p = prompt.lower()

        tiers: list[tuple[str, list[str]]] = [
            ("critical", [
                "security audit", "full audit", "production code review",
                "code review", "architecture decision", "system design",
                "comprehensive refactor", "entire codebase", "medical diagnosis",
                "legal advice", "full refactor", "performance bottleneck",
            ]),
            ("reasoning", [
                "step by step", "explain why", "analyze this", "compare and contrast",
                "why does", "pros and cons", "trade-offs", "debug this entire",
                "detailed research", "evaluate", "in-depth", "stack trace",
            ]),
            ("coding", [
                "write a function", "fix this bug", "fix the bug", "implement",
                "add type hints", "add docstring", "write unit tests", "write tests",
                "refactor", "what does this function", "what does this code",
                "add a comment", "write a class", "fix indentation",
            ]),
            ("creative", [
                "write a blog", "draft an email", "write a story", "write an article",
                "rewrite this", "product description", "write a post",
                "write a poem", "summarize this", "rephrase",
            ]),
            ("simple", [
                "what is", "who is", "where is", "when did", "define",
                "translate", "tldr", "quick question", "short answer",
                "one sentence", "what does", "how many", "fix typo",
            ]),
        ]

        for bucket, keywords in tiers:
            if any(kw in p for kw in keywords):
                return bucket, 0.65   # fixed confidence for keyword match

        return "simple", 0.30   # safe default


# ─── Module-level singleton ───────────────────────────────────────────────────

_classifier: Optional[CentroidClassifier] = None


def init_classifier(buckets: dict) -> CentroidClassifier:
    """Initialize the global classifier from config.yaml centroid_buckets."""
    global _classifier
    _classifier = CentroidClassifier(buckets)
    return _classifier


def classify_prompt(prompt: str) -> tuple[str, float]:
    """
    Classify a prompt into a capability bucket.
    Returns ("simple", 0.0) if the classifier was never initialized.
    """
    if _classifier is None:
        return "simple", 0.0
    return _classifier.classify(prompt)


def get_classifier_tier(bucket: str) -> int:
    """Return the routing tier for a classified bucket."""
    if _classifier is None:
        return 0
    return _classifier.get_tier(bucket)
