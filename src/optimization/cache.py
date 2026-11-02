"""Thread-safe in-memory allocation cache for prescriptive discount optimization queries."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
import time
from typing import Any, Dict, Optional, Union
import numpy as np

from src.optimization.base import OptimizationResult


@dataclass(frozen=True)
class _CacheEntry:
    """Internal cache entry storing optimization result with ingestion timestamp."""

    result: OptimizationResult
    created_at: float


class AllocationCache:
    """Thread-safe LRU-style cache with TTL expiration for OptimizationResult objects.

    Caches solver outcomes keyed by a deterministic SHA-256 digest of input parameters:
    (treatment_effects, treatment_costs, customer_values, budget, objective).
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0):
        if max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got: {max_size}")
        if ttl_seconds <= 0.0:
            raise ValueError(f"ttl_seconds must be positive, got: {ttl_seconds}")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def compute_key(
        self,
        treatment_effects: np.ndarray,
        treatment_costs: np.ndarray,
        customer_values: np.ndarray,
        budget: float,
        objective: Union[str, Any],
    ) -> str:
        """Compute deterministic SHA-256 hash key from solver input parameters.

        Parameters:
            treatment_effects: Array of treatment effect estimates (uplift).
            treatment_costs: Array of promotional discount costs.
            customer_values: Array of customer baseline values (AOV or CLV).
            budget: Campaign budget ceiling.
            objective: Optimization objective enum or string name.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        te = np.ascontiguousarray(np.asarray(treatment_effects, dtype=np.float64))
        tc = np.ascontiguousarray(np.asarray(treatment_costs, dtype=np.float64))
        cv = np.ascontiguousarray(np.asarray(customer_values, dtype=np.float64))
        obj_name = objective.value if hasattr(objective, "value") else str(objective)

        hasher = hashlib.sha256()
        hasher.update(te.tobytes())
        hasher.update(tc.tobytes())
        hasher.update(cv.tobytes())
        hasher.update(f"{float(budget):.6f}".encode("utf-8"))
        hasher.update(obj_name.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[OptimizationResult]:
        """Retrieve cached OptimizationResult by key, returning None on miss or TTL expiry.

        Parameters:
            key: SHA-256 cache key.

        Returns:
            Cached OptimizationResult or None.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            now = time.monotonic()
            if (now - entry.created_at) > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Promote to most recently used
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.result

    def set(self, key: str, result: OptimizationResult) -> None:
        """Store OptimizationResult under key, evicting oldest item if max_size exceeded.

        Parameters:
            key: SHA-256 cache key.
            result: OptimizationResult to store.
        """
        with self._lock:
            now = time.monotonic()
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = _CacheEntry(result=result, created_at=now)

    def clear(self) -> None:
        """Clear all entries and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> Dict[str, Any]:
        """Return operational telemetry metrics for the cache layer."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hit_ratio": hit_ratio,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if (time.monotonic() - entry.created_at) > self.ttl_seconds:
                del self._cache[key]
                return False
            return True

