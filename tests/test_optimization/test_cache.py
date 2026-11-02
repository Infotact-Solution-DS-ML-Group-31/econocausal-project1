"""Unit tests for in-memory allocation cache layer and solver integration."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import OptimizationObjective
from src.optimization.base import OptimizationResult
from src.optimization.cache import AllocationCache
from src.optimization.solver import SciPyConstrainedSolver


class TestAllocationCache(unittest.TestCase):
    """Test suite verifying cache operations, LRU eviction, TTL expiry, and solver caching."""

    def setUp(self):
        self.mock_result = OptimizationResult(
            success=True,
            status_message="Optimization succeeded",
            total_spend=30.0,
            expected_incremental_revenue=65.0,
            allocations=np.array([1, 1, 0, 1, 0], dtype=np.int32),
            metadata={"targeted_count": 3, "expected_net_profit": 35.0},
        )
        self.te = np.array([0.25, 0.30, -0.05, 0.20, 0.05])
        self.tc = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.cv = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        self.budget = 30.0
        self.objective = OptimizationObjective.MAXIMIZE_PROFIT

    def test_compute_key_deterministic_and_unique(self):
        """Verify compute_key generates deterministic SHA-256 hashes and distinguishes distinct inputs."""
        cache = AllocationCache()
        key1 = cache.compute_key(self.te, self.tc, self.cv, self.budget, self.objective)
        key2 = cache.compute_key(self.te, self.tc, self.cv, self.budget, self.objective)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)  # 64-char hex SHA-256

        # Different budget -> different key
        key_diff_budget = cache.compute_key(self.te, self.tc, self.cv, 40.0, self.objective)
        self.assertNotEqual(key1, key_diff_budget)

        # Different objective -> different key
        key_diff_obj = cache.compute_key(self.te, self.tc, self.cv, self.budget, "maximize_revenue")
        self.assertNotEqual(key1, key_diff_obj)

        # Different ITE -> different key
        te_diff = self.te.copy()
        te_diff[0] += 0.01
        key_diff_te = cache.compute_key(te_diff, self.tc, self.cv, self.budget, self.objective)
        self.assertNotEqual(key1, key_diff_te)

    def test_cache_hits_misses_and_stats(self):
        """Verify get/set operations, hit/miss tracking, and stats reporting."""
        cache = AllocationCache(max_size=10, ttl_seconds=60.0)
        key = cache.compute_key(self.te, self.tc, self.cv, self.budget, self.objective)

        # Initial get -> cache miss
        self.assertIsNone(cache.get(key))
        stats = cache.stats()
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["size"], 0)

        # Set entry
        cache.set(key, self.mock_result)
        self.assertEqual(len(cache), 1)
        self.assertIn(key, cache)

        # Retrieve entry -> cache hit
        retrieved = cache.get(key)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.total_spend, self.mock_result.total_spend)
        self.assertEqual(
            retrieved.expected_incremental_revenue,
            self.mock_result.expected_incremental_revenue,
        )

        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hit_ratio"], 0.5)

    def test_cache_lru_eviction(self):
        """Verify least-recently-used item is evicted when max_size capacity is reached."""
        cache = AllocationCache(max_size=2, ttl_seconds=60.0)

        cache.set("key_1", self.mock_result)
        cache.set("key_2", self.mock_result)

        # Access key_1 to make key_2 the least recently used
        _ = cache.get("key_1")

        # Insert key_3 -> key_2 should be evicted
        cache.set("key_3", self.mock_result)

        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("key_2"))
        self.assertIsNotNone(cache.get("key_1"))
        self.assertIsNotNone(cache.get("key_3"))

        stats = cache.stats()
        self.assertEqual(stats["evictions"], 1)

    def test_cache_ttl_expiration(self):
        """Verify cached entries expire after ttl_seconds elapses."""
        cache = AllocationCache(max_size=10, ttl_seconds=0.05)
        key = "short_lived_key"
        cache.set(key, self.mock_result)

        # Immediately available
        self.assertIsNotNone(cache.get(key))

        # Wait for TTL expiration
        time.sleep(0.06)

        # Should expire and return None
        self.assertIsNone(cache.get(key))
        self.assertNotIn(key, cache)
        self.assertEqual(len(cache), 0)

    def test_cache_clear(self):
        """Verify clear method empties cache and resets operational telemetry."""
        cache = AllocationCache(max_size=10)
        cache.set("k1", self.mock_result)
        _ = cache.get("k1")
        _ = cache.get("k_miss")

        cache.clear()
        self.assertEqual(len(cache), 0)
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["size"], 0)

    def test_solver_integration_caching(self):
        """Verify SciPyConstrainedSolver uses cache on repeated queries and marks metadata['cache_hit']."""
        cache = AllocationCache(max_size=10, ttl_seconds=60.0)
        solver = SciPyConstrainedSolver(budget=30.0, cache=cache)

        # First pass: Cache miss
        res1 = solver.solve(
            treatment_effects=self.te,
            treatment_costs=self.tc,
            customer_values=self.cv,
        )
        self.assertTrue(res1.success)
        self.assertFalse(res1.metadata.get("cache_hit", False))
        self.assertEqual(cache.stats()["misses"], 1)
        self.assertEqual(cache.stats()["hits"], 0)

        # Second pass: Cache hit
        res2 = solver.solve(
            treatment_effects=self.te,
            treatment_costs=self.tc,
            customer_values=self.cv,
        )
        self.assertTrue(res2.success)
        self.assertTrue(res2.metadata.get("cache_hit", False))
        self.assertEqual(cache.stats()["hits"], 1)

        # Results must be identical
        self.assertEqual(res1.total_spend, res2.total_spend)
        self.assertEqual(
            res1.expected_incremental_revenue,
            res2.expected_incremental_revenue,
        )
        np.testing.assert_array_equal(res1.allocations, res2.allocations)

    def test_solver_without_cache(self):
        """Verify SciPyConstrainedSolver functions identically when cache is None."""
        solver = SciPyConstrainedSolver(budget=30.0, cache=None)
        res = solver.solve(
            treatment_effects=self.te,
            treatment_costs=self.tc,
            customer_values=self.cv,
        )
        self.assertTrue(res.success)
        self.assertNotIn("cache_hit", res.metadata)

    def test_thread_safety_concurrent_access(self):
        """Verify AllocationCache remains consistent under multithreaded concurrent read/write load."""
        cache = AllocationCache(max_size=50, ttl_seconds=10.0)

        def worker(thread_id: int):
            for i in range(20):
                k = f"key_{thread_id % 5}_{i % 5}"
                cache.set(k, self.mock_result)
                _ = cache.get(k)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, tid) for tid in range(8)]
            for f in futures:
                f.result()

        stats = cache.stats()
        self.assertGreater(stats["hits"], 0)
        self.assertLessEqual(stats["size"], 50)


if __name__ == "__main__":
    unittest.main()

