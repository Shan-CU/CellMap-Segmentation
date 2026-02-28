# 3D Training Host-RAM OOM: Root Cause & Fix

## TL;DR

**Problem:** 3D training jobs were OOM-killed at ~400 GB host RAM within a
single epoch despite only using ~2 GB of model parameters.

**Root cause:** Each `CellMapImage` in `cellmap-data` calls
`tensorstore.open(context=None)`, which creates an **independent, unbounded**
LRU chunk cache. With 261 training datasets × ~49 label arrays each ≈ 12,789
independent caches, each growing without eviction, host RAM balloons past
400 GB in under 100 training steps.

**Fix:** Upgrade `cellmap-data` to `>= 2026.2.27`, which adds a built-in
`tensorstore_cache_bytes` parameter to `CellMapDataLoader` (default: 2 GiB).
The library creates a single shared `ts.Context({"cache_pool":
{"total_bytes_limit": N}})` and recursively sets it on every `CellMapImage`
before worker processes are spawned. All 12,789 arrays share one bounded
cache pool with LRU eviction.

---

## Timeline of Discovery

| Date       | Event |
|------------|-------|
| 2026-02-24 | All five 3D Phase 2 jobs OOM-killed at ~512 GB within 30 min |
| 2026-02-24 | **Wrong hypothesis #1:** blamed `persistent_workers=True` + `CellMapDataLoader.refresh()` — thought old DataLoader workers were accumulating cache across epochs |
| 2026-02-25 | Canary job with `--persistent_workers false` **still OOM'd** at 402 GB RSS at step 81/300 of epoch 1 — proved the per-worker theory wrong |
| 2026-02-26 | **Real root cause found:** `CellMapImage._ts_store` calls `ts.open(context=None)` → each array gets its own unbounded chunk cache. 261 datasets × 49 arrays = ~12,789 independent caches with no eviction |
| 2026-02-27 | **First fix attempt (broken):** Manual `ts.Context` in `train.py`, passed as `extra_dl_kwargs["context"]` to `get_dataloader()`. But `get_dataloader()` passes `**kwargs` to `CellMapDataLoader`, NOT to `CellMapDataSplit` — so the context never reached the images |
| 2026-02-27 | **Discovered upstream fix:** `cellmap-data` main branch (2026.2.27) has `tensorstore_cache_bytes` built into `CellMapDataLoader.__init__` with proper recursive `_set_tensorstore_context()`. Upgraded both `cellmap-data` and `cellmap-segmentation-challenge` from git main |
| 2026-02-27 | Removed broken manual `ts.Context` code from `train.py`. Library handles it properly now |

---

## Technical Deep Dive

### Why did it OOM?

TensorStore (the C++ array library underlying cellmap-data) maintains an
in-process chunk cache for each opened array. When you call:

```python
ts.open(spec, read=True, write=False, context=None)
```

...the `context=None` means "create a new default context" — which includes a
**completely independent** chunk cache with **no byte limit**. The cache only
evicts when the process exits or the TensorStore handle is garbage-collected.

In the CellMap challenge:
- 261 training datasets (each a zarr volume on S3)
- Each dataset has ~49 arrays (raw EM + 48 label classes across scales)
- = **~12,789 independent unbounded caches**

As the training loop randomly samples crops from all datasets, each cache
fills with chunks that are never evicted. By step ~80 of the first epoch the
aggregate cache exceeds the SLURM memory limit (384–768 GB).

### Why `persistent_workers=False` didn't help

The wrong hypothesis was: "worker processes accumulate cache; killing them on
`refresh()` releases it." In reality:

1. `CellMapDataLoader.refresh()` creates a new `torch.DataLoader` each epoch
2. But the `CellMapDataSplit` → `CellMapDataset` → `CellMapImage` objects
   live in the **main process** (they're passed by reference to workers via
   fork on Linux)
3. The TensorStore chunk caches are inside `CellMapImage._ts_store`, which
   persists in the main process regardless of worker lifecycle
4. `persistent_workers=False` only prevents workers from keeping their own
   copy of cached data — the main process caches are unaffected

### How the upstream fix works

`cellmap-data >= 2026.2.27` adds:

1. **`CellMapDataLoader.__init__(tensorstore_cache_bytes=N)`** — creates a
   single shared `ts.Context({"cache_pool": {"total_bytes_limit": per_worker}})` 
   where `per_worker = N // max(1, num_workers)`

2. **`_set_tensorstore_context(dataset, context)`** — recursively walks the
   dataset tree (`CellMapMultiDataset` → `CellMapSubset` → `CellMapDataset`
   → `CellMapImage`) and sets `image.context = bounded_context` on every
   image **before** workers are spawned

3. **`CellMapImage._ts_store`** — calls `ts.open(spec, context=self.context)`,
   so the bounded pool is used when the array is first opened

4. **`CELLMAP_TENSORSTORE_CACHE_BYTES` env var** — fallback if the constructor
   parameter is not set. Default: 2 GiB.

All 12,789 arrays now share one bounded LRU cache pool. When the pool fills,
TensorStore evicts the least-recently-used chunks automatically.

### Our configuration

| Setting | 2D jobs | 3D jobs |
|---------|---------|---------|
| `CELLMAP_TENSORSTORE_CACHE_BYTES` | *(not set → 2 GiB default)* | *(not set → 2 GiB default)* |
| `--persistent_workers` | `auto` (PyTorch default) | `auto` (PyTorch default) |
| `--mem` (SLURM) | 64G | 128G |

The 2 GiB default is ample for both 2D and 3D. No env var override needed.

---

## Files Changed

| File | Change |
|------|--------|
| `training/train.py` | Removed broken manual `ts.Context` hack; added comments pointing to library fix |
| `training/slurm/phase2_3d_l40s.sbatch` | Updated comments explaining `CELLMAP_TENSORSTORE_CACHE_BYTES` |
| `pyproject.toml` *(recommended)* | Pin `cellmap-data >= 2026.2.27` to prevent regression |

## Package Versions

```
cellmap-data==2026.2.27.2238           # from git+https://github.com/janelia-cellmap/cellmap-data.git@main
cellmap-segmentation-challenge==0.0.1  # from git+https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git@main
```

---

## How to Reproduce the Bug (DON'T)

If you ever need to verify the fix is working, check the training logs for:

```
TensorStore cache bounded: total=536870912 bytes / 4 worker(s) = 134217728 bytes each
```

If you see the OOM again, check:
1. `pip show cellmap-data` — must be `>= 2026.2.27`
2. `CELLMAP_TENSORSTORE_CACHE_BYTES` env var — must be set for 3D jobs
3. The `CellMapDataLoader` log output — should show the cache bound message

To monitor RSS during training:
```bash
while true; do ps -o pid,rss,comm -p $(pgrep -f "training.train") | awk '{print strftime("%T"), $0}'; sleep 30; done
```

---

## Lessons Learned

1. **TensorStore's default context is unbounded** — this is not documented
   prominently. Any code doing `ts.open(context=None)` on many arrays will
   eventually OOM. Always create a shared `ts.Context` with a cache limit.

2. **The real culprit hid behind the obvious suspect** — `persistent_workers`
   and `refresh()` looked guilty but were innocent. The actual leak was in
   C++ memory (TensorStore's native chunk cache), invisible to Python's
   `gc.collect()` and `tracemalloc`.

3. **Check upstream before writing hacks** — the `cellmap-data` team had
   already implemented the proper fix (`tensorstore_cache_bytes` with
   recursive context setting). Our manual `ts.Context` bypass didn't even
   work because `get_dataloader()` doesn't pass `**kwargs` to
   `CellMapDataSplit`.

4. **The env var `CELLMAP_TENSORSTORE_CACHE_BYTES` was present in cellmap-data
   v2026.2.20 on `CellMapDataSplit` but NOT on `CellMapDataLoader`** — the
   constructor parameter, the recursive `_set_tensorstore_context()`, and the
   `_DEFAULT_TENSORSTORE_CACHE_BYTES` constant were all added in v2026.2.27.
   Subtle version differences matter.
