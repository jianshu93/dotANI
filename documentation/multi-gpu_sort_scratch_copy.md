# CUDA Multi-GPU Lanes, Scratch Reuse, And Sort-Path Copy Removal

## High Level

- Multi-GPU sketching pipeline: CPU workers feed the CUDA devices
- Each lane processes one file at a time, no batching
- Each lane has reusable CUDA context/module/stream and scratch across files
- `sort_unstable` is the faster dedup path (explained below) and the default; `hashset` remains selectable via `--cuda-dedup hashset`
- `sort_unstable` + ULL avoids an extra vector copy by sorting and deduplicating `full_hashes` in place after ULL consumes the hash stream

Sketch output is unchanged; end-to-end CPU/CUDA parity tests verify byte-identical outputs.

## Multi-GPU lanes; CPU workers

Tying each CPU worker to a single GPU underutilizes the devices (GPU utilization was observed at roughly 4-10%). Instead, the scheduling unit is the CPU worker (lane) itself:

```text
worker_count = min(-T, number_of_files), at least 1
```

Each lane dynamically takes the next file index:

```text
next_file.fetch_add(1, Ordering::Relaxed)
```

Metrics use `cuda_stream_lane` for the worker id ("lane" and "worker" are the same thing: the CPU-side CUDA worker). With multiple GPUs, lanes are assigned to devices evenly in repeating order, so many CPU workers are distributed across all visible GPUs:

```text
device = visible_devices[worker_id % visible_devices.len()]
```

Output order is unchanged. Each lane returns the original input index, and results are stored by input index so they can be written in input order after all workers finish.

## Sort Unstable

After k-mer hashing, the CPU holds the hash stream in a contiguous `Vec<u64>` (hashes next to each other in memory). ULL needs the full hash stream, and HD encoding needs the unique deduplicated hashes. The standard way to obtain unique hashes is a `HashSet`:

```text
for hash in hashes:
    set.insert(hash)
```

This is expensive on the full workload: many hash-table insertions, table growth, and non-contiguous memory access. The `sort_unstable` strategy instead deduplicates the existing vector in place rather than building a separate hash table:

```rust
hashes.sort_unstable();
hashes.dedup();
```

`sort_unstable` (the Rust function) sorts the vector in place without preserving the original order. Order does not matter here: equal hashes are identical, the next step removes duplicates, and the deduplicated hash list sent to the GPU for HD encoding does not depend on insertion order. Sorting groups equal hashes next to each other, and `dedup()` removes the duplicates:

```text
before sort:
[9, 2, 9, 5, 2, 7]

after sort:
[2, 2, 5, 7, 9, 9]

after dedup:
[2, 5, 7, 9]
```

This gives the same unique hash set as `HashSet` without building a separate hash table. It is entirely CPU-side and does not change anything around the k-mer hash algorithm; it only changes how the CPU prepares the unique hash list that is sent to the GPU for HD encoding. GPU-fed lanes must prepare deduplicated hash lists quickly: with many CPU workers feeding the GPUs, slow dedup leaves the GPUs waiting on the CPU.

Runs using `sort_unstable` produce outputs identical to the `HashSet` strategy; end-to-end byte-parity tests cover both strategies. `hashset` remains available for comparison or debugging:

```sh
--cuda-dedup hashset
```

`sort_unstable` is the default:

```sh
--cuda-dedup sort_unstable
```

### Use of `sort_unstable`

In the GPU path, ULL needs the full hash stream and HD needs the unique deduplicated hashes. `sort_unstable` is only used for the HD dedup input; it does not prevent ULL from seeing the full stream.

With ULL enabled, the order is as follows:

```text
full_hashes = CUDA k-mer hash output

for h in full_hashes:
    ull.add(h)

sort_unstable(full_hashes)
dedup(full_hashes)

HD(full_hashes as unique hashes)
```

## Scratch Reuse (per Lane)

Constructing CUDA resources and scratch per file adds wasted allocations and repeated dedup setup. Each lane processes many files, so reusable state lives at the lane level. Each lane owns:

- CUDA context/module/stream
- Cached function handles
- Reusable vectors and buffers
- Reusable hash set where applicable

## Copy Removal

The `sort_unstable` path removes a host copy: with ULL enabled, the full hash vector is sorted and deduplicated in place after ULL consumes the stream, so no separate unique-hash vector is built or copied before HD encoding.
