# Dist Optimization Experiments

## High Level

- Dist speedup of ~4.5x on Russell (single GPU) and ~3.8x on local (internal timing)
- .ani outputs are preserved as an unordered row set; raw row order can change (at a high level, pipelining/scheduling on GPU provides the bulk of the speedup)
- GTDB ~113k validated with identical sorted SHA 256
- Accuracy validation against the BinDash baseline retained identical results

Note: local wall-clock timings from these runs were unreliable, so the local machine's numbers use the internal `dist_total_s` metric.

| Machine | Baseline | Optimized | Speedup | Timing source |
| --- | ---: | ---: | ---: | --- |
| Russell single GPU run 1 | 104.58s | 23.50s | 4.45x | wall clock |
| Russell single GPU run 2 | 103.78s | 23.11s | 4.49x | wall clock |
| Local (median of 3 runs) | 209.170s | 54.913s | 3.81x | dist_total_s |

## Timing

### Russell (single GPU)

Run on -T 128

Run 1:

| Phase | Wall (s) | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `1:44.58` | NA | NA |
| progress timing | `1:39.43` | `98.663` | `93.235` |
| stream breakdown | `1:39.93` | `99.170` | `93.764` |
| ref H2D cache | `1:34.36` | `93.622` | `88.040` |
| resident symmetric | `1:24.44` | `83.355` | `77.855` |
| pipeline postprocess | `0:23.50` | `22.434` | `16.964` |

Run 2:

| Phase | Wall (s) | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `1:43.78` | NA | NA |
| progress timing | `1:42.20` | `101.666` | `96.379` |
| stream breakdown | `1:40.72` | `100.208` | `94.871` |
| ref H2D cache | `1:34.58` | `93.858` | `88.261` |
| resident symmetric | `1:25.10` | `84.110` | `78.768` |
| pipeline postprocess | `0:23.11` | `22.531` | `16.859` |

vs baseline:

Run 1:
wall speedup = 104.58 / 23.50 = 4.45x
dist total speedup = 98.663 / 22.434 = 4.40x
stream speedup     = 93.235 / 16.964 = 5.50x
Run 2:
wall speedup = 103.78 / 23.11 = 4.49x
dist total speedup = 101.666 / 22.531 = 4.51x
stream speedup     = 96.379 / 16.859 = 5.72x

Stream here is the main ANI computation phase, where dist goes through comparison tiles, computes ANI, and writes output

### Russell (2x GPU)

Run on -T 128

Note: no wall clock, these are internal timings only

| `dist_total_s` | `stream_s` |
| --- | ---: |
| `15.702s` | `10.122s` |

vs single gpu run:
dist total speedup  = 22.434 / 15.702 = 1.43x
stream speedup = 16.964 / 10.122 = 1.68x


### Local

Median of 3 runs:
| Label | Wall (s) | `dist_total_s` | `stream_s` |
| --- | ---: | ---: | ---: |
| baseline | `203.000` | NA | NA |
| progress timing | `192.670` | `209.170` | `190.333` |
| stream breakdown | `192.640` | `208.646` | `190.907` |
| ref H2D cache | `175.790` | `189.076` | `170.642` |
| resident symmetric | `146.810` | `155.692` | `138.918` |
| pipeline postprocess | `51.160` | `54.913` | `37.184` |

vs baseline:
dist total speedup = 209.170 / 54.913 = 3.81x
stream speedup     = 190.333 / 37.184 = 5.12x


## Correctness

Optimized path should be judged by row identity and formatted ANI values, not by row order. .ani file is a set of pairwise results:

```text
reference_id<TAB>query_id<TAB>ani
```

The row-set check is:

```text
sort(old.ani) == sort(new.ani)
```

Pipelining changes when a tile's rows are written as output rows. It does not change the following core parts:

- dot product calculation
- ULL cardinality estimate
- ANI formula
- threshold behavior
- row format
- final set of rows that pass threshold in optimized path

Before postprocess pipelining, each tile (matrix tile is a block of pairwise comparisons) had a serial path: GPU dot product->CPU ANI/filter/format->output. \
After pipelining, this is separated: GPU workers can start later tiles while CPU workers format earlier ones.\
Because tiles now finish independently though, output order of rows can be different, but row set and the actual ANI values are unchanged.\
This is why the raw output is not byte identical anymore, but row order isn't actually important for the .ani result; when we sort the rows we get the same SHA256 hash.

### 113k GTDB Row-Set Validation

Rows were sorted and hashed with:
```text
LC_ALL=C sort -T "$OUT/sort_tmp" "$ani_file" | sha256sum
```

| Check | Baseline | Optimized |
| --- | ---: | ---: |
| Lines | `17429080` | `17429080` |
| Bytes | `3433528760` | `3433528760` |
| Raw byte-identical | yes | no |
| Sorted row-set match | yes | yes |
| Sorted SHA-256 | `97650baa2fcbfd3fbf5a1662e9cc6e64425fe6eef8183d1214f00ba2923bc1d0` | `97650baa2fcbfd3fbf5a1662e9cc6e64425fe6eef8183d1214f00ba2923bc1d0` |

### Accuracy Validation

The optimized output must also retain the same accuracy against the truth set from BinDash:

```text
baseline lines=120
optimized lines=120
baseline bytes=18195
optimized bytes=18195
byte_identical=0
sorted baseline sha256=bf03596a21564e53d672bae40a79e880607c99b64fa4e58f943af0a4a64ad2ea
sorted optimized sha256=bf03596a21564e53d672bae40a79e880607c99b64fa4e58f943af0a4a64ad2ea
```

Recomputed accuracy was byte identical to the baseline

```text
pair_count=120
missing_pairs=0
extra_pairs=0
mean_absolute_error=0.0007131666667
max_absolute_error=0.00171
```

So our optimized dist is performance uplift only; output order changed but pair results and accuracy did not.



## Optimizations

The work was done in phases:

1. Progress and timing
2. GPU stream breakdown (a threshold prefilter attempted in this phase was reverted)
3. Reference host-to-device cache
4. Resident symmetric GPU matrix
5. Pipelined GPU compute and CPU postprocessing

### Progress and Timing

- Added stage timing to see where to optimize; this phase also fixes the problem of the progress bar not appearing until dist was nearly complete
- Overall, this phase fixed progress/output handling so completed tile results were received, written, and counted while GPU workers were still running rather than after

Metrics:

- ULL load
- sketch load
- validation
- decompression
- compute/write
- total time
- `compute_hv_ani` stream time

Baselines:

| Env | `dist_total` | `stream` |
| --- | ---: | ---: |
| Russell single GPU | `98.663s` | `93.235s` |
| Local median | `209.170s` | `190.333s` |

### Stream Breakdown and Prefilter (reverted)

- Added GPU stream breakdown with more detailed stage metrics, including flattening, transfer, GPU tile work, postprocess, write, and wall time.
- Also attempted was a threshold prefilter, where the idea was to skip pairs that did not meet the threshold from ANI calculation. This didn't work though, and ended up dropping 90 rows on the default ANI threshold of 85.

After removal:

```text
candidates = processed pairs
prefilter_skipped = 0
final filter = existing ani >= ani_threshold
```

### Reference Host to Device (CPU -> GPU) Cache

- Reduced repeated host to device uploads of reference tiles
- Reference tiles in this context: for pairwise comparison; sketch 1 x sketch 2 -> reference x query
- For a fixed reference tile, we compare to many query tiles, so we can use the same reference tile over again
- Previously, we transferred the same tile to the GPU repeatedly, now reference tile is only uploaded to GPU on cache miss

Added more stage timing metrics:

```text
query_h2d_ns
ref_h2d_ns
compute_d2h_ns
total_ns
query_h2d_bytes
ref_h2d_bytes
out_d2h_bytes
ref_upload_performed
```

Slight performance uplift, but is mainly useful for later optimizations

New bottleneck picture:

| Metric | Time (s) |
| --- | ---: |
| `ref_h2d` | `0.175s` |
| `query_h2d` | `20.5s` |
| `flatten_query` | `18-21s` |
| `compute_d2h` | `52s` |
| `postprocess` | `76-77s` |

### (GPU) Resident Symmetric Matrix

Resident here means data is transferred to GPU once and then stays in GPU memory for reuse, so the sketch matrix becomes resident on the GPU

- Comparison in dist is self comparison, coming from the sketch file (reference sketch x query sketch)
- Thus the reference and query sketches are the same matrix, and dist is just filling in a large pairwise ANI matrix when comparing the two
- Because the matrix is symmetric, ANI(A, B) == ANI(B, A), so from matrix symmetry we know that we only have to compute the upper triangle of the matrix
- The lower triangle is duplicate work; diagonal is our self comparison

Previously, each tile prepared and transferred query blocks ("block" is a group of sketch rows/genomes), even though these query blocks come from the same sketch matrix as the reference

Optimization is as follows:

```text
Before:
  for each tile:
    prep reference block
    prep query block
    transfer blocks to GPU
    compute tile
```
```text
After:
  transfer full sketch matrix to GPU one time

  for each tile in upper triangle:
    use relevant row range from GPU matrix for the reference block
    use relevant row range from GPU matrix for the query block
    compute tile
```

Thus, we reduce repeated sketch query prep and CPU->GPU transfer time

| Metric | Before (s) | After (s) |
| --- | ---: | ---: |
| query flattening | 18-21s | 0s |
| query H2D upload | 20.5s | 0s |
| ref H2D upload | 0.175s | 0s |

This path is conservative: it only runs on symmetric self comparisons, checks that there is enough GPU memory, and falls back to the previous tiled transfer path if the resident transfer path can't be used

### Pipeline GPU and CPU Postprocess

At this stage the visible bottlenecks were:

```text
compute_d2h ~53-55s
postprocess ~80-82s
write ~4-5s
```
Or by what's actually happening:
```text
GPU dot-product / GPU-to-CPU output copy: ~53-55s
CPU ANI/filter/format work: ~80-82s
Output writing: ~4-5s
```

No single stage is slow, rather we're wasting time since the stages are serial:
```text
for each tile:
    compute GPU dot products for tile
    copy results back to CPU
    convert dot products to ANI values
    filter and format output rows
    write results
```

This can be pipelined:

```text
Before:
    GPU work -> CPU formatting -> write -> next tile
```
```text
After:
    GPU work for tile N+1 can happen independently while CPU does work on tile N
```
The performance benefit comes from overlap:
- GPU computes dot products
- CPU workers convert completed tiles into ANI rows
- Completed output is written

This lets the GPU start computing later tiles while the CPU is working on earlier tiles. The ANI formula, threshold check, and output format did not change; only scheduling work changed.

However, this does change raw output order. Since tiles now complete independently, rows can be written in a different order than the serial traversal. We can verify it's still correct by sorting the .ani rows and confirming the row sets are identical

Also added backpressure metrics (see if one stage of the pipeline is slowing down an earlier stage)

| Metric | Means |
| --- | --- |
| `postprocess_workers` | # of CPU workers formatting GPU tile results |
| `gpu_send_blocked` | How long GPU workers waited because the CPU postprocess queue was full |
| `postprocess_worker_sum` | Total CPU postprocess time across workers |
| `postprocess_result_send_blocked` | How long CPU workers waited because the output writer queue was full |

Testing showed low `gpu_send_blocked`, meaning backpressure was not a main bottleneck:

| Run | `gpu_send_blocked` |
| --- | ---: |
| Server single GPU | `0.042s` |
| Server two GPU | `0.989s` |
| Local median | `0.649s` |

In the pipelined path, the `-T` flag sizes the CPU postprocessing thread pool:

```text
postprocess_workers = min(clamp(-T, 1, 128), total_jobs)
```

Note: if `gpu_send_blocked` is low, increasing `-T` (and therefore postprocess workers) is unlikely to give much further speedup

## Limitations and Notes

- No full dist run on 900k GTDB (~63.4x pair count of small GTDB)
- Row order is no longer identical to the serial path; correctness is judged by sorted row-set identity (see Correctness)
