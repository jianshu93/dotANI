# Changes vs upstream (dotANI) and findings

## Changes

High Level:
- Added mode selection (cpu vs cuda) to the CLI for sketching
- Stage timing metrics
- Verification to make sure CPU and CUDA runs have the same output
- CUDA now performs both kmer hashing and HD encode on GPU (existing `.sketch` and `.ull` format preserved) for ~5x speedup
- CUDA sketching now supports multiple host lanes distributed across multi GPU if available (multi GPU new as of 5/14 update)
- CUDA sketching lanes now reuse local context, host vectors, buffers, and scratch across files
- Optional `--cuda-dedup sort_unstable` for faster sort/dedup path (especially on multi GPU), but `hashset` still default
- The `sort_unstable` + ULL path removes an extra full-hash host vector copy by sorting/deduping `full_hashes` in place after ULL consumes the full stream
- Partially tested on local machine and Russell; results are promising but no rigorous testing yet

## Metrics
In ns by default, `sketch_wall_ns` is wall clock (end to end time)

- `input_bases`
  Total nucleotide bases processed across all input files.
- `hashes_seen`
  Total hashed k-mers seen (before deduplication)
- `unique_hashes`
  Total unique hashes kept
- `fasta_ns`
  Input side prep time
  For CUDA this is `read_merge_seq`, which reads/decompresses and merges
  each FASTA into a buffer
- `hash_and_dedup_ns`
  CPU-side k-mer hash stream handling after input prep
  For CUDA this is the host side ULL update and `HashSet` construction after the
  GPU returns raw hashes
- `hd_encode_ns`
  Time spent encoding the deduplicated hash set into the HD vector.
  On CUDA builds this is the host-observed GPU HD encode stage.
- `hv_norm_ns`
  Time spent computing the HV L2 norm
- `hd_compress_ns`
  Time spent compressing the HV sketch (compress for output)
- `total_worker_ns`
  Total worker time (Rayon threads/workers)
- `sketch_wall_ns`
  Wall clock time
- `cuda_h2d_ns`
  PCIe time, or time copying FASTA sequence buffers from host memory to GPU memory
- `cuda_alloc_ns`
  Time to allocate GPU memory for hash output
- `cuda_launch_ns`
  Time for the CPU to start CUDA kernel execution (not kernel runtime)
- `cuda_d2h_ns`
  Time to copy raw hash buffer from GPU to CPU
- `cuda_zero_filter_ns`
  Time to remove padding from returned hash buffer (zeros)
- `cuda_filter_ns`
  CPU time to build ULL and HashSet from CUDA output
  `hash_and_dedup_ns` for comparison
- `cuda_hd_hash_h2d_ns`
  Time to copy sampled hashes to GPU for HD encoding
- `cuda_hd_hv_h2d_ns`
  Time to copy HD vector to GPU
- `cuda_hd_alloc_ns`
  Time to allocate GPU buffers for HD encoding
- `cuda_hd_kernel_launch_ns`
  Time for the CPU to launch the CUDA HD encode kernel
- `cuda_hd_d2h_ns`
  Time to copy the encoded HD vector back to CPU

# Last update (5/07): GPU HD Encode

## Findings/Testing

Local machine: Ryzen 9 8945HS, 32GB DDR5, RTX 4070 mobile
Lab server (Russell): Ryzen Threadripper 9985WX, 768GB DDR5, RTX PRO 6000 (only used 1x GPU)

GPU now hashes kmers and does HD encoding, for about a ~5x speedup

On original subset `GCA/946`
(~1120 genomes) with `-T 16 -d 4096`:

*Cuda HD metrics unavailable as were not yet implemented

| Metric | Seconds |
|---|---|
| `sketch_wall_s` | 230.671 |
| `fasta_s` | 21.365 |
| `hash_dedup_s` | 266.229 |
| `hd_encode_s` | 3331.676 |
| `hd_compress_s` | 0.025 |
| `cuda_h2d_s` | 2.404 |
| `cuda_alloc_s` | 5.007 |
| `cuda_launch_s` | 0.387 |
| `cuda_d2h_s` | 27.944 |
| `cuda_zero_filter_s` | 2.911 |
| `cuda_filter_s` | 266.229 |
| `cuda_hd_hash_h2d_s` | NA |
| `cuda_hd_hv_h2d_s` | NA |
| `cuda_hd_alloc_s` | NA |
| `cuda_hd_launch_s` | NA |
| `cuda_hd_d2h_s` | NA |


After moving HD encoding to GPU (3 run median):

| Metric | Seconds |
|---|---|
| `sketch_wall_s` | 44.666 |
| `fasta_s` | 18.858 |
| `hash_dedup_s` | 592.706 |
| `hd_encode_s` | 39.985 |
| `hd_compress_s` | 0.026 |
| `cuda_h2d_s` | 3.060 |
| `cuda_alloc_s` | 11.016 |
| `cuda_launch_s` | 1.167 |
| `cuda_d2h_s` | 10.012 |
| `cuda_zero_filter_s` | 2.706 |
| `cuda_filter_s` | 592.706 |
| `cuda_hd_hash_h2d_s` | 5.389 |
| `cuda_hd_hv_h2d_s` | 1.613 |
| `cuda_hd_alloc_s` | 9.887 |
| `cuda_hd_launch_s` | 0.986 |
| `cuda_hd_d2h_s` | 16.023 |

HD encode worker time dropped from `3331.676s` to `39.985s`, ~83x faster.

Wall clock time dropped from `230.671s` to `44.666`, ~5.2x faster.

### Testing at scale (full GTDB database):

Local:

| Metric | Seconds |
|---|---|
| `sketch_wall_s` | 5711.572 |
| `fasta_s` | 2476.974 |
| `hash_dedup_s` | 79545.811 |
| `hd_encode_s` | 4101.001 |
| `hd_compress_s` | 2.284 |
| `cuda_h2d_s` | 281.053 |
| `cuda_alloc_s` | 946.676 |
| `cuda_launch_s` | 101.854 |
| `cuda_d2h_s` | 950.724 |
| `cuda_zero_filter_s` | 348.173 |
| `cuda_filter_s` | 79545.811 |
| `cuda_hd_hash_h2d_s` | 593.772 |
| `cuda_hd_hv_h2d_s` | 148.233 |
| `cuda_hd_alloc_s` | 858.284 |
| `cuda_hd_launch_s` | 83.699 |
| `cuda_hd_d2h_s` | 1893.399 |

Russell:

| Metric | Seconds |
|---|---|
| `sketch_wall_s` | 1347.999 |
| `fasta_s` | 1152.078 |
| `hash_dedup_s` | 21402.781 |
| `hd_encode_s` | 66854.554 |
| `hd_compress_s` | 1.141 |
| `cuda_h2d_s` | 12062.720 |
| `cuda_alloc_s` | 22211.992 |
| `cuda_launch_s` | 1683.784 |
| `cuda_d2h_s` | 8131.035 |
| `cuda_zero_filter_s` | 179.630 |
| `cuda_filter_s` | 21402.781 |
| `cuda_hd_hash_h2d_s` | 12941.544 |
| `cuda_hd_hv_h2d_s` | 2661.834 |
| `cuda_hd_alloc_s` | 22478.886 |
| `cuda_hd_launch_s` | 3283.399 |
| `cuda_hd_d2h_s` | 3515.591 |

Local finished in `5711.572s` or ~95 minutes, previous ETA was ~8 hours, which checks out for a ~5.05x speedup

Russell finished in `1347.999s` or ~22.5 minutes, previous ETA was ~110 minutes (if I remember correctly), but would be a ~4.89x speedup

Interestingly, on local machine CPU was constantly hammered at full CPU util, while GPU util fluctuated from around 50-80%. Memory was maxed out as well.

However, on Russell, neither CPU or GPU was fully being utilized. Load average was ~15 even though running with -T 128. GPU util mirrored local, fluctuating from 50-80% util.

Not sure what the bottleneck is here, entire dataset was loaded into ram so we can rule out the hdd being the bottleneck.

## Current Update:

- HD encoding is no longer the main bottleneck
- CPU decompression is not as much of an issue as we thought earlier
- CPU still spends non insignificant amount of time on hash deduplication / ULL work
- There is still GPU overhead (avg ~60% util)
- Next: multi file GPU batching, GPU ULL, GPU dedup (or reduce CPU time on it), GPU compression, less CPU<->GPU transfer?

## HyperSpec

Drew from HyperSpec's GPU method, but not a direct port as they are incompatible.
(HyperSpec's mass spectra clustering has different inputs, HD vector representation, distance math, and file output format.)

# Update 5/14: Multi GPU, CUDA Scheduling, Scratch Work

## Findings/Testing

Local machine: Ryzen 9 8945HS, 32GB DDR5, RTX 4070 mobile
Lab server (Russell): Ryzen Threadripper 9985WX, 768GB DDR5, 4x RTX PRO 6000
- Multi GPU testing done on 3 GPUs, as 4th was in use

On original subset `GCA/946`
(~1120 genomes) with `-T 16 -d 4096`:

Local Machine (3 run median):\

| Metric | seconds |
|---|---:|
| `sketch_wall_s` | 22.595 |
| `fasta_s` | 10.309 |
| `hash_dedup_s` | 53.282 |
| `hd_encode_s` | 154.882 |
| `worker_total_s` | 342.038 |
| `cuda_h2d_s` | 66.955 |
| `cuda_alloc_s` | 8.186 |
| `cuda_launch_s` | 3.533 |
| `cuda_d2h_s` | 41.115 |
| `cuda_zero_filter_s` | 4.004 |
| `cuda_filter_s` | 53.282 |
| `cuda_hd_hash_h2d_s` | 69.537 |
| `cuda_hd_hv_h2d_s` | 48.407 |
| `cuda_hd_alloc_s` | 3.681 |
| `cuda_hd_launch_s` | 5.049 |
| `cuda_hd_d2h_s` | 24.812 |

Wall clock time dropped from `44.666s` to `22.595s`, ~1.98x faster

All outputs matched the known `GCA/946` hashes:

```text
.sketch  f602adddef6674ad80c9f923d2c32f891573a32648376b710f8c49e4b9bb0465
.ull     cff6bdb6577ee4658bf58de5ad71d7f1bf48ac96c853ad1a02821dd79365c713
```

### Testing at scale (full GTDB database):

Local Machine (1 run only):

| Metric | Seconds |
|---|---:|
| `sketch_wall_s` | 3020.280 |
| `fasta_s` | 2060.620 |
| `hash_dedup_s` | 9223.340 |
| `hd_encode_s` | 20594.300 |
| `hv_norm_s` | 0.783 |
| `hd_compress_s` | 2.090 |
| `worker_total_s` | 47495.200 |
| `cuda_h2d_s` | 8704.310 |
| `cuda_alloc_s` | 479.867 |
| `cuda_kmer_launch_s` | 810.182 |
| `cuda_d2h_s` | 5085.800 |
| `cuda_zero_filter_s` | 481.750 |
| `cuda_filter_s` | 9223.340 |
| `cuda_hd_hash_h2d_s` | 9274.860 |
| `cuda_hd_hv_h2d_s` | 6681.800 |
| `cuda_hd_alloc_s` | 5.071 |
| `cuda_hd_kernel_s` | 743.918 |
| `cuda_hd_d2h_s` | 3887.640 |

Finished in `3020.280s`, compared to previous `5711.572s` this is a ~1.9x speedup

On Russell:
Early testing, not enough trials run and multi GPU only running on 3 GPUs, will run on 4 GPU once all GPUs are free.

| Iteration | Mode | Wall s | Change |
|---|---|---:|---:|
| 0 | Old single GPU, `hashset`, | 1347.999 | old baseline |
| 1 | Single GPU, `sort_unstable`, run 1 | 1229.410 | ~9% faster old run, new baseline |
| 1 | Single GPU, `sort_unstable`, run 2 | 1191.010 | repeat |
| 2 | Single GPU, `sort_unstable` + scratch reuse + copy removal | 863.578 | ~28-30% faster than the two single-GPU `sort_unstable` runs |
| 3 | Multi GPU, `hashset` | 749.298 | ~38% faster than the two single GPU baseline runs |
| 4 | Multi GPU, `sort_unstable` | 447.872 | ~40% faster than multi GPU `hashset` |
| 5 | Multi GPU, `sort_unstable` + scratch reuse | 397.261 | ~11% faster than multi GPU `sort_unstable` |
| 6 | Multi GPU, `sort_unstable` + scratch reuse + copy removal, run 1 | 351.189 | ~12% faster than scratch reuse without copy removal |
| 6 | Multi GPU, `sort_unstable` + scratch reuse + copy removal, run 2 | 347.352 | repeat; ~13% faster than scratch reuse without copy removal |

Overall from old baseline, wall clock reduced from 1347.999 down to ~349s for ~3.86x speedup

More detailed metrics:

More detailed summary metrics from the server runs (new only unfortunately):

| Mode | Wall s | FASTA s | Dedup s | HD encode s | Worker total s | CUDA alloc s | K-mer launch s | CUDA D2H s | HD hash H2D s | HD HV H2D s | HD kernel s | HD D2H s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single GPU, `sort_unstable`, run 1 | 1229.410 | 25017.200 | 4453.140 | 62256.200 | 156099.000 | 16369.400 | 3232.550 | 5629.590 | 12182.800 | 3844.960 | 3172.630 | 3477.860 |
| Single GPU, `sort_unstable`, run 2 | 1191.010 | 1125.530 | 4460.740 | 71625.300 | 151194.000 | 19017.300 | 3595.680 | 6284.420 | 14039.500 | 4259.440 | 3640.030 | 3974.220 |
| Single GPU, `sort_unstable` + scratch reuse + copy removal | 863.578 | 1089.820 | 5036.020 | 45920.700 | 109268.000 | 1662.900 | 6525.500 | 18098.600 | 31135.700 | 4609.040 | 5705.780 | 4338.260 |
| Multi-GPU, `hashset` | 749.298 | 1436.850 | 43024.700 | 23150.400 | 93266.700 | 5693.950 | 1425.170 | 5227.870 | 4621.910 | 1730.590 | 1360.610 | 1793.290 |
| Multi-GPU, `sort_unstable` | 447.872 | 1160.280 | 4723.130 | 7098.130 | 20953.700 | 1797.470 | 422.743 | 1548.390 | 1410.800 | 480.712 | 372.786 | 737.564 |
| Multi-GPU, `sort_unstable` + scratch reuse | 397.261 | 1159.620 | 5652.180 | 22474.700 | 49482.000 | 1239.800 | 3142.750 | 6601.180 | 9167.260 | 6208.910 | 3407.170 | 3641.770 |
| Multi-GPU, `sort_unstable` + scratch reuse + copy removal | 347.352 | 1153.410 | 5514.160 | 16427.900 | 43066.900 | 565.033 | 2160.600 | 6666.390 | 10339.600 | 2293.270 | 1881.110 | 1866.810 |

Phase totals not always improved as they are summed across lanes; wall time is main improvement\
Matched previous full server output byte for byte and full output, so output has not changed

## Current Update
- Still a bit of GPU overhead (~15-20% on Russell)
- Dist has not been touched
- GPU batching seems possible but complicated, worth trying?

# (Everything below this line generated by AI)

# Usage / CLI Guide

This section is the practical command guide: build, sketching, CUDA lane behavior, dedup options, metrics, ANI estimation, and compatibility notes.

## Output Files And Compatibility

This branch adds explicit CPU/CUDA sketch device selection, stage timing
metrics, recursive compressed FASTA input discovery, and CUDA acceleration for
both k-mer hashing and HD encoding. The current CUDA path also supports host
worker lanes distributed across visible CUDA devices, reusable lane-local CUDA
scratch state, and selectable CUDA dedup strategies. CPU sketching remains the
correctness baseline.

A sketch command writes two files:

- DotHash sketch: `<output>.sketch`
- UltraLogLog sketch: `<output>.sketch.ull`

The existing `.sketch` and `.ull` formats are preserved. This branch does not
add binary HVs, GPU distance, GPU ULL, GPU dedup, GPU norm, GPU compression, or
multi-file GPU batching.

## Quickstart

### Build

_dotANI_ requires Rust and Cargo. CUDA support requires an NVIDIA driver and a
working CUDA installation. Check GPU availability with:

```sh
nvidia-smi
```

CPU-only build:

```sh
cargo build --release
```

CUDA-capable build:

```sh
cargo build --release --features cuda
```

`--features cuda` is a build-time flag. After building with CUDA support, choose
the sketch execution device at runtime with `--device cpu|cuda`.

## Sketching

Basic CPU sketch:

```sh
target/release/dotani sketch --device cpu \
  -p ./data \
  -o ./fna.sketch
```

Basic CUDA sketch:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch
```

CUDA sketch with the conservative HashSet dedup path explicitly selected:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch \
  --cuda-dedup hashset
```

`hashset` is the default CUDA dedup strategy, so the explicit flag is optional.
It is still useful in benchmark scripts where the intended mode should be
visible in the command itself.

CUDA sketch with the faster sort/dedup path:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch \
  --cuda-dedup sort_unstable
```

CUDA-enabled builds default to CUDA, so this:

```sh
target/release/dotani sketch \
  -p ./data \
  -o ./fna.sketch
```

is equivalent to:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch
```

CPU-only builds default to CPU. In a CPU-only build, `--device cuda` exits with a
clear error.

### Sketch Options

```text
Usage: dotani sketch [OPTIONS] --path <path> --out <out>

Options:
  -p, --path <path>                Input folder path containing .fna/.fa/.fasta files
                                   gzip/bzip2/xz/zstd compressed files are supported
  -o, --out <out>                  Output DotHash sketch file
  -T, --threads <threads>          Number of threads, default all logical cores
  -C, --canonical <canonical>      Whether to use canonical k-mers [default: true]
  -k, --ksize <ksize>              k-mer size for sketching [default: 16]
  -S, --seed <seed>                Hash seed [default: 1447]
      --ull                        Use UltraLogLog cardinality estimation (default)
      --ell                        Use ExaLogLog cardinality estimation
      --ull-p <ull_p>              UltraLogLog precision parameter [default: 14]
      --ell-t <ell_t>              ExaLogLog t parameter [default: 2]
      --ell-d <ell_d>              ExaLogLog d parameter [default: 24]
      --ell-p <ell_p>              ExaLogLog precision parameter [default: 12]
  -d, --hv-d <hv_d>                Dimension for hypervector [default: 4096]
  -Q, --quant-scale <scale>        Scaling factor for HV quantization [default: 1.0]
      --device <cpu|cuda>          Sketch execution device
      --cuda-dedup <hashset|sort_unstable>
                                   CUDA dedup strategy [default: hashset]
      --metrics-out <prefix>       Write metrics TSV files using this prefix
  -h, --help                       Print help
  -V, --version                    Print version
```

Input discovery is recursive under `--path`, so nested GTDB-style directory
trees such as `database/GCA/946/.../*.fna.gz` are supported.

ULL is selected when neither estimator flag is present and writes
`<output>.ull`; ELL writes `<output>.ell`. Use separate runs for comparisons:

```bash
target/release/dotani sketch --ull -p data -o ull.sketch
target/release/dotani dist --ull -r ull.sketch -q ull.sketch -o ull.ani

target/release/dotani sketch --ell -p data -o ell.sketch
target/release/dotani dist --ell -r ell.sketch -q ell.sketch -o ell.ani
```

Defaults are ULL `p=14` and ELL `t=2,d=24,p=12`, both with 16,384 bytes of
raw estimator state per genome. Compare sketch wall/stage metrics, sidecar
sizes, dist cardinality timing, and ANI differences. Identical inputs and
sketch parameters should produce identical DotHash `.sketch` files.

### CUDA Host Lanes And Multi-GPU Scheduling

CUDA sketching uses host worker lanes. `-T/--threads` controls the number of
host lanes, not the number of GPUs directly. When multiple CUDA devices are
visible, lanes are assigned across those devices round-robin by lane id.

For example:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch \
  -T 48
```

uses up to 48 host lanes distributed across the three visible CUDA devices.
Each lane still processes one file at a time. The scheduler is file-level and
does not yet batch multiple files into one GPU launch.

Russell has 4 RTX PRO 6000 GPUs available, but the documented 5/14 multi-GPU
server runs used 3 visible GPUs because one GPU was in use by another job.

Output ordering remains deterministic. Lanes store results by original input
file index, and the final sketch is assembled in input order after all workers
finish.

Per-file metrics include:

- `cuda_stream_lane`
  Host lane id that processed the file.
- `cuda_device_id`
  CUDA visible device id used by that lane.

These columns are useful for checking lane/device balance. With
`CUDA_VISIBLE_DEVICES=1,2,3`, metric device ids are still visible ordinals
`0`, `1`, and `2` after CUDA remapping.

### CUDA Dedup Strategy

CUDA dedup defaults to the conservative HashSet path:

```sh
--cuda-dedup hashset
```

This can be passed explicitly even though it is the default. Use it when a run
should be clearly labeled as the conservative dedup path.

The faster opt-in path is:

```sh
--cuda-dedup sort_unstable
```

In `sort_unstable`, ULL still consumes the full hash stream before
deduplication. The deduplicated hashes are then sorted and fed to HD encoding.
This changes internal unique-hash order, but HD contributions are additive and
order-independent. In current tests, `hashset` and `sort_unstable` produced
byte-identical `.sketch` and `.ull` outputs.

## Metrics Guide

Metrics are disabled by default. Enable them with:

```sh
target/release/dotani sketch --device cuda \
  -p ./data \
  -o ./fna.sketch \
  --metrics-out ./fna_metrics
```

This writes:

```text
./fna_metrics.summary.tsv
./fna_metrics.files.tsv
```

CPU and CUDA runs use the same TSV schema. CPU rows emit `NA` for CUDA-specific
columns. CUDA rows emit `NA` for `cuda_hd_*` only when no GPU HD work applies,
such as empty sampled hashes or `hv_d < 64`.

Important columns:

- `input_bases`
- `hashes_seen`
- `unique_hashes`
- `fasta_ns`
- `hash_and_dedup_ns`
- `hd_encode_ns`
- `hv_norm_ns`
- `hd_compress_ns`
- `total_worker_ns`
- `sketch_wall_ns`
- `cuda_h2d_ns`
- `cuda_alloc_ns`
- `cuda_launch_ns`
- `cuda_d2h_ns`
- `cuda_zero_filter_ns`
- `cuda_filter_ns`
- `cuda_hd_hash_h2d_ns`
- `cuda_hd_hv_h2d_ns`
- `cuda_hd_alloc_ns`
- `cuda_hd_kernel_launch_ns`
- `cuda_hd_d2h_ns`
- `cuda_stream_lane`
- `cuda_device_id`

`cuda_launch_ns` is host enqueue time, not true kernel duration. Use Nsight
Systems or Nsight Compute for true CUDA kernel timing.

`sketch_wall_ns` is populated on the summary row and is `NA` for individual file
rows. `cuda_zero_filter_ns` is the host-side removal of zero padding from the
CUDA output buffer. `cuda_filter_ns` is the CUDA path's host-side ULL and
dedup construction time, and is also included in `hash_and_dedup_ns` for
CPU/CUDA comparison.

CUDA HD encode columns:

- `cuda_hd_hash_h2d_ns`
  Time to copy the unique sampled hash vector to GPU for HD encoding.
- `cuda_hd_hv_h2d_ns`
  Time to copy the initialized HD vector to GPU.
- `cuda_hd_alloc_ns`
  Time to allocate GPU buffers for HD encoding.
- `cuda_hd_kernel_launch_ns`
  Host-side CUDA HD kernel enqueue time.
- `cuda_hd_d2h_ns`
  Time to copy the encoded HD vector back to host.
- `cuda_stream_lane`
  Host lane id that processed the file.
- `cuda_device_id`
  CUDA visible device id used by the lane.

Readable seconds view:

```sh
awk -F'\t' '
NR==1 { for (i=1;i<=NF;i++) h[$i]=i; next }
{
  printf "sketch_wall_s=%.3f\n", $h["sketch_wall_ns"]/1e9
  printf "fasta_s=%.3f\n", $h["fasta_ns"]/1e9
  printf "hash_dedup_s=%.3f\n", $h["hash_and_dedup_ns"]/1e9
  printf "hd_encode_s=%.3f\n", $h["hd_encode_ns"]/1e9
  printf "hd_compress_s=%.3f\n", $h["hd_compress_ns"]/1e9
  printf "cuda_h2d_s=%.3f\n", $h["cuda_h2d_ns"]/1e9
  printf "cuda_alloc_s=%.3f\n", $h["cuda_alloc_ns"]/1e9
  printf "cuda_launch_s=%.3f\n", $h["cuda_launch_ns"]/1e9
  printf "cuda_d2h_s=%.3f\n", $h["cuda_d2h_ns"]/1e9
  printf "cuda_zero_filter_s=%.3f\n", $h["cuda_zero_filter_ns"]/1e9
  printf "cuda_filter_s=%.3f\n", $h["cuda_filter_ns"]/1e9
  printf "cuda_hd_hash_h2d_s=%.3f\n", $h["cuda_hd_hash_h2d_ns"]/1e9
  printf "cuda_hd_hv_h2d_s=%.3f\n", $h["cuda_hd_hv_h2d_ns"]/1e9
  printf "cuda_hd_alloc_s=%.3f\n", $h["cuda_hd_alloc_ns"]/1e9
  printf "cuda_hd_launch_s=%.3f\n", $h["cuda_hd_kernel_launch_ns"]/1e9
  printf "cuda_hd_d2h_s=%.3f\n", $h["cuda_hd_d2h_ns"]/1e9
}' /tmp/gca946_cuda_4096.summary.tsv
```

## Metrics Reference
In ns by default, `sketch_wall_ns` is wall clock (end to end time)

- `input_bases`
  Total nucleotide bases processed across all input files.
- `hashes_seen`
  Total hashed k-mers seen (before deduplication)
- `unique_hashes`
  Total unique hashes kept
- `fasta_ns`
  Input side prep time
  For CUDA this is `read_merge_seq`, which reads/decompresses and merges
  each FASTA into a buffer
- `hash_and_dedup_ns`
  CPU-side k-mer hash stream handling after input prep
  For CUDA this is the host side ULL update and `HashSet` construction after the
  GPU returns raw hashes
- `hd_encode_ns`
  Time spent encoding the deduplicated hash set into the HD vector.
  On CUDA builds this is the host-observed GPU HD encode stage.
- `hv_norm_ns`
  Time spent computing the HV L2 norm
- `hd_compress_ns`
  Time spent compressing the HV sketch (compress for output)
- `total_worker_ns`
  Total worker time (Rayon threads/workers)
- `sketch_wall_ns`
  Wall clock time
- `cuda_h2d_ns`
  PCIe time, or time copying FASTA sequence buffers from host memory to GPU memory
- `cuda_alloc_ns`
  Time to allocate GPU memory for hash output
- `cuda_launch_ns`
  Time for the CPU to start CUDA kernel execution (not kernel runtime)
- `cuda_d2h_ns`
  Time to copy raw hash buffer from GPU to CPU
- `cuda_zero_filter_ns`
  Time to remove padding from returned hash buffer (zeros)
- `cuda_filter_ns`
  CPU time to build ULL and deduplicated hashes from CUDA output
  `hash_and_dedup_ns` for comparison
- `cuda_hd_hash_h2d_ns`
  Time to copy sampled hashes to GPU for HD encoding
- `cuda_hd_hv_h2d_ns`
  Time to copy HD vector to GPU
- `cuda_hd_alloc_ns`
  Time to allocate GPU buffers for HD encoding
- `cuda_hd_kernel_launch_ns`
  Time for the CPU to launch the CUDA HD encode kernel
- `cuda_hd_d2h_ns`
  Time to copy the encoded HD vector back to CPU
- `cuda_stream_lane`
  CUDA host lane id that processed the file
- `cuda_device_id`
  CUDA visible device id used by that lane

## ANI Estimation

```sh
target/release/dotani dist \
  -r fna1.sketch \
  -q fna2.sketch \
  -o output.ani
```

The corresponding ULL files are expected next to each sketch as:

```text
fna1.sketch.ull
fna2.sketch.ull
```

## Backward Compatibility Notes

Older commands may have used names like `dotani-cuda` or `dotani_gpu`. This
branch builds a single binary at `target/release/dotani`. If an old script
requires `dotani_gpu`, create a local symlink after building.
