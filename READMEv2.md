# dotANI

dotANI creates compact sketches of genome FASTA files and estimates average nucleotide identity (ANI) between genomes from those sketches. Sketching is fast and memory-efficient, and runs on the CPU or on NVIDIA GPUs.

## 1. Build

Install Rust and Cargo, then build from the repository root:

```sh
# CPU-only build
cargo build --release

# NVIDIA GPU build
cargo build --release --features cuda
```

GPU support requires an NVIDIA driver; check with `nvidia-smi` or `nvcc -V`. Only NVIDIA GPUs are supported.

## 2. Sketch genomes

Point sketch at a folder of FASTA files and give an output path:

```sh
dotani sketch -p ./data -o ./fna.sketch
```

This creates two files:

- `fna.sketch` — the sketch used for comparisons
- `fna.sketch.ull` — the genome-size (cardinality) estimates needed to compute ANI

Both files are required for comparison, so keep them together. If you pass `--ell` (ExaLogLog instead of the default UltraLogLog), the second file is named `fna.sketch.ell`; use the same flag when comparing later.

Inputs are `.fna`, `.fa`, and `.fasta` files, optionally compressed with gzip, bzip2, xz, or zstd (for example `.fna.gz` or `.fa.bz2`).

### Sketching from a TSV input list

`-p` sketches every matching file in a folder and derives names from the file paths. For exact control over which files are used and what they are called, pass `--manifest` with a two-column TSV file:

```sh
dotani sketch --manifest reads.tsv -o ./fna.sketch
```

`reads.tsv`:

```text
read_path	file_id
/data/genomes/genome_a.fna.gz	genome_a
/data/genomes/genome_b.fna	genome_b
```

The header must name the two columns `read_path` and `file_id`. Row order and `file_id` values are preserved in the sketch output. Paths are resolved relative to the directory you run dotani from, so absolute paths are the safest choice.

## 3. Compare sketches

```sh
dotani dist -r ref.sketch -q query.sketch -o ani_results.tsv
```

For each input sketch, dist also reads the additional `.ull` or `.ell` file required for comparison from the same location, so keep each sketch pair where you created it.

By default the output contains one line per genome pair whose estimated ANI reaches the threshold (85.0 by default; change with `-a/--ani-th`). Pass `--output-mode count` to get a short totals summary instead of one row per pair: how many pairs were compared, how many passed the threshold, and the threshold used. Passing the same sketch file as both `-r` and `-q` computes all pairwise comparisons within that one file.

## 4. Use GPU acceleration

In a CUDA build, sketching runs on the GPU by default and `dotani dist` uses the GPU automatically; no extra flags are needed:

```sh
dotani sketch -p ./data -o ./fna.sketch                # runs on the GPU
dotani sketch --device cpu -p ./data -o ./fna.sketch   # forces the CPU
```

In a CPU-only build, sketching runs on the CPU, and `--device cuda` is rejected with an error telling you to rebuild with `--features cuda`.

## 5. Less-common options

Most runs need none of these. Run `dotani sketch --help` and `dotani dist --help` for the complete, always-current option list.

- `--cuda-dedup hashset|sort_unstable` — only matters when sketching on a GPU: chooses how duplicate hashes are removed before encoding. `sort_unstable` is the default; `hashset` is the older method, kept for comparison. Both produce identical sketches.
- `--max-readers N` — GPU sketching only: caps how many FASTA read/decompress workers feed the GPUs at once.
- `--metrics-out PREFIX` — writes sketch timing summaries to `PREFIX.summary.tsv` and per-file metrics to `PREFIX.files.tsv`.
