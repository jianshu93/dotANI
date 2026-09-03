# GPU HD Encode

## Overview

dotANI moves hypervector (HV) encoding to the GPU while preserving sketching behavior, the distance path, and the `.sketch`/`.ull`/`.ell` file formats. The GPU reproduces the exact pseudorandom WyRng word that the CPU would produce for each `(hash, 64-coordinate chunk)` without calling the Rust WyRng function, so independent CUDA threads can compute chunk contributions without walking a serial WyRng stream. The parallel-coordinate strategy was inspired by HyperSpec's GPU encoder, but this is not a direct port (HyperSpec encodes mass-spectra peaks into a packed binary HV; dotANI encodes unique sampled k-mer hashes into a signed `i32` count vector).

## Pipeline context

Genome sketching in dotANI proceeds as:

1. Decompress and read FASTA input
2. Hash k-mers
3. Deduplicate sampled hashes
4. Run ULL/ELL to estimate genome cardinality
5. Encode the sampled hash set into an HD vector
6. Compute the HD vector norm
7. Compress the HD vector for output
8. Write the `.sketch` (DotHash) file and the `.ull`/`.ell` sidecar

Upstream, HD encoding ran on the CPU (AVX-512/AVX-2 code paths) and was CPU-bound:

```rust
let hv = if is_x86_feature_detected!("avx512f") {
    unsafe { hd::encode_hash_hd_avx512(&sampled_hash_set, &sketch) }
} else if is_x86_feature_detected!("avx2") {
    unsafe { hd::encode_hash_hd_avx2(&sampled_hash_set, &sketch) }
} else {
    hd::encode_hash_hd(&sampled_hash_set, &sketch)
};
```

The CUDA path keeps GPU k-mer hashing, moves HD encoding to the GPU, and leaves the rest of the pipeline unchanged:

- GPU: FASTA sequence buffer to k-mer hashes
- CPU: ULL/ELL update and hash deduplication
- GPU: deduplicated sampled hashes to signed HD count vector
- CPU: norm, compression, output

Deduplicated hashes are collected into a `Vec<u64>` and sent to the GPU for HD encoding. The focused host wrapper used by kernel tests lives in `src/hd_cuda.rs::encode_hash_hd_cuda`; production sketching launches the same kernel through the scratch-reusing `encode_hash_hd_cuda_into` path in `src/sketch_cuda.rs`. This is not a full GPU rewrite: cardinality estimation, deduplication, compression, and the distance path are unchanged and format-compatible.

## CPU HD encode

The baseline CPU implementation:

```rust
pub fn encode_hash_hd(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    let hv_d = sketch.hv_d;
    let seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let mut hv = vec![-(kmer_hash_set.len() as i32); hv_d];

    for hash in seed_vec {
        let mut rng = WyRng::seed_from_u64(hash);

        for i in 0..(hv_d / 64) {
            let rnd_bits = rng.next_u64();

            for j in 0..64 {
                hv[i * 64 + j] += (((rnd_bits >> j) & 1) << 1) as i32;
            }
        }
    }

    hv
}
```

Let:

- `hv` be the hd sketch vector
- `H` be the unique sampled hash set (dedup HashSet<u64> of sampled hashes; this is the input to HD encode)
- `N = |H|` (cardinality of H)
- `D = hd vector dimension`
- `chunk = d / 64` (which 64bit pseudorandom output controls coordinate d)
- `bit = d % 64` (bit position in 64 bit word; LSB)
- `word(h, chunk)` is the `next_u64()` output at position `chunk` in the WyRng stream created by `WyRng::seed_from_u64(h)`, where position `0` is the first `next_u64()` call.

For each coordinate `d`, CPU definition is as follows:

$$
hv[d] = \sum_{h \in H} \text{contribution}(h, d)
$$

$$
\text{contribution}(h, d) =
\begin{cases}
    +1 & \text{if } \text{bit}(\text{word}(h, \lfloor d / 64 \rfloor), d \bmod 64) = 1 \\
    -1 & \text{otherwise}
\end{cases}
$$

And is computed by:

```text
hv[d] = -N + 2 * ones(d)

ones(d) = count of hashes h where bit(word(h, d / 64), d % 64) == 1
```

For one coordinate d, every hash in H contributes either:

+1 if its random bit for coordinate d is 1
-1 if its random bit for coordinate d is 0

So if there are N hashes, hv[d] is the sum of N contributions.

hv[d] starts at -N
```text
for each hash:
    if bit is 1:
        hv[d] += 2
    else:
        do nothing
```
So if N = 5 and three hashes have a 1 bit at coordinate d, then\
hv[d] = -5 + 2 + 2 + 2 = 1

### WyRng chunking

For each sampled hash h, CPU starts a fresh WyRng stream:
```text
rng = WyRng::seed_from_u64(hash)
```
Because each u64 has 64 bits, one `next_u64()` output controls 64 hd vector coordinates.
```text
coordinates 0..63     use chunk 0, the first  next_u64()
coordinates 64..127   use chunk 1, the second next_u64()
coordinates 128..191  use chunk 2, the third  next_u64()
```

Within a word, coordinate `chunk * 64 + bit` reads bit position `bit` from the LSB numbering by:

```rust
(rnd_bits >> j) & 1
```

The CUDA implementation reproduces the same pseudorandom WyRng word for the same `(hash, chunk)` pair and must use the same bit numbering.

## Direct WyRng computation on the GPU

A direct GPU port of the serial CPU loop would not parallelize, so the GPU computes the RNG output for a specific chunk directly instead of walking the stream.

```cuda
static const uint64_t WY_P0 = UINT64_C(0xa0761d6478bd642f);
static const uint64_t WY_P1 = UINT64_C(0xe7037ed1a0b428db);

extern "C" __device__ __forceinline__ uint64_t wymum_u64(uint64_t a,
                                                          uint64_t b) {
  uint64_t high = __umul64hi(a, b);
  uint64_t low = a * b;
  return high ^ low;
}

extern "C" __device__ __forceinline__ uint64_t
wyrng_at_chunk(uint64_t hash, int chunk) {
  uint64_t state = hash + (((uint64_t)chunk + 1) * WY_P0);
  return wymum_u64(state ^ WY_P1, state);
}
```

- WyRng advances state by adding a fixed constant each `next_u64()`, so the per-chunk constant is 0xa0761d6478bd642f
- WyRng then turns state into a pseudorandom u64 with the second constant 0xe7037ed1a0b428db and the wymum multiply/xor:
```cuda
  uint64_t high = __umul64hi(a, b);
  uint64_t low = a * b;
  return high ^ low;
```

The Rust WyRng function is not called; the same formula is reimplemented, producing the same result. This lets a CUDA thread compute

```text
rnd = WyRng(seed = hash).nth_u64(chunk)
```

without walking through every chunk sequentially.

## Correctness

CPU and GPU HD encoders compute the same equation; only execution order differs.

```text
initial hv[d] = -N
for each hash h:
    rnd = WyRng(seed = h).next_u64_for_chunk(d / 64)
    if bit(d % 64) in rnd is 1:
        hv[d] += 2
```

- CPU iterates through hashes, then chunks, then bits
- CUDA processes hashes and chunks in parallel across threads and blocks, then adds partial counts within each block

Reordering does not change the result because the operations are repeated integer additions to the same initialized vector.

### Unit tests

The direct-seek WyRng logic is tested against sequential Rust `WyRng`, and the CUDA implementation is exercised through the `cuda_test_wyrng_at_chunk` kernel entry point. The CUDA HD output is compared against the CPU encoder for both the encoded vector and its L2 norm.

Tested cases include:

- Empty hash inputs.
- `hv_d < 64`.
- Single representative hashes.
- Hash value `0`.
- Mixed representative hashes including `u64::MAX`.
- `hv_d = 1024`.
- `hv_d = 4096`.

### End-to-end validation

An end-to-end run used the GTDB `GCA/946` subset (1124 `.fna.gz` files, 728M on disk) with identical input and sketch parameters for CPU and CUDA:

```sh
dotani sketch --device cpu -p gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946 \
  -o gtdb_gca946_full_cpu_hvd4096.sketch -T 16 -d 4096 \
  --metrics-out gtdb_gca946_full_cpu_hvd4096

dotani sketch --device cuda -p gtdb_genomes/gtdb_genomes_reps_r220/database/GCA/946 \
  -o gtdb_gca946_full_cuda_hvd4096.sketch -T 16 -d 4096 \
  --metrics-out gtdb_gca946_full_cuda_hvd4096
```

The resulting `.sketch` and `.ull` files were byte-identical between CPU and CUDA (matching SHA-256 digests). Run logs reported:

```text
CPU:  Sketching 1124 files took 255.44s - Speed: 4.4 files/s
CUDA: Sketching 1124 files took 39.33s - Speed: 28.6 files/s
```
