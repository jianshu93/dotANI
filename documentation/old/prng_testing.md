# WyRng Reimplementation vs Existing CUDA PRNG Functions
This has already been reverted, changes were in commit 2356382.

## Summary

`wyrng` has the best small accuracy result, and is consistently ever so slightly faster than `curand_philox10`. Other tried candidates from the paper are much worse than both.

## Choosing PRNG from Paper

The paper has a bunch of PRNGs, but our HD Encode requires for each (hash, chunk) a 64 bit pseudorandom word without maintaining stream per hash or depending on cpu thread scheduling.
Thus this best hints to `curand_philox10` as it's counter based and stateless. `lcg64` also has potential as it's fast for direct seek and cheap, but quality may not be the best. Also tried `curand_xorwow` because it's the curand standard. None of the othe prng algorithms seem very good for our use case, but I could have missed something (let me know)

## Accuracy (adapted from Claire's notebook)

Using the same 16 testing genomes from Claire's notebook; just looked at all per pair error metrics to get average and max. If we wanted to test better we could probably add more metrics like root mean square error, etc. and test on full bindash on Russell, but this is *probably good enough for now for direction purposes (keep wyrng)

| PRNG | Mean Absolute Error | Max Absolute Error | Notes |
|---|---:|---:|---|
| `wyrng` | `0.0007131666667` | `0.00171` | Best accuracy |
| `curand_philox10` | `0.001153916667` | `0.0024` | Slightly worse than wyrng |
| `curand_xorwow` | `0.2421538333` | `1` | Horrible error |
| `lcg64` | `0.03419058333` | `0.9997` | Horrible error |

## Run on ~1100 genomes (GCA/946)

| PRNG | Wall (s) | Files/s | HD Encode Time |
|---|---:|---:|---:|
| `wyrng` | `22.59s` | `49.8` | `144.606s` |
| `curand_philox10` | `24.65s` | `45.6` | `162.389s` |
| `curand_xorwow` | `554.99s` | `2.0` | `4651.440s` |
| `lcg64` | `28.05s` | `40.1` | `194.523s` |

`wyrng` ~9% faster than `curand_philox10`. Throwing out `curand_xorwow` and `lcg64` as they are too slow, but their accuracy wasn't good enough anyway

## GTDB ~113k on Russell

### 3x GPU Runs (128 threads)
*Run may have been affected by background CPU usage, so ran 2x GPU runs afterwards on 32 threads

| PRNG | Wall (s) | Files/s |
|---|---:|---:|
| `wyrng` | `338.04s` | `334.6` |
| `curand_philox10` | `338.83s` | `333.8` |

Basically a tie, although `wyrng` is faster by about ~0.2%

### 2x GPU Runs (32 threads)

| Run | PRNG | Wall Time | Files/s |
|---|---|---:|---:|
| `1` | `wyrng` | `459.22s` | `246.3` |
| `1` | `curand_philox10` | `462.15s` | `244.7` |
| `2` | `wyrng` | `455.13s` | `248.5` |
| `2` | `curand_philox10` | `461.53s` | `245.1` |

`wyrng` average `457.175s`, `curand_philox10` average `461.84s`, `wyrng` ~1% faster

Stage metrics if interested:

Run 1:
I will
| PRNG | metrics wall_s | hash_dedup_s | hd_encode_s | hd_kernel_s | d2h_s | total_worker_s |
|---|---:|---:|---:|---:|---:|---:|
| `curand_philox10` | `470.15` | `5001.35` | `3971.24` | `419.05` | `724.55` | `14696.79` |
| `wyrng` | `467.48` | `5015.07` | `3787.46` | `364.41` | `671.80` | `14603.53` |

Run 2:

| PRNG | metrics wall_s | hash_dedup_s | hd_encode_s | hd_kernel_s | d2h_s | total_worker_s |
|---|---:|---:|---:|---:|---:|---:|
| `curand_philox10` | `469.54` | `4961.77` | `3888.55` | `359.96` | `717.04` | `14675.21` |
| `wyrng` | `463.23` | `5032.55` | `3808.98` | `384.55` | `685.26` | `14474.75` |
