use log::info;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use needletail::{Sequence, parse_fastx_file};
use rayon::prelude::*;

use crate::cardinality::CardinalitySketch;
use crate::types::*;
use crate::{dist, hd, utils};

#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) -> Result<()> {
    utils::validate_sketch_params(&params)?;
    let sketch_wall_start = Instant::now();
    let inputs = utils::get_sketch_inputs(&params)?;
    let n_file = inputs.len();

    info!("Start sketching...");
    let pb = utils::get_progress_bar(n_file);

    let results: Vec<(FileSketch, FileCardinalitySketch, FileSketchMetrics)> = inputs
        .par_iter()
        .map(|input| -> Result<_> {
            let worker_start = Instant::now();
            let mut sketch = FileSketch {
                ksize: params.ksize,
                scaled: params.scaled,
                seed: params.seed,
                canonical: params.canonical,
                hv_d: params.hv_d,
                hv_quant_bits: 0u8,
                hv_norm_2: 0,
                file_str: input.file_id.clone(),
                hv: Vec::<i32>::new(),
            };

            let cardinality_sketch = CardinalitySketch::new(&params)?;
            let (kmer_hash_set, cardinality_sketch, mut metrics) = match cardinality_sketch {
                CardinalitySketch::Ull(mut ull) => {
                    let (hashes, metrics) =
                        extract_kmer_hashes(&input.read_path, &sketch, |hash| {
                            ull.add(hash);
                        })?;
                    (hashes, CardinalitySketch::Ull(ull), metrics)
                }
                CardinalitySketch::Ell(mut ell) => {
                    let (hashes, metrics) =
                        extract_kmer_hashes(&input.read_path, &sketch, |hash| {
                            ell.add_hash(hash);
                        })?;
                    (hashes, CardinalitySketch::Ell(ell), metrics)
                }
            };
            metrics.file = sketch.file_str.clone();
            metrics.unique_hashes = kmer_hash_set.len();

            let start = Instant::now();
            let hv = if is_x86_feature_detected!("avx512f") {
                unsafe { hd::encode_hash_hd_avx512(&kmer_hash_set, &sketch) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { hd::encode_hash_hd_avx2(&kmer_hash_set, &sketch) }
            } else {
                hd::encode_hash_hd(&kmer_hash_set, &sketch)
            };
            metrics.hd_encode_ns = start.elapsed().as_nanos();

            let start = Instant::now();
            sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);
            metrics.hv_norm_ns = start.elapsed().as_nanos();

            let start = Instant::now();
            if params.if_compressed {
                sketch.hv_quant_bits = hd::compress_hd_sketch(&mut sketch, &hv)?;
            } else {
                sketch.hv = hv.clone();
            }
            metrics.hd_compress_ns = start.elapsed().as_nanos();

            let cardinality_record =
                cardinality_sketch.into_record(&params, sketch.file_str.clone());

            pb.inc(1);
            pb.eta();

            metrics.total_worker_ns = worker_start.elapsed().as_nanos();

            Ok((sketch, cardinality_record, metrics))
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_and_clear();

    let mut all_filesketch = Vec::with_capacity(results.len());
    let mut all_cardinality_sketches = Vec::with_capacity(results.len());
    let mut all_metrics = Vec::with_capacity(results.len());
    for (sketch, cardinality, metrics) in results {
        all_filesketch.push(sketch);
        all_cardinality_sketches.push(cardinality);
        all_metrics.push(metrics);
    }

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        n_file,
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file)?;

    utils::dump_cardinality_sketch(
        all_cardinality_sketches,
        params.cardinality_estimator,
        &params.cardinality_out_file,
    )?;

    if let Some(prefix) = &params.metrics_out {
        utils::dump_sketch_metrics(&all_metrics, prefix, sketch_wall_start.elapsed().as_nanos())?;
    }

    Ok(())
}

fn extract_kmer_hashes<F>(
    read_path: &Path,
    sketch: &FileSketch,
    mut add_cardinality_hash: F,
) -> Result<(HashSet<u64>, FileSketchMetrics)>
where
    F: FnMut(u64),
{
    let ksize = sketch.ksize;
    let seed = sketch.seed;
    let mut metrics = FileSketchMetrics::default();

    let start = Instant::now();
    let mut fastx_reader = parse_fastx_file(read_path)
        .map_err(|e| anyhow!("failed to open FASTA/FASTQ {}: {e}", read_path.display()))?;
    metrics.fasta_ns = start.elapsed().as_nanos();

    let mut hash_set = HashSet::<u64>::new();
    let mut valid_kmers = 0usize;

    while let Some(record) = fastx_reader.next() {
        let start = Instant::now();
        let seqrec: needletail::parser::SequenceRecord<'_> = record.with_context(|| {
            format!(
                "failed to parse FASTA/FASTQ record in {}",
                read_path.display()
            )
        })?;

        let norm_seq = seqrec.normalize(false);
        metrics.input_bases += norm_seq.len();
        let rc = norm_seq.reverse_complement();
        metrics.fasta_ns += start.elapsed().as_nanos();

        let start = Instant::now();
        if sketch.canonical {
            for (_, kmer, _) in norm_seq.canonical_kmers(ksize, &rc) {
                let h = t1ha::t1ha2_atonce(kmer, seed);
                add_cardinality_hash(h);
                metrics.hashes_seen += 1;
                valid_kmers += 1;
                hash_set.insert(h);
            }
        } else {
            for kmer in norm_seq.kmers(ksize) {
                if !kmer
                    .iter()
                    .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
                {
                    continue;
                }
                let h = t1ha::t1ha2_atonce(kmer, seed);
                add_cardinality_hash(h);
                metrics.hashes_seen += 1;
                valid_kmers += 1;
                hash_set.insert(h);
            }
        }
        metrics.hash_and_dedup_ns += start.elapsed().as_nanos();
    }

    if valid_kmers == 0 {
        bail!(
            "input file {} produced no valid {}-mers",
            read_path.display(),
            ksize
        );
    }

    Ok((hash_set, metrics))
}

#[cfg(test)]
mod tests {
    use super::extract_kmer_hashes;
    use crate::types::FileSketch;
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    fn test_file(name: &str, contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "dotani_sketch_{}_{}_{}.fna",
            std::process::id(),
            name,
            std::thread::current().name().unwrap_or("unnamed")
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    fn test_sketch(ksize: u8, canonical: bool) -> FileSketch {
        FileSketch {
            ksize,
            scaled: 1,
            canonical,
            seed: 1447,
            hv_d: 256,
            hv_quant_bits: 0,
            hv_norm_2: 0,
            file_str: String::from("test"),
            hv: vec![0; 256],
        }
    }

    #[test]
    fn cpu_noncanonical_hashes_forward_valid_windows_and_skips_ambiguous() {
        let path = test_file("forward", b">record\nACNTGT\n");
        let sketch = test_sketch(2, false);

        let (hashes, metrics) = extract_kmer_hashes(&path, &sketch, |_| {}).unwrap();
        let expected: HashSet<u64> = [b"AC", b"TG", b"GT"]
            .into_iter()
            .map(|kmer| t1ha::t1ha2_atonce(kmer, sketch.seed))
            .collect();

        assert_eq!(hashes, expected);
        assert_eq!(metrics.input_bases, 6);
        assert_eq!(metrics.hashes_seen, 3);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn cpu_canonical_hashes_are_reverse_complement_invariant() {
        let forward = test_file("canonical-forward", b">record\nACGTA\n");
        let reverse = test_file("canonical-reverse", b">record\nTACGT\n");
        let sketch = test_sketch(3, true);

        let (forward_hashes, _) = extract_kmer_hashes(&forward, &sketch, |_| {}).unwrap();
        let (reverse_hashes, _) = extract_kmer_hashes(&reverse, &sketch, |_| {}).unwrap();

        assert_eq!(forward_hashes, reverse_hashes);
        fs::remove_file(forward).unwrap();
        fs::remove_file(reverse).unwrap();
    }
}
