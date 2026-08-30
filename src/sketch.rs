use anyhow::{Context, Result, anyhow, bail};
use log::info;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use needletail::{Sequence, parse_fastx_file};
use rayon::prelude::*;

use crate::types::*;
use crate::{dist, hd, utils};
use ultraloglog::UltraLogLog;

#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) -> Result<()> {
    utils::validate_sketch_params(&params)?;
    let sketch_wall_start = Instant::now();
    let inputs = utils::get_sketch_inputs(&params)?;
    let n_file = inputs.len();

    info!("Start sketching...");
    let pb = utils::get_progress_bar(n_file);

    let results: Vec<(FileSketch, Option<FileUllSketch>, FileSketchMetrics)> = inputs
        .par_iter()
        .map(|input| {
            let worker_start = Instant::now();
            let mut sketch = FileSketch {
                ksize: params.ksize,
                scaled: params.scaled,
                seed: params.seed,
                canonical: params.canonical,
                hv_d: params.hv_d,
                hv_quant_bits: 16u8,
                hv_norm_2: 0,
                file_str: input.file_id.clone(),
                hv: Vec::<i32>::new(),
            };

            let (kmer_hash_set, ull, mut metrics) =
                extract_kmer_hash_and_ull(&input.read_path, &sketch, params.ull_p)?;
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

            let ull_record = if params.if_ull {
                Some(FileUllSketch {
                    ksize: params.ksize,
                    canonical: params.canonical,
                    seed: params.seed,
                    ull_p: params.ull_p,
                    file_str: input.file_id.clone(),
                    ull_state: ull.get_state().to_vec(),
                })
            } else {
                None
            };

            pb.inc(1);
            pb.eta();

            metrics.total_worker_ns = worker_start.elapsed().as_nanos();

            Ok((sketch, ull_record, metrics))
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_and_clear();

    let all_filesketch: Vec<FileSketch> = results.iter().map(|(fs, _, _)| fs.clone()).collect();
    let all_ullsketch: Vec<FileUllSketch> =
        results.iter().filter_map(|(_, u, _)| u.clone()).collect();
    let all_metrics: Vec<FileSketchMetrics> = results.into_iter().map(|(_, _, m)| m).collect();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        n_file,
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file)?;

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file)?;
    }

    if let Some(prefix) = &params.metrics_out {
        utils::dump_sketch_metrics(&all_metrics, prefix, sketch_wall_start.elapsed().as_nanos())?;
    }

    Ok(())
}

fn extract_kmer_hash_and_ull(
    read_path: &Path,
    sketch: &FileSketch,
    ull_p: u32,
) -> Result<(HashSet<u64>, UltraLogLog, FileSketchMetrics)> {
    let ksize = sketch.ksize;
    let threshold = u64::MAX / sketch.scaled;
    let seed = sketch.seed;
    let mut metrics = FileSketchMetrics::default();

    let start = Instant::now();
    let mut fastx_reader = parse_fastx_file(read_path)
        .map_err(|e| anyhow!("failed to open FASTA/FASTQ {}: {e}", read_path.display()))?;
    metrics.fasta_ns = start.elapsed().as_nanos();

    let mut hash_set = HashSet::<u64>::new();
    let mut ull = UltraLogLog::new(ull_p)
        .map_err(|e| anyhow!("invalid UltraLogLog precision {ull_p}: {e}"))?;
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

                // ULL tracks the full hashed k-mer stream
                ull.add(h);
                metrics.hashes_seen += 1;
                valid_kmers += 1;
                // dothash tracks all hashed kmers
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

                // ULL tracks the full hashed k-mer stream
                ull.add(h);
                metrics.hashes_seen += 1;
                valid_kmers += 1;
                // dothash tracks all hashed kmers
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

    Ok((hash_set, ull, metrics))
}
