use log::info;
use std::collections::HashSet;
use std::path::Path;

use needletail::{Sequence, parse_fastx_file};
use rayon::prelude::*;

use crate::types::*;
use crate::{dist, hd, utils};
use ultraloglog::UltraLogLog;

#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) {
    let inputs = utils::get_sketch_inputs(&params).expect("failed to resolve sketch inputs");
    let n_file = inputs.len();

    info!("Start sketching...");
    let pb = utils::get_progress_bar(n_file);

    let results: Vec<(FileSketch, Option<FileUllSketch>)> = inputs
        .par_iter()
        .map(|input| {
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

            let (kmer_hash_set, ull) =
                extract_kmer_hash_and_ull(&input.read_path, &sketch, params.ull_p);

            let hv = if is_x86_feature_detected!("avx512f") {
                unsafe { hd::encode_hash_hd_avx512(&kmer_hash_set, &sketch) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { hd::encode_hash_hd_avx2(&kmer_hash_set, &sketch) }
            } else {
                hd::encode_hash_hd(&kmer_hash_set, &sketch)
            };

            sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);

            if params.if_compressed {
                sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, &hv) };
            } else {
                sketch.hv = hv.clone();
            }

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

            (sketch, ull_record)
        })
        .collect();

    pb.finish_and_clear();

    let all_filesketch: Vec<FileSketch> = results.iter().map(|(fs, _)| fs.clone()).collect();
    let all_ullsketch: Vec<FileUllSketch> = results.into_iter().filter_map(|(_, u)| u).collect();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        n_file,
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file);

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
    }
}

fn extract_kmer_hash_and_ull(
    read_path: &Path,
    sketch: &FileSketch,
    ull_p: u32,
) -> (HashSet<u64>, UltraLogLog) {
    let ksize = sketch.ksize;
    let threshold = u64::MAX / sketch.scaled;
    let seed = sketch.seed;

    let mut fastx_reader = parse_fastx_file(read_path).expect("Opening .fna files failed");

    let mut hash_set = HashSet::<u64>::new();
    let mut ull = UltraLogLog::new(ull_p).expect("Invalid UltraLogLog precision");

    while let Some(record) = fastx_reader.next() {
        let seqrec: needletail::parser::SequenceRecord<'_> = record.expect("invalid record");

        let norm_seq = seqrec.normalize(false);
        let rc = norm_seq.reverse_complement();

        for (_, kmer, _) in norm_seq.canonical_kmers(ksize, &rc) {
            let h = t1ha::t1ha2_atonce(kmer, seed);

            // ULL tracks the full hashed k-mer stream
            ull.add(h);
            // dothash tracks all hashed kmers
            hash_set.insert(h);
        }
    }

    (hash_set, ull)
}
