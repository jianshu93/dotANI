use ultraloglog::UltraLogLog;

use crate::hd;
use crate::types::*;
use crate::utils;

use anyhow::{Result, bail};
use log::{debug, info, warn};
use rayon::prelude::*;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
use crate::cuda_dot::{GpuDotExecutor, device_count};

pub fn dist(sketch_dist: &mut SketchDist) -> Result<()> {
    if !sketch_dist.ani_threshold.is_finite() || !(0.0..=100.0).contains(&sketch_dist.ani_threshold)
    {
        bail!("ANI threshold must be finite and in 0..=100");
    }
    if sketch_dist.threads == 0 {
        bail!("distance thread count must be greater than zero");
    }
    if sketch_dist.out_file.as_os_str().is_empty() {
        bail!("ANI output path must not be empty");
    }

    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    let ull_load_start = Instant::now();
    let ref_ull_sketch = utils::load_ull_sketch(sketch_dist.path_ref_ull.as_path())?;
    let query_ull_sketch = if if_sym {
        ref_ull_sketch.clone()
    } else {
        utils::load_ull_sketch(sketch_dist.path_query_ull.as_path())?
    };
    let ull_load_secs = ull_load_start.elapsed().as_secs_f32();

    let sketch_load_start = Instant::now();
    let mut ref_file_sketch = utils::load_sketch(sketch_dist.path_ref_sketch.as_path())?;
    let mut query_file_sketch = if if_sym {
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(sketch_dist.path_query_sketch.as_path())?
    };
    let sketch_load_secs = sketch_load_start.elapsed().as_secs_f32();

    let validation_start = Instant::now();

    assert_eq!(
        ref_file_sketch.len(),
        ref_ull_sketch.len(),
        "Ref HD and ULL sketch counts differ"
    );
    assert_eq!(
        query_file_sketch.len(),
        query_ull_sketch.len(),
        "Query HD and ULL sketch counts differ"
    );

    for i in 0..ref_file_sketch.len() {
        assert_eq!(
            ref_file_sketch[i].file_str, ref_ull_sketch[i].file_str,
            "Ref HD/ULL file order mismatch"
        );
    }
    for i in 0..query_file_sketch.len() {
        assert_eq!(
            query_file_sketch[i].file_str, query_ull_sketch[i].file_str,
            "Query HD/ULL file order mismatch"
        );
    }

    let ksize_ref = ref_file_sketch[0].ksize;
    let ksize_query = query_file_sketch[0].ksize;
    assert_eq!(
        ksize_ref, ksize_query,
        "Ref and query sketches use different kmer sizes!"
    );

    let hv_d_ref = ref_file_sketch[0].hv_d;
    let hv_d_query = query_file_sketch[0].hv_d;
    assert_eq!(
        hv_d_ref, hv_d_query,
        "Ref and query sketches use different HV dimensions!"
    );

    let validation_secs = validation_start.elapsed().as_secs_f32();

    let decompress_start = Instant::now();
    hd::decompress_file_sketch(&mut ref_file_sketch)?;
    hd::decompress_file_sketch(&mut query_file_sketch)?;
    let decompress_secs = decompress_start.elapsed().as_secs_f32();

    let compute_start = Instant::now();

    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        &query_file_sketch,
        &ref_ull_sketch,
        &query_ull_sketch,
        ksize_ref,
        if_sym,
    )?;
    let compute_secs = compute_start.elapsed().as_secs_f32();

    debug!(
        "dist phase timings: ull_load={:.3}s sketch_load={:.3}s validation={:.3}s decompress={:.3}s compute_write={:.3}s",
        ull_load_secs, sketch_load_secs, validation_secs, decompress_secs, compute_secs,
    );
    info!(
        "dist total={:.3}s refs={} queries={}",
        tstart.elapsed().as_secs_f32(),
        ref_file_sketch.len(),
        query_file_sketch.len()
    );

    Ok(())
}

pub fn compute_hv_l2_norm(hv: &[i32]) -> i64 {
    hv.iter()
        .map(|&num| {
            let x = num as i64;
            x * x
        })
        .sum()
}

#[inline]
pub fn ull_cardinality_from_state(state: &[u8]) -> f64 {
    let ull = UltraLogLog::wrap(state.to_vec()).expect("Invalid UltraLogLog state");
    ull.get_distinct_count_estimate()
}

#[inline]
pub fn compute_pairwise_dot(r: &[i32], q: &[i32]) -> i64 {
    r.iter()
        .zip(q.iter())
        .map(|(x, y)| (*x as i64) * (*y as i64))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_dot_avx2(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n8 = len / 8;

    let mut acc_even = _mm256_setzero_si256();
    let mut acc_odd = _mm256_setzero_si256();

    for i in 0..n8 {
        let base = i * 8;

        let vr = unsafe { _mm256_loadu_si256(r.as_ptr().add(base) as *const __m256i) };
        let vq = unsafe { _mm256_loadu_si256(q.as_ptr().add(base) as *const __m256i) };

        let prod_even = _mm256_mul_epi32(vr, vq);

        let vr_shift = _mm256_srli_epi64(vr, 32);
        let vq_shift = _mm256_srli_epi64(vq, 32);
        let prod_odd = _mm256_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm256_add_epi64(acc_even, prod_even);
        acc_odd = _mm256_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm256_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 4];
    unsafe { _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc) };

    let mut sum = tmp.iter().sum::<i64>();

    for i in (n8 * 8)..len {
        sum += (r[i] as i64) * (q[i] as i64);
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pairwise_dot_avx512(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n16 = len / 16;

    let mut acc_even = _mm512_setzero_si512();
    let mut acc_odd = _mm512_setzero_si512();

    for i in 0..n16 {
        let base = i * 16;

        let vr = unsafe { _mm512_loadu_si512(r.as_ptr().add(base) as *const __m512i) };
        let vq = unsafe { _mm512_loadu_si512(q.as_ptr().add(base) as *const __m512i) };

        let prod_even = _mm512_mul_epi32(vr, vq);

        let vr_shift = _mm512_srli_epi64(vr, 32);
        let vq_shift = _mm512_srli_epi64(vq, 32);
        let prod_odd = _mm512_mul_epi32(vr_shift, vq_shift);

        acc_even = _mm512_add_epi64(acc_even, prod_even);
        acc_odd = _mm512_add_epi64(acc_odd, prod_odd);
    }

    let acc = _mm512_add_epi64(acc_even, acc_odd);
    let mut tmp = [0i64; 8];
    unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc) };

    let mut sum = tmp.iter().sum::<i64>();

    for i in (n16 * 16)..len {
        sum += (r[i] as i64) * (q[i] as i64);
    }

    sum
}

#[inline]
pub fn ani_from_intersection_and_cardinalities(
    inter_hat: f64,
    card_r: f64,
    card_q: f64,
    ksize: u8,
) -> f32 {
    if ksize == 0 || !inter_hat.is_finite() || !card_r.is_finite() || !card_q.is_finite() {
        return 0.0;
    }
    if inter_hat <= 0.0 {
        return 0.0;
    }

    let union_hat = card_r + card_q - inter_hat;
    if union_hat <= 0.0 {
        return 0.0;
    }

    let jaccard = inter_hat / union_hat;
    if !jaccard.is_finite() || jaccard <= 0.0 {
        return 0.0;
    }

    if jaccard > 1.0 {
        return 100.0;
    }

    let ani = (2.0 * jaccard as f32 / (1.0 + jaccard as f32)).powf(1.0 / ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.clamp(0.0, 1.0) * 100.0
    }
}

#[inline]
#[cfg(any(not(feature = "cuda"), test))]
fn compute_pairwise_dot_best(r: &[i32], q: &[i32]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { compute_pairwise_dot_avx512(r, q) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_pairwise_dot_avx2(r, q) };
        }
    }

    compute_pairwise_dot(r, q)
}

#[cfg(feature = "cuda")]
fn flatten_hv_matrix(filesketch: &[FileSketch]) -> Vec<i32> {
    if filesketch.is_empty() {
        return Vec::new();
    }

    let hv_d = filesketch[0].hv_d;
    let mut out = Vec::with_capacity(filesketch.len() * hv_d);
    for fs in filesketch {
        out.extend_from_slice(&fs.hv);
    }
    out
}

#[cfg(any(not(feature = "cuda"), test))]
fn stream_hv_ani_cpu(
    out_path: &std::path::Path,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
) -> usize {
    const ROW_BLOCK: usize = 32;
    const FLUSH_BYTES: usize = 8 * 1024 * 1024;

    let writer = Arc::new(Mutex::new(BufWriter::new(
        File::create(out_path).expect("Failed to create ANI output file"),
    )));
    let total_hits = AtomicUsize::new(0);
    let total_ani_evals = AtomicUsize::new(0);
    let total_nonpositive_skipped = AtomicUsize::new(0);

    let row_starts: Vec<usize> = (0..ref_filesketch.len()).step_by(ROW_BLOCK).collect();

    row_starts.into_par_iter().for_each(|i0| {
        let i1 = (i0 + ROW_BLOCK).min(ref_filesketch.len());

        let mut local_text = String::with_capacity(1 << 20);
        let mut local_hits = 0usize;
        let mut local_pairs_done = 0usize;
        let mut local_ani_evals = 0usize;
        let mut local_nonpositive_skipped = 0usize;

        for i in i0..i1 {
            let j_start = if if_symmetric { i + 1 } else { 0 };

            for j in j_start..query_filesketch.len() {
                let r = &ref_filesketch[i];
                let q = &query_filesketch[j];

                let dot = compute_pairwise_dot_best(&r.hv, &q.hv) as f64;
                let inter_hat = dot / r.hv_d as f64;
                if inter_hat <= 0.0 && ani_threshold > 0.0 {
                    local_nonpositive_skipped += 1;
                    local_pairs_done += 1;
                    continue;
                }

                local_ani_evals += 1;
                let ani = ani_from_intersection_and_cardinalities(
                    inter_hat,
                    ref_cards[i],
                    query_cards[j],
                    ksize,
                );

                if ani >= ani_threshold {
                    use std::fmt::Write as _;
                    let _ = writeln!(
                        &mut local_text,
                        "{}\t{}\t{:.3}",
                        r.file_str, q.file_str, ani
                    );
                    local_hits += 1;
                }

                local_pairs_done += 1;

                if local_text.len() >= FLUSH_BYTES {
                    let mut guard = writer.lock().expect("ANI writer mutex poisoned");
                    guard
                        .write_all(local_text.as_bytes())
                        .expect("Failed to write ANI batch");
                    local_text.clear();
                }
            }
        }

        if !local_text.is_empty() {
            let mut guard = writer.lock().expect("ANI writer mutex poisoned");
            guard
                .write_all(local_text.as_bytes())
                .expect("Failed to write ANI batch");
        }

        total_hits.fetch_add(local_hits, Ordering::Relaxed);
        total_ani_evals.fetch_add(local_ani_evals, Ordering::Relaxed);
        total_nonpositive_skipped.fetch_add(local_nonpositive_skipped, Ordering::Relaxed);
        pb.inc(local_pairs_done as u64);
    });

    writer
        .lock()
        .expect("ANI writer mutex poisoned")
        .flush()
        .expect("Failed to flush ANI output");

    info!(
        "cpu stream breakdown: hits={} ani_evals={} nonpositive_skipped={}",
        total_hits.load(Ordering::Relaxed),
        total_ani_evals.load(Ordering::Relaxed),
        total_nonpositive_skipped.load(Ordering::Relaxed)
    );

    total_hits.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct GpuTileJob {
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
}

#[cfg(feature = "cuda")]
struct TileBatchResult {
    text: String,
    num_hits: usize,
    num_pairs_done: usize,
    candidate_pairs: usize,
    prefilter_skipped: usize,
    ani_evals: usize,
    nonpositive_skipped: usize,
    text_bytes: usize,
    ref_flatten_events: usize,
    flatten_ref_ns: u128,
    flatten_query_ns: u128,
    query_h2d_ns: u128,
    ref_h2d_ns: u128,
    compute_d2h_ns: u128,
    gpu_tile_total_ns: u128,
    postprocess_ns: u128,
    query_h2d_bytes: usize,
    ref_h2d_bytes: usize,
    out_d2h_bytes: usize,
    ref_uploads: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct GpuStreamBreakdown {
    jobs: usize,
    pairs: usize,
    hits: usize,
    candidates: usize,
    prefilter_skipped: usize,
    ani_evals: usize,
    nonpositive_skipped: usize,
    output_bytes: usize,
    ref_flatten_events: usize,
    flatten_ref_ns: u128,
    flatten_query_ns: u128,
    query_h2d_ns: u128,
    ref_h2d_ns: u128,
    compute_d2h_ns: u128,
    gpu_tile_total_ns: u128,
    postprocess_ns: u128,
    write_ns: u128,
    query_h2d_bytes: usize,
    ref_h2d_bytes: usize,
    out_d2h_bytes: usize,
    ref_uploads: usize,
}

#[cfg(feature = "cuda")]
impl GpuStreamBreakdown {
    fn add_batch(&mut self, batch: &TileBatchResult) {
        self.jobs += 1;
        self.pairs += batch.num_pairs_done;
        self.hits += batch.num_hits;
        self.candidates += batch.candidate_pairs;
        self.prefilter_skipped += batch.prefilter_skipped;
        self.ani_evals += batch.ani_evals;
        self.nonpositive_skipped += batch.nonpositive_skipped;
        self.output_bytes += batch.text_bytes;
        self.ref_flatten_events += batch.ref_flatten_events;
        self.flatten_ref_ns += batch.flatten_ref_ns;
        self.flatten_query_ns += batch.flatten_query_ns;
        self.query_h2d_ns += batch.query_h2d_ns;
        self.ref_h2d_ns += batch.ref_h2d_ns;
        self.compute_d2h_ns += batch.compute_d2h_ns;
        self.gpu_tile_total_ns += batch.gpu_tile_total_ns;
        self.postprocess_ns += batch.postprocess_ns;
        self.query_h2d_bytes += batch.query_h2d_bytes;
        self.ref_h2d_bytes += batch.ref_h2d_bytes;
        self.out_d2h_bytes += batch.out_d2h_bytes;
        self.ref_uploads += batch.ref_uploads;
    }
}

#[cfg(feature = "cuda")]
#[inline]
fn ns_to_secs(ns: u128) -> f64 {
    ns as f64 / 1_000_000_000.0
}

#[cfg(feature = "cuda")]
fn stream_hv_ani_gpu_multi(
    writer: &mut BufWriter<File>,
    pb: &indicatif::ProgressBar,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_cards: &[f64],
    query_cards: &[f64],
    ksize: u8,
    if_symmetric: bool,
    ani_threshold: f32,
) -> anyhow::Result<usize> {
    if ref_filesketch.is_empty() || query_filesketch.is_empty() {
        return Ok(0);
    }

    let hv_d = ref_filesketch[0].hv_d;
    let tile_ref = 512usize;
    let tile_query = 512usize;

    let ng = device_count()?.max(1);
    info!("Using {} GPU worker(s) for tiled dot-product", ng);

    let mut jobs = Vec::<GpuTileJob>::new();
    for i0 in (0..ref_filesketch.len()).step_by(tile_ref) {
        let i1 = (i0 + tile_ref).min(ref_filesketch.len());
        let j0_start = if if_symmetric { i0 } else { 0 };

        for j0 in (j0_start..query_filesketch.len()).step_by(tile_query) {
            let j1 = (j0 + tile_query).min(query_filesketch.len());
            jobs.push(GpuTileJob { i0, i1, j0, j1 });
        }
    }

    let total_jobs = jobs.len();
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<anyhow::Result<TileBatchResult>>();

    std::thread::scope(|scope| -> anyhow::Result<usize> {
        for dev_id in 0..ng {
            let tx = tx.clone();
            let jobs = Arc::clone(&jobs);
            let next = Arc::clone(&next);

            scope.spawn(move || {
                let worker = || -> anyhow::Result<()> {
                    let mut gpu = GpuDotExecutor::new(dev_id)?;

                    let mut cached_i0 = usize::MAX;
                    let mut cached_i1 = usize::MAX;
                    let mut cached_ref_flat = Vec::<i32>::new();
                    let mut cached_nr = 0usize;

                    loop {
                        let job_idx = next.fetch_add(1, Ordering::Relaxed);
                        if job_idx >= jobs.len() {
                            break;
                        }

                        let job = jobs[job_idx];

                        let mut flatten_ref_ns = 0u128;
                        let mut ref_flatten_events = 0usize;
                        if job.i0 != cached_i0 || job.i1 != cached_i1 {
                            cached_i0 = job.i0;
                            cached_i1 = job.i1;

                            let ref_block = &ref_filesketch[job.i0..job.i1];
                            cached_nr = ref_block.len();
                            let flatten_ref_start = Instant::now();
                            cached_ref_flat = flatten_hv_matrix(ref_block);
                            flatten_ref_ns = flatten_ref_start.elapsed().as_nanos();
                            ref_flatten_events = 1;
                        }

                        let query_block = &query_filesketch[job.j0..job.j1];
                        let nq = query_block.len();
                        let flatten_query_start = Instant::now();
                        let query_flat = flatten_hv_matrix(query_block);
                        let flatten_query_ns = flatten_query_start.elapsed().as_nanos();

                        let mut tile_dots = vec![0i64; nq * cached_nr];
                        let gpu_timings = gpu.compute_tile(
                            &query_flat,
                            nq,
                            &cached_ref_flat,
                            cached_nr,
                            hv_d,
                            &mut tile_dots,
                            ref_flatten_events > 0,
                        )?;

                        let postprocess_start = Instant::now();
                        let mut text = String::new();
                        let mut num_hits = 0usize;
                        let mut num_pairs_done = 0usize;
                        let mut ani_evals = 0usize;
                        let mut nonpositive_skipped = 0usize;

                        for q_local in 0..nq {
                            for r_local in 0..cached_nr {
                                let i = job.i0 + r_local;
                                let j = job.j0 + q_local;

                                if if_symmetric && i >= j {
                                    continue;
                                }

                                num_pairs_done += 1;

                                let dot = tile_dots[q_local * cached_nr + r_local] as f64;
                                let inter_hat = dot / hv_d as f64;
                                if inter_hat <= 0.0 && ani_threshold > 0.0 {
                                    nonpositive_skipped += 1;
                                    continue;
                                }

                                ani_evals += 1;
                                let ani = ani_from_intersection_and_cardinalities(
                                    inter_hat,
                                    ref_cards[i],
                                    query_cards[j],
                                    ksize,
                                );

                                if ani >= ani_threshold {
                                    use std::fmt::Write as _;
                                    let _ = writeln!(
                                        &mut text,
                                        "{}\t{}\t{:.3}",
                                        ref_filesketch[i].file_str,
                                        query_filesketch[j].file_str,
                                        ani
                                    );
                                    num_hits += 1;
                                }
                            }
                        }
                        let postprocess_ns = postprocess_start.elapsed().as_nanos();
                        let text_bytes = text.len();

                        let _ = tx.send(Ok(TileBatchResult {
                            text,
                            num_hits,
                            num_pairs_done,
                            candidate_pairs: num_pairs_done,
                            prefilter_skipped: 0,
                            ani_evals,
                            nonpositive_skipped,
                            text_bytes,
                            ref_flatten_events,
                            flatten_ref_ns,
                            flatten_query_ns,
                            query_h2d_ns: gpu_timings.query_h2d_ns,
                            ref_h2d_ns: gpu_timings.ref_h2d_ns,
                            compute_d2h_ns: gpu_timings.compute_d2h_ns,
                            gpu_tile_total_ns: gpu_timings.total_ns,
                            postprocess_ns,
                            query_h2d_bytes: gpu_timings.query_h2d_bytes,
                            ref_h2d_bytes: gpu_timings.ref_h2d_bytes,
                            out_d2h_bytes: gpu_timings.out_d2h_bytes,
                            ref_uploads: usize::from(gpu_timings.ref_upload_performed),
                        }));
                    }

                    Ok(())
                };

                if let Err(e) = worker() {
                    let _ = tx.send(Err(e));
                }
            });
        }

        drop(tx);

        let mut total_hits = 0usize;
        let mut received_jobs = 0usize;
        let mut first_error = None;
        let stream_wall_start = Instant::now();
        let mut breakdown = GpuStreamBreakdown::default();

        while received_jobs < total_jobs {
            match rx.recv() {
                Ok(Ok(batch)) => {
                    received_jobs += 1;
                    let write_start = Instant::now();
                    writer
                        .write_all(batch.text.as_bytes())
                        .expect("Failed to write ANI batch");
                    breakdown.write_ns += write_start.elapsed().as_nanos();
                    total_hits += batch.num_hits;
                    pb.inc(batch.num_pairs_done as u64);
                    breakdown.add_batch(&batch);
                }
                Ok(Err(e)) => {
                    first_error.get_or_insert(e);
                }
                Err(e) => {
                    first_error.get_or_insert_with(|| {
                        anyhow::anyhow!("GPU worker channel closed unexpectedly: {e}")
                    });
                    break;
                }
            }
        }

        if let Some(e) = first_error {
            Err(e)
        } else {
            // Worker timings are aggregate worker-sums; with multiple GPUs they can exceed wall.
            info!(
                "gpu stream breakdown: jobs={} pairs={} hits={} candidates={} prefilter_skipped={} ani_evals={} nonpositive_skipped={} output_mb={:.3} ref_flatten_events={} ref_uploads={} query_h2d_mb={:.3} ref_h2d_mb={:.3} out_d2h_mb={:.3} flatten_ref_cache_miss={:.3}s flatten_query={:.3}s query_h2d={:.3}s ref_h2d={:.3}s compute_d2h={:.3}s gpu_tile_total={:.3}s postprocess={:.3}s write={:.3}s wall={:.3}s",
                breakdown.jobs,
                breakdown.pairs,
                breakdown.hits,
                breakdown.candidates,
                breakdown.prefilter_skipped,
                breakdown.ani_evals,
                breakdown.nonpositive_skipped,
                breakdown.output_bytes as f64 / (1024.0 * 1024.0),
                breakdown.ref_flatten_events,
                breakdown.ref_uploads,
                breakdown.query_h2d_bytes as f64 / (1024.0 * 1024.0),
                breakdown.ref_h2d_bytes as f64 / (1024.0 * 1024.0),
                breakdown.out_d2h_bytes as f64 / (1024.0 * 1024.0),
                ns_to_secs(breakdown.flatten_ref_ns),
                ns_to_secs(breakdown.flatten_query_ns),
                ns_to_secs(breakdown.query_h2d_ns),
                ns_to_secs(breakdown.ref_h2d_ns),
                ns_to_secs(breakdown.compute_d2h_ns),
                ns_to_secs(breakdown.gpu_tile_total_ns),
                ns_to_secs(breakdown.postprocess_ns),
                ns_to_secs(breakdown.write_ns),
                stream_wall_start.elapsed().as_secs_f64(),
            );
            Ok(total_hits)
        }
    })
}

pub fn compute_hv_ani(
    sketch_dist: &mut SketchDist,
    ref_filesketch: &[FileSketch],
    query_filesketch: &[FileSketch],
    ref_ull_sketch: &[FileUllSketch],
    query_ull_sketch: &[FileUllSketch],
    ksize: u8,
    if_symmetric: bool,
) -> Result<()> {
    info!("Computing ANI..");

    let num_ref_files = ref_filesketch.len();
    let num_query_files = query_filesketch.len();

    let num_dists = if if_symmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb = utils::get_progress_bar(num_dists);

    let compute_start = Instant::now();
    let cardinality_start = Instant::now();
    let ref_cards: Vec<f64> = ref_ull_sketch
        .par_iter()
        .map(|s| ull_cardinality_from_state(&s.ull_state))
        .collect();

    let query_cards: Vec<f64> = if if_symmetric {
        ref_cards.clone()
    } else {
        query_ull_sketch
            .par_iter()
            .map(|s| ull_cardinality_from_state(&s.ull_state))
            .collect()
    };
    let cardinality_secs = cardinality_start.elapsed().as_secs_f32();

    let stream_start = Instant::now();
    #[cfg(feature = "cuda")]
    let (num_hits, output_open_secs, flush_secs, stream_mode) = {
        let output_open_start = Instant::now();
        let out_file =
            File::create(sketch_dist.out_file.as_path()).expect("Failed to create ANI output file");
        let mut writer = BufWriter::new(out_file);
        let output_open_secs = output_open_start.elapsed().as_secs_f32();

        let n = stream_hv_ani_gpu_multi(
            &mut writer,
            &pb,
            ref_filesketch,
            query_filesketch,
            &ref_cards,
            &query_cards,
            ksize,
            if_symmetric,
            sketch_dist.ani_threshold,
        )?;
        let flush_start = Instant::now();
        writer.flush().expect("Failed to flush ANI output");
        let flush_secs = flush_start.elapsed().as_secs_f32();
        info!("Multi-GPU tiled dot-product completed successfully");
        (n, output_open_secs, flush_secs, "gpu")
    };

    #[cfg(not(feature = "cuda"))]
    let (num_hits, output_open_secs, flush_secs, stream_mode) = {
        let n = stream_hv_ani_cpu(
            sketch_dist.out_file.as_path(),
            &pb,
            ref_filesketch,
            query_filesketch,
            &ref_cards,
            &query_cards,
            ksize,
            if_symmetric,
            sketch_dist.ani_threshold,
        );
        (n, 0.0, 0.0, "cpu")
    };
    let stream_secs = stream_start.elapsed().as_secs_f32();

    pb.finish_and_clear();

    let total_dist = num_dists as u64;
    let cnt = num_hits as u64;
    let perc = if total_dist > 0 {
        (cnt as f64) / (total_dist as f64) * 100.0
    } else {
        0.0
    };

    if perc < 5.0 {
        warn!(
            "Output ANIs with threshold {:.1} are too divergent: {} of {} ({:.2}%) ANIs are reported",
            sketch_dist.ani_threshold, cnt, total_dist, perc
        );
    } else {
        info!(
            "Output {} of {} ANIs above threshold {:.1} to file {}",
            cnt,
            total_dist,
            sketch_dist.ani_threshold,
            sketch_dist.out_file.to_string_lossy()
        );
    }

    let summary_start = Instant::now();
    let summary_secs = summary_start.elapsed().as_secs_f32();

    info!(
        "compute_hv_ani timings: cardinality={:.3}s output_open={:.3}s stream_mode={} stream={:.3}s flush={:.3}s summary={:.3}s total={:.3}s",
        cardinality_secs,
        output_open_secs,
        stream_mode,
        stream_secs,
        flush_secs,
        summary_secs,
        compute_start.elapsed().as_secs_f32()
    );

    Ok(())
}

#[cfg(test)]
mod tests;
