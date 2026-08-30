use crate::types::*;

#[cfg(feature = "cuda")]
use {
    crate::{dist, fastx_reader, fastx_reader::ReaderGate, hd, hd_cuda, utils},
    anyhow::{Result, anyhow},
    cudarc::{
        driver::{
            CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
            PushKernelArg, sys,
        },
        nvrtc::Ptx,
    },
    log::info,
    rayon::prelude::*,
    std::collections::HashSet,
    std::path::{Path, PathBuf},
    std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    std::time::Instant,
    ultraloglog::UltraLogLog,
};

#[cfg(feature = "cuda")]
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

#[cfg(feature = "cuda")]
const SEQ_NT4_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

#[cfg(feature = "cuda")]
struct IndexedSketchResult {
    index: usize,
    sketch: FileSketch,
    ull_record: Option<FileUllSketch>,
    metrics: FileSketchMetrics,
}

#[cfg(feature = "cuda")]
struct CudaSketchLaneScratch {
    lane_id: usize,
    dev_id: usize,
    _ctx: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    kmer_fn: CudaFunction,
    hd_fn: CudaFunction,
    full_hashes: Vec<u64>,
    sampled_hashes: Vec<u64>,
    sampled_hash_set: HashSet<u64>,
    host_kmer_hash: Vec<u64>,
    host_kmer_counts: Vec<u32>,
    hv_host: Vec<i32>,
    d_seq: Option<CudaSlice<u8>>,
    d_kmer_hash: Option<CudaSlice<u64>>,
    d_kmer_counts: Option<CudaSlice<u32>>,
    d_hd_hashes: Option<CudaSlice<u64>>,
    d_hv: Option<CudaSlice<i32>>,
    kmer_h2d_start: CudaEvent,
    kmer_h2d_end: CudaEvent,
    kmer_kernel_start: CudaEvent,
    kmer_kernel_end: CudaEvent,
    kmer_d2h_start: CudaEvent,
    kmer_d2h_end: CudaEvent,
    hd_hash_h2d_start: CudaEvent,
    hd_hash_h2d_end: CudaEvent,
    hd_hv_h2d_start: CudaEvent,
    hd_hv_h2d_end: CudaEvent,
    hd_kernel_start: CudaEvent,
    hd_kernel_end: CudaEvent,
    hd_d2h_start: CudaEvent,
    hd_d2h_end: CudaEvent,
}

#[cfg(feature = "cuda")]
impl CudaSketchLaneScratch {
    fn new(lane_id: usize, dev_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(dev_id)?;
        let module = ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))?;
        let stream = ctx.default_stream();
        let kmer_fn = module.load_function("cuda_kmer_t1ha2")?;
        let hd_fn = module.load_function("cuda_hd_encode_counts_direct")?;
        let event_flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let kmer_h2d_start = ctx.new_event(event_flags)?;
        let kmer_h2d_end = ctx.new_event(event_flags)?;
        let kmer_kernel_start = ctx.new_event(event_flags)?;
        let kmer_kernel_end = ctx.new_event(event_flags)?;
        let kmer_d2h_start = ctx.new_event(event_flags)?;
        let kmer_d2h_end = ctx.new_event(event_flags)?;
        let hd_hash_h2d_start = ctx.new_event(event_flags)?;
        let hd_hash_h2d_end = ctx.new_event(event_flags)?;
        let hd_hv_h2d_start = ctx.new_event(event_flags)?;
        let hd_hv_h2d_end = ctx.new_event(event_flags)?;
        let hd_kernel_start = ctx.new_event(event_flags)?;
        let hd_kernel_end = ctx.new_event(event_flags)?;
        let hd_d2h_start = ctx.new_event(event_flags)?;
        let hd_d2h_end = ctx.new_event(event_flags)?;

        Ok(Self {
            lane_id,
            dev_id,
            _ctx: ctx,
            _module: module,
            stream,
            kmer_fn,
            hd_fn,
            full_hashes: Vec::new(),
            sampled_hashes: Vec::new(),
            sampled_hash_set: HashSet::new(),
            host_kmer_hash: Vec::new(),
            host_kmer_counts: Vec::new(),
            hv_host: Vec::new(),
            d_seq: None,
            d_kmer_hash: None,
            d_kmer_counts: None,
            d_hd_hashes: None,
            d_hv: None,
            kmer_h2d_start,
            kmer_h2d_end,
            kmer_kernel_start,
            kmer_kernel_end,
            kmer_d2h_start,
            kmer_d2h_end,
            hd_hash_h2d_start,
            hd_hash_h2d_end,
            hd_hv_h2d_start,
            hd_hv_h2d_end,
            hd_kernel_start,
            hd_kernel_end,
            hd_d2h_start,
            hd_d2h_end,
        })
    }

    fn ensure_seq_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_seq.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity = grow_capacity(self.d_seq.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_seq = Some(unsafe { self.stream.alloc::<u8>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_kmer_hash_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_kmer_hash.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity =
                grow_capacity(self.d_kmer_hash.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_kmer_hash = Some(self.stream.alloc_zeros::<u64>(capacity)?);
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_kmer_count_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_kmer_counts.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity = grow_capacity(
                self.d_kmer_counts.as_ref().map_or(0, |buf| buf.len()),
                needed,
            );
            self.d_kmer_counts = Some(self.stream.alloc_zeros::<u32>(capacity)?);
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_hd_hash_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_hd_hashes.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity =
                grow_capacity(self.d_hd_hashes.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_hd_hashes = Some(unsafe { self.stream.alloc::<u64>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }

    fn ensure_hv_capacity(&mut self, needed: usize) -> Result<u128> {
        let start = Instant::now();
        if self.d_hv.as_ref().map_or(0, |buf| buf.len()) < needed {
            let capacity = grow_capacity(self.d_hv.as_ref().map_or(0, |buf| buf.len()), needed);
            self.d_hv = Some(unsafe { self.stream.alloc::<i32>(capacity)? });
        }
        Ok(start.elapsed().as_nanos())
    }
}

#[cfg(feature = "cuda")]
fn grow_capacity(current: usize, needed: usize) -> usize {
    if needed == 0 {
        return current;
    }

    let mut capacity = current.max(1);
    while capacity < needed {
        capacity = capacity.saturating_mul(2);
        if capacity == usize::MAX {
            break;
        }
    }
    capacity
}

#[allow(unused_variables)]
#[cfg(not(feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) -> anyhow::Result<()> {
    let _ = params;
    Err(anyhow::anyhow!(
        "CUDA sketching is unavailable because this binary was built without the cuda feature"
    ))
}

#[cfg(all(target_arch = "x86_64", feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) -> Result<()> {
    let sketch_wall_start = Instant::now();
    let inputs = utils::get_sketch_inputs(&params)?;
    let n_file = inputs.len();

    info!("Start GPU sketching...");
    let pb = utils::get_progress_bar(n_file);

    let device_ids = visible_cuda_device_ids()?;
    let lane_count = params.threads.max(1).min(n_file);
    info!(
        "Using {} GPU worker host lane(s) for sketching across {} usable CUDA device(s)",
        lane_count,
        device_ids.len()
    );
    info!(
        "Using CUDA dedup strategy: {}",
        params.cuda_dedup_strategy.as_str()
    );
    let reader_gate = if let Some(limit) = params.max_readers {
        info!("Limiting concurrent FASTA readers to {}", limit);
        Some(Arc::new(ReaderGate::new(limit)))
    } else {
        None
    };

    let next_file = Arc::new(AtomicUsize::new(0));
    let stop_workers = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<IndexedSketchResult>>();
    let mut result_slots: Vec<Option<IndexedSketchResult>> = (0..n_file).map(|_| None).collect();
    let mut worker_error = None;

    std::thread::scope(|scope| {
        for lane_id in 0..lane_count {
            let dev_id = device_ids[lane_id % device_ids.len()];
            let inputs = &inputs;
            let params = &params;
            let next_file = Arc::clone(&next_file);
            let stop_workers = Arc::clone(&stop_workers);
            let reader_gate = reader_gate.clone();
            let tx = tx.clone();

            scope.spawn(move || {
                let worker = || -> Result<()> {
                    let mut scratch = CudaSketchLaneScratch::new(lane_id, dev_id)?;

                    loop {
                        if stop_workers.load(Ordering::Relaxed) {
                            break;
                        }

                        let index = next_file.fetch_add(1, Ordering::Relaxed);
                        if index >= inputs.len() {
                            break;
                        }

                        let result = sketch_one_file_cuda(
                            index,
                            &inputs[index],
                            params,
                            reader_gate.as_ref(),
                            &mut scratch,
                        )?;
                        if tx.send(Ok(result)).is_err() {
                            break;
                        }
                    }

                    Ok(())
                };

                if let Err(e) = worker() {
                    stop_workers.store(true, Ordering::Relaxed);
                    let _ = tx.send(Err(e));
                }
            });
        }

        drop(tx);

        let mut received = 0usize;
        while received < n_file {
            match rx.recv() {
                Ok(Ok(result)) => {
                    if let Err(e) = store_indexed_sketch_result(&mut result_slots, result) {
                        worker_error = Some(e);
                        stop_workers.store(true, Ordering::Relaxed);
                        break;
                    }

                    received += 1;
                    pb.inc(1);
                    pb.eta();
                }
                Ok(Err(e)) => {
                    worker_error = Some(e);
                    stop_workers.store(true, Ordering::Relaxed);
                    break;
                }
                Err(e) => {
                    worker_error = Some(anyhow!(
                        "CUDA sketch worker channel closed before all files finished: {}",
                        e
                    ));
                    stop_workers.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
    });

    if let Some(e) = worker_error {
        return Err(e);
    }

    pb.finish_and_clear();

    let results = ordered_indexed_sketch_results(result_slots)?;

    let mut all_filesketch = Vec::with_capacity(results.len());
    let mut all_ullsketch = Vec::with_capacity(results.len());
    let mut all_metrics = Vec::with_capacity(results.len());
    for result in results {
        all_filesketch.push(result.sketch);
        if let Some(ull_record) = result.ull_record {
            all_ullsketch.push(ull_record);
        }
        all_metrics.push(result.metrics);
    }

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        inputs.len(),
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file);

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
    }

    if let Some(prefix) = &params.metrics_out {
        utils::dump_sketch_metrics(&all_metrics, prefix, sketch_wall_start.elapsed().as_nanos())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn visible_cuda_device_ids() -> Result<Vec<usize>> {
    let count = CudaContext::device_count()? as usize;
    if count == 0 {
        return Err(anyhow!("No CUDA devices are visible for GPU sketching"));
    }
    Ok((0..count).collect())
}

#[cfg(feature = "cuda")]
fn store_indexed_sketch_result(
    result_slots: &mut [Option<IndexedSketchResult>],
    result: IndexedSketchResult,
) -> Result<()> {
    let index = result.index;
    if index >= result_slots.len() {
        return Err(anyhow!(
            "CUDA sketch worker returned out-of-range file index {}",
            index
        ));
    }
    if result_slots[index].is_some() {
        return Err(anyhow!(
            "CUDA sketch worker returned duplicate file index {}",
            index
        ));
    }

    result_slots[index] = Some(result);
    Ok(())
}

#[cfg(feature = "cuda")]
fn ordered_indexed_sketch_results(
    result_slots: Vec<Option<IndexedSketchResult>>,
) -> Result<Vec<IndexedSketchResult>> {
    result_slots
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.ok_or_else(|| anyhow!("missing CUDA sketch result for file index {index}"))
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn sketch_one_file_cuda(
    index: usize,
    input: &SketchInput,
    params: &SketchParams,
    reader_gate: Option<&Arc<ReaderGate>>,
    scratch: &mut CudaSketchLaneScratch,
) -> Result<IndexedSketchResult> {
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

    let mut metrics =
        extract_kmer_t1ha2_cuda_full_hashes_into(&input.read_path, &sketch, reader_gate, scratch)?;
    metrics.file = sketch.file_str.clone();
    metrics.hashes_seen = scratch.full_hashes.len();
    if scratch.full_hashes.is_empty() {
        return Err(anyhow!(
            "input file {} produced no valid {}-mers",
            input.read_path.display(),
            params.ksize
        ));
    }

    let hash_and_dedup_start = Instant::now();
    scratch.sampled_hashes.clear();
    let mut restore_full_hashes_after_hd = false;
    let ull_record = if params.if_ull {
        let mut ull = UltraLogLog::new(params.ull_p).expect("Invalid UltraLogLog precision");
        match params.cuda_dedup_strategy {
            CudaDedupStrategy::HashSet => {
                scratch.sampled_hash_set.clear();
                for &hash in &scratch.full_hashes {
                    ull.add(hash);
                    scratch.sampled_hash_set.insert(hash);
                }
                scratch
                    .sampled_hashes
                    .extend(scratch.sampled_hash_set.iter().copied());
            }
            CudaDedupStrategy::SortUnstable => {
                for &hash in &scratch.full_hashes {
                    ull.add(hash);
                }
                scratch.full_hashes.sort_unstable();
                scratch.full_hashes.dedup();
                std::mem::swap(&mut scratch.sampled_hashes, &mut scratch.full_hashes);
                restore_full_hashes_after_hd = true;
            }
        }
        Some(FileUllSketch {
            ksize: params.ksize,
            canonical: params.canonical,
            seed: params.seed,
            ull_p: params.ull_p,
            file_str: sketch.file_str.clone(),
            ull_state: ull.get_state().to_vec(),
        })
    } else {
        match params.cuda_dedup_strategy {
            CudaDedupStrategy::HashSet => {
                scratch.sampled_hash_set.clear();
                for &hash in &scratch.full_hashes {
                    scratch.sampled_hash_set.insert(hash);
                }
                scratch
                    .sampled_hashes
                    .extend(scratch.sampled_hash_set.iter().copied());
            }
            CudaDedupStrategy::SortUnstable => {
                scratch
                    .sampled_hashes
                    .extend(scratch.full_hashes.iter().copied());
                scratch.sampled_hashes.sort_unstable();
                scratch.sampled_hashes.dedup();
            }
        }
        None
    };
    let hash_and_dedup_ns = hash_and_dedup_start.elapsed().as_nanos();
    metrics.hash_and_dedup_ns = hash_and_dedup_ns;
    metrics.cuda_filter_ns = Some(hash_and_dedup_ns);
    metrics.unique_hashes = scratch.sampled_hashes.len();

    let start = Instant::now();
    let hd_metrics = encode_hash_hd_cuda_into(scratch, sketch.hv_d)?;
    metrics.hd_encode_ns = start.elapsed().as_nanos();
    if !scratch.sampled_hashes.is_empty() && sketch.hv_d >= 64 {
        metrics.cuda_hd_hash_h2d_ns = Some(hd_metrics.cuda_hd_hash_h2d_ns);
        metrics.cuda_hd_hash_h2d_event_ns = Some(hd_metrics.cuda_hd_hash_h2d_event_ns);
        metrics.cuda_hd_hv_h2d_ns = Some(hd_metrics.cuda_hd_hv_h2d_ns);
        metrics.cuda_hd_hv_h2d_event_ns = Some(hd_metrics.cuda_hd_hv_h2d_event_ns);
        metrics.cuda_hd_alloc_ns = Some(hd_metrics.cuda_hd_alloc_ns);
        metrics.cuda_hd_kernel_launch_ns = Some(hd_metrics.cuda_hd_kernel_launch_ns);
        metrics.cuda_hd_kernel_event_ns = Some(hd_metrics.cuda_hd_kernel_event_ns);
        metrics.cuda_hd_d2h_ns = Some(hd_metrics.cuda_hd_d2h_ns);
        metrics.cuda_hd_d2h_event_ns = Some(hd_metrics.cuda_hd_d2h_event_ns);
    }

    let start = Instant::now();
    sketch.hv_norm_2 = dist::compute_hv_l2_norm(&scratch.hv_host);
    metrics.hv_norm_ns = start.elapsed().as_nanos();

    let start = Instant::now();
    if params.if_compressed {
        sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, &scratch.hv_host) };
    } else {
        sketch.hv = scratch.hv_host.clone();
    }
    metrics.hd_compress_ns = start.elapsed().as_nanos();
    metrics.total_worker_ns = worker_start.elapsed().as_nanos();

    if restore_full_hashes_after_hd {
        std::mem::swap(&mut scratch.sampled_hashes, &mut scratch.full_hashes);
    }

    Ok(IndexedSketchResult {
        index,
        sketch,
        ull_record,
        metrics,
    })
}

#[cfg(feature = "cuda")]
fn extract_kmer_t1ha2_cuda_full_hashes_into(
    read_path: &Path,
    sketch: &FileSketch,
    reader_gate: Option<&Arc<ReaderGate>>,
    scratch: &mut CudaSketchLaneScratch,
) -> Result<FileSketchMetrics> {
    scratch.full_hashes.clear();

    let wait_start = Instant::now();
    let permit = reader_gate.map(|gate| gate.acquire());
    let fasta_wait_ns = if permit.is_some() {
        wait_start.elapsed().as_nanos()
    } else {
        0
    };
    let fasta_start = Instant::now();
    let merged = fastx_reader::read_merge_seq(read_path);
    let fastx_reader::MergedSequence {
        sequence: fna_seqs,
        input_bases,
    } = merged;
    let fasta_ns = fasta_start.elapsed().as_nanos();
    drop(permit);

    let n_bps = fna_seqs.len();
    let mut metrics = FileSketchMetrics {
        input_bases,
        fasta_ns,
        fasta_wait_ns,
        cuda_stream_lane: Some(scratch.lane_id),
        cuda_device_id: Some(scratch.dev_id),
        ..FileSketchMetrics::default()
    };
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let seed = sketch.seed;

    if n_bps < ksize {
        return Ok(metrics);
    }

    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512usize;
    let n_threads = n_kmers.div_ceil(kmer_per_thread);

    let n_hash_per_thread = kmer_per_thread;
    let n_hash_array = n_hash_per_thread * n_threads;
    let seq_alloc_ns = scratch.ensure_seq_capacity(n_bps)?;
    let hash_alloc_ns = scratch.ensure_kmer_hash_capacity(n_hash_array)?;
    let count_alloc_ns = scratch.ensure_kmer_count_capacity(n_threads)?;
    metrics.cuda_alloc_ns = Some(seq_alloc_ns + hash_alloc_ns + count_alloc_ns);

    let gpu_seq = scratch
        .d_seq
        .as_mut()
        .ok_or_else(|| anyhow!("sequence device buffer allocation is missing"))?;
    let gpu_kmer_hash = scratch
        .d_kmer_hash
        .as_mut()
        .ok_or_else(|| anyhow!("k-mer hash device buffer allocation is missing"))?;
    let gpu_kmer_counts = scratch
        .d_kmer_counts
        .as_mut()
        .ok_or_else(|| anyhow!("k-mer count device buffer allocation is missing"))?;

    let h2d_start = Instant::now();
    scratch.kmer_h2d_start.record(&scratch.stream)?;
    scratch
        .stream
        .memcpy_htod(&fna_seqs, &mut gpu_seq.slice_mut(0..n_bps))?;
    scratch.kmer_h2d_end.record(&scratch.stream)?;
    metrics.cuda_h2d_ns = Some(h2d_start.elapsed().as_nanos());

    let mut builder = scratch.stream.launch_builder(&scratch.kmer_fn);
    builder.arg(&*gpu_seq);
    builder.arg(&n_bps);
    builder.arg(&kmer_per_thread);
    builder.arg(&n_hash_per_thread);
    builder.arg(&ksize);

    let full_threshold = u64::MAX;
    builder.arg(&full_threshold);

    builder.arg(&seed);
    builder.arg(&canonical);
    builder.arg(&n_threads);
    builder.arg(&mut *gpu_kmer_hash);
    builder.arg(&mut *gpu_kmer_counts);

    let launch_start = Instant::now();
    scratch.kmer_kernel_start.record(&scratch.stream)?;
    unsafe {
        builder.launch(LaunchConfig::for_num_elems(n_threads as u32))?;
    }
    scratch.kmer_kernel_end.record(&scratch.stream)?;
    metrics.cuda_launch_ns = Some(launch_start.elapsed().as_nanos());

    scratch.host_kmer_hash.resize(n_hash_array, 0);
    scratch.host_kmer_counts.resize(n_threads, 0);
    let d2h_start = Instant::now();
    scratch.kmer_d2h_start.record(&scratch.stream)?;
    scratch.stream.memcpy_dtoh(
        &gpu_kmer_hash.slice(0..n_hash_array),
        &mut scratch.host_kmer_hash[..n_hash_array],
    )?;
    scratch.stream.memcpy_dtoh(
        &gpu_kmer_counts.slice(0..n_threads),
        &mut scratch.host_kmer_counts[..n_threads],
    )?;
    scratch.kmer_d2h_end.record(&scratch.stream)?;
    metrics.cuda_d2h_ns = Some(d2h_start.elapsed().as_nanos());
    metrics.cuda_h2d_event_ns = Some(event_ms_to_ns(
        scratch.kmer_h2d_start.elapsed_ms(&scratch.kmer_h2d_end)?,
    ));
    metrics.cuda_kernel_event_ns = Some(event_ms_to_ns(
        scratch
            .kmer_kernel_start
            .elapsed_ms(&scratch.kmer_kernel_end)?,
    ));
    metrics.cuda_d2h_event_ns = Some(event_ms_to_ns(
        scratch.kmer_d2h_start.elapsed_ms(&scratch.kmer_d2h_end)?,
    ));

    let compact_start = Instant::now();
    for (thread, &count_u32) in scratch.host_kmer_counts.iter().enumerate().take(n_threads) {
        let count = count_u32 as usize;
        if count > n_hash_per_thread {
            return Err(anyhow!(
                "CUDA k-mer worker {thread} produced {count} hashes, exceeding its capacity {n_hash_per_thread}"
            ));
        }
        let start = thread * n_hash_per_thread;
        scratch
            .full_hashes
            .extend_from_slice(&scratch.host_kmer_hash[start..start + count]);
    }
    metrics.cuda_compact_ns = Some(compact_start.elapsed().as_nanos());

    Ok(metrics)
}

#[cfg(feature = "cuda")]
fn encode_hash_hd_cuda_into(
    scratch: &mut CudaSketchLaneScratch,
    hv_d: usize,
) -> Result<hd_cuda::GpuHdEncodeMetrics> {
    hd::validate_hv_dimension(hv_d)?;
    if scratch.sampled_hashes.len() > i32::MAX as usize {
        return Err(anyhow!(
            "too many hashes for i32 HD count vector: {}",
            scratch.sampled_hashes.len()
        ));
    }

    let mut metrics = hd_cuda::GpuHdEncodeMetrics::default();
    scratch.hv_host.resize(hv_d, 0);

    if scratch.sampled_hashes.is_empty() {
        scratch.hv_host[..hv_d].fill(0);
        return Ok(metrics);
    }

    scratch.hv_host[..hv_d].fill(-(scratch.sampled_hashes.len() as i32));
    let num_chunks = hv_d / 64;
    if num_chunks == 0 {
        return Ok(metrics);
    }

    let hash_alloc_ns = scratch.ensure_hd_hash_capacity(scratch.sampled_hashes.len())?;
    let hv_alloc_ns = scratch.ensure_hv_capacity(hv_d)?;
    metrics.cuda_hd_alloc_ns = hash_alloc_ns + hv_alloc_ns;

    let d_hashes = scratch
        .d_hd_hashes
        .as_mut()
        .ok_or_else(|| anyhow!("HD hash device buffer allocation is missing"))?;
    let d_hv = scratch
        .d_hv
        .as_mut()
        .ok_or_else(|| anyhow!("HD vector device buffer allocation is missing"))?;

    let hash_h2d_start = Instant::now();
    scratch.hd_hash_h2d_start.record(&scratch.stream)?;
    scratch.stream.memcpy_htod(
        &scratch.sampled_hashes,
        &mut d_hashes.slice_mut(0..scratch.sampled_hashes.len()),
    )?;
    scratch.hd_hash_h2d_end.record(&scratch.stream)?;
    metrics.cuda_hd_hash_h2d_ns = hash_h2d_start.elapsed().as_nanos();

    let hv_h2d_start = Instant::now();
    scratch.hd_hv_h2d_start.record(&scratch.stream)?;
    scratch
        .stream
        .memcpy_htod(&scratch.hv_host[..hv_d], &mut d_hv.slice_mut(0..hv_d))?;
    scratch.hd_hv_h2d_end.record(&scratch.stream)?;
    metrics.cuda_hd_hv_h2d_ns = hv_h2d_start.elapsed().as_nanos();

    let num_hashes = scratch.sampled_hashes.len() as i32;
    let hv_d_i32 = hv_d as i32;
    let cfg = LaunchConfig {
        grid_dim: (
            num_chunks as u32,
            scratch.sampled_hashes.len().div_ceil(hd_cuda::HASH_TILE) as u32,
            1,
        ),
        block_dim: (hd_cuda::HASH_TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = scratch.stream.launch_builder(&scratch.hd_fn);
    launch.arg(&*d_hashes);
    launch.arg(&num_hashes);
    launch.arg(&hv_d_i32);
    launch.arg(&mut *d_hv);

    let kernel_launch_start = Instant::now();
    scratch.hd_kernel_start.record(&scratch.stream)?;
    unsafe {
        launch.launch(cfg)?;
    }
    scratch.hd_kernel_end.record(&scratch.stream)?;
    metrics.cuda_hd_kernel_launch_ns = kernel_launch_start.elapsed().as_nanos();

    let d2h_start = Instant::now();
    scratch.hd_d2h_start.record(&scratch.stream)?;
    scratch
        .stream
        .memcpy_dtoh(&d_hv.slice(0..hv_d), &mut scratch.hv_host[..hv_d])?;
    scratch.hd_d2h_end.record(&scratch.stream)?;
    metrics.cuda_hd_d2h_ns = d2h_start.elapsed().as_nanos();
    metrics.cuda_hd_hash_h2d_event_ns = event_ms_to_ns(
        scratch
            .hd_hash_h2d_start
            .elapsed_ms(&scratch.hd_hash_h2d_end)?,
    );
    metrics.cuda_hd_hv_h2d_event_ns =
        event_ms_to_ns(scratch.hd_hv_h2d_start.elapsed_ms(&scratch.hd_hv_h2d_end)?);
    metrics.cuda_hd_kernel_event_ns =
        event_ms_to_ns(scratch.hd_kernel_start.elapsed_ms(&scratch.hd_kernel_end)?);
    metrics.cuda_hd_d2h_event_ns =
        event_ms_to_ns(scratch.hd_d2h_start.elapsed_ms(&scratch.hd_d2h_end)?);

    Ok(metrics)
}

#[cfg(feature = "cuda")]
#[inline]
fn event_ms_to_ns(ms: f32) -> u128 {
    (ms.max(0.0) as f64 * 1_000_000.0).round() as u128
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn indexed_result(index: usize, file: &str) -> IndexedSketchResult {
        IndexedSketchResult {
            index,
            sketch: FileSketch {
                ksize: 21,
                scaled: 1,
                seed: 123,
                canonical: true,
                hv_d: 64,
                hv_quant_bits: 16,
                hv_norm_2: 0,
                file_str: file.to_string(),
                hv: Vec::new(),
            },
            ull_record: Some(FileUllSketch {
                ksize: 21,
                canonical: true,
                seed: 123,
                ull_p: 14,
                file_str: file.to_string(),
                ull_state: vec![0; 1 << 14],
            }),
            metrics: FileSketchMetrics {
                file: file.to_string(),
                cuda_device_id: Some(index),
                ..FileSketchMetrics::default()
            },
        }
    }

    #[test]
    fn indexed_results_assemble_in_input_order() {
        let mut slots: Vec<Option<IndexedSketchResult>> = (0..3).map(|_| None).collect();

        store_indexed_sketch_result(&mut slots, indexed_result(2, "c.fna")).unwrap();
        store_indexed_sketch_result(&mut slots, indexed_result(0, "a.fna")).unwrap();
        store_indexed_sketch_result(&mut slots, indexed_result(1, "b.fna")).unwrap();

        let ordered = ordered_indexed_sketch_results(slots).unwrap();
        let files: Vec<&str> = ordered.iter().map(|r| r.sketch.file_str.as_str()).collect();
        assert_eq!(files, vec!["a.fna", "b.fna", "c.fna"]);
    }

    #[test]
    fn indexed_result_store_rejects_duplicates() {
        let mut slots: Vec<Option<IndexedSketchResult>> = (0..1).map(|_| None).collect();

        store_indexed_sketch_result(&mut slots, indexed_result(0, "a.fna")).unwrap();
        let err = store_indexed_sketch_result(&mut slots, indexed_result(0, "again.fna"))
            .expect_err("duplicate index should be rejected");

        assert!(err.to_string().contains("duplicate file index 0"));
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Result<Vec<HashSet<u64>>> {
    if ksize == 0 || ksize > 32 {
        return Err(anyhow!("CUDA ksize must be in 1..=32"));
    }
    if scaled == 0 {
        return Err(anyhow!("scaled must be greater than zero"));
    }
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));
    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0)?);
    let module = Arc::new(ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))?);

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| -> Result<HashSet<u64>> {
            let fna_seqs = fastx_reader::read_merge_seq(&files[i]).sequence;

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return Ok(HashSet::new());
            }

            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(bp_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs)?;
            let gpu_seq_nt4_table = stream.clone_htod(&SEQ_NT4_TABLE)?;

            let n_hash_per_thread = bp_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_bit_hash = stream.alloc_zeros::<u64>(n_hash_array)?;
            let mut gpu_kmer_count = stream.alloc_zeros::<u32>(n_threads)?;

            let f = module.load_function("cuda_kmer_bit_pack_mmhash")?;
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&bp_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&canonical);
            builder.arg(&gpu_seq_nt4_table);
            builder.arg(&n_threads);
            builder.arg(&mut gpu_kmer_bit_hash);
            builder.arg(&mut gpu_kmer_count);

            unsafe {
                builder.launch(LaunchConfig::for_num_elems(n_threads as u32))?;
            }

            let host_kmer_bit_hash = stream.clone_dtoh(&gpu_kmer_bit_hash)?;
            let mut host_kmer_count = vec![0u32; n_threads];
            stream.memcpy_dtoh(&gpu_kmer_count, &mut host_kmer_count)?;
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            let mut hashes = HashSet::new();
            for (thread, &count_u32) in host_kmer_count.iter().enumerate().take(n_threads) {
                let count = count_u32 as usize;
                if count > n_hash_per_thread {
                    return Err(anyhow!("CUDA k-mer worker produced too many hashes"));
                }
                let start = thread * n_hash_per_thread;
                hashes.extend(
                    host_kmer_bit_hash[start..start + count]
                        .iter()
                        .copied()
                        .filter(|&h| hash_passes_threshold(h, threshold)),
                );
            }
            Ok(hashes)
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_and_clear();
    Ok(sketch_kmer_sets)
}

#[cfg(feature = "cuda")]
pub fn cuda_t1ha2_hash_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
    seed: u64,
) -> Result<Vec<HashSet<u64>>> {
    if ksize == 0 || ksize > 32 {
        return Err(anyhow!("CUDA ksize must be in 1..=32"));
    }
    if scaled == 0 {
        return Err(anyhow!("scaled must be greater than zero"));
    }
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));

    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0)?);
    let module = Arc::new(ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))?);

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| -> Result<HashSet<u64>> {
            let fna_seqs = fastx_reader::read_merge_seq(&files[*i]).sequence;

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return Ok(HashSet::new());
            }

            let n_kmers = n_bps - ksize + 1;
            let kmer_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(kmer_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs)?;

            let n_hash_per_thread = kmer_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array)?;
            let mut gpu_kmer_count = stream.alloc_zeros::<u32>(n_threads)?;

            let f = module.load_function("cuda_kmer_t1ha2")?;
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&kmer_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&seed);
            builder.arg(&canonical);
            builder.arg(&n_threads);
            builder.arg(&mut gpu_kmer_hash);
            builder.arg(&mut gpu_kmer_count);

            unsafe {
                builder.launch(LaunchConfig::for_num_elems(n_threads as u32))?;
            }

            let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash)?;
            let mut host_kmer_count = vec![0u32; n_threads];
            stream.memcpy_dtoh(&gpu_kmer_count, &mut host_kmer_count)?;
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            let mut hashes = HashSet::new();
            for (thread, &count_u32) in host_kmer_count.iter().enumerate().take(n_threads) {
                let count = count_u32 as usize;
                if count > n_hash_per_thread {
                    return Err(anyhow!("CUDA k-mer worker produced too many hashes"));
                }
                let start = thread * n_hash_per_thread;
                hashes.extend(
                    host_kmer_hash[start..start + count]
                        .iter()
                        .copied()
                        .filter(|&h| hash_passes_threshold(h, threshold)),
                );
            }
            Ok(hashes)
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_and_clear();
    Ok(sketch_kmer_sets)
}
