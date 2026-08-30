use crate::types::*;

#[cfg(feature = "cuda")]
use {
    crate::{dist, fastx_reader, hd, hd_cuda, utils},
    cudarc::{
        driver::{CudaContext, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
    },
    glob::glob,
    log::info,
    rayon::prelude::*,
    std::collections::HashSet,
    std::path::{Path, PathBuf},
    std::sync::Arc,
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

#[allow(unused_variables)]
#[cfg(not(feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) {
    use log::error;

    error!(
        "Cuda sketching is not supported. Please add `--features cuda-sketch` for installation to enable it."
    );
}

#[cfg(all(target_arch = "x86_64", feature = "cuda"))]
pub fn sketch_cuda(params: SketchParams) {
    let sketch_wall_start = Instant::now();
    let inputs = utils::get_sketch_inputs(&params).expect("failed to resolve sketch inputs");
    let n_file = inputs.len();

    info!("Start GPU sketching...");
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

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

            let (full_hashes, mut metrics) =
                extract_kmer_t1ha2_cuda_full_hashes(&input.read_path, &sketch, &ctx, &module);
            metrics.file = sketch.file_str.clone();
            metrics.hashes_seen = full_hashes.len();

            let threshold = u64::MAX / sketch.scaled;
            let mut sampled_hash_set = HashSet::<u64>::new();

            let hash_and_dedup_start = Instant::now();
            let ull_record = if params.if_ull {
                let mut ull =
                    UltraLogLog::new(params.ull_p).expect("Invalid UltraLogLog precision");

                for &h in &full_hashes {
                    ull.add(h);
                    sampled_hash_set.insert(h);
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
                for &h in &full_hashes {
                    if h < threshold {
                        sampled_hash_set.insert(h);
                    }
                }
                None
            };
            let hash_and_dedup_ns = hash_and_dedup_start.elapsed().as_nanos();
            metrics.hash_and_dedup_ns = hash_and_dedup_ns;
            metrics.cuda_filter_ns = Some(hash_and_dedup_ns);
            metrics.unique_hashes = sampled_hash_set.len();

            let sampled_hashes: Vec<u64> = sampled_hash_set.iter().copied().collect();
            let start = Instant::now();
            let (hv, hd_metrics) =
                hd_cuda::encode_hash_hd_cuda(&sampled_hashes, sketch.hv_d, &ctx, &module).unwrap();
            metrics.hd_encode_ns = start.elapsed().as_nanos();
            if !sampled_hashes.is_empty() && sketch.hv_d >= 64 {
                metrics.cuda_hd_hash_h2d_ns = Some(hd_metrics.cuda_hd_hash_h2d_ns);
                metrics.cuda_hd_hv_h2d_ns = Some(hd_metrics.cuda_hd_hv_h2d_ns);
                metrics.cuda_hd_alloc_ns = Some(hd_metrics.cuda_hd_alloc_ns);
                metrics.cuda_hd_kernel_launch_ns = Some(hd_metrics.cuda_hd_kernel_launch_ns);
                metrics.cuda_hd_d2h_ns = Some(hd_metrics.cuda_hd_d2h_ns);
            }

            let start = Instant::now();
            sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);
            metrics.hv_norm_ns = start.elapsed().as_nanos();

            let start = Instant::now();
            if params.if_compressed {
                sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, &hv) };
            } else {
                sketch.hv = hv.clone();
            }
            metrics.hd_compress_ns = start.elapsed().as_nanos();

            pb.inc(1);
            pb.eta();

            metrics.total_worker_ns = worker_start.elapsed().as_nanos();

            (sketch, ull_record, metrics)
        })
        .collect();

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

    utils::dump_sketch(&all_filesketch, &params.out_file);

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
    }

    if let Some(prefix) = &params.metrics_out {
        utils::dump_sketch_metrics(&all_metrics, prefix, sketch_wall_start.elapsed().as_nanos());
    }
}

#[cfg(feature = "cuda")]
fn extract_kmer_t1ha2_cuda_full_hashes(
    read_path: &PathBuf,
    sketch: &FileSketch,
    ctx: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
) -> (Vec<u64>, FileSketchMetrics) {
    let fasta_start = Instant::now();
    let fna_seqs = fastx_reader::read_merge_seq(read_path);
    let fasta_ns = fasta_start.elapsed().as_nanos();

    let n_bps = fna_seqs.len();
    let mut metrics = FileSketchMetrics {
        input_bases: n_bps,
        fasta_ns,
        // Placeholder until Sprint 1A introduces explicit stream lanes.
        cuda_stream_lane: Some(0),
        ..FileSketchMetrics::default()
    };
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let seed = sketch.seed;

    if n_bps < ksize {
        return (Vec::new(), metrics);
    }

    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512usize;
    let n_threads = n_kmers.div_ceil(kmer_per_thread);

    let stream = ctx.default_stream();
    let n_hash_per_thread = kmer_per_thread;
    let n_hash_array = n_hash_per_thread * n_threads;
    let alloc_start = Instant::now();
    let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();
    metrics.cuda_alloc_ns = Some(alloc_start.elapsed().as_nanos());

    let h2d_start = Instant::now();
    let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();
    metrics.cuda_h2d_ns = Some(h2d_start.elapsed().as_nanos());

    let f = module.load_function("cuda_kmer_t1ha2").unwrap();
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
    builder.arg(&mut gpu_kmer_hash);

    let launch_start = Instant::now();
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n_threads as u32))
            .unwrap();
    }
    metrics.cuda_launch_ns = Some(launch_start.elapsed().as_nanos());

    let d2h_start = Instant::now();
    let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();
    metrics.cuda_d2h_ns = Some(d2h_start.elapsed().as_nanos());

    let filter_start = Instant::now();
    let hashes: Vec<u64> = host_kmer_hash.into_iter().filter(|&h| h != 0).collect();
    metrics.cuda_zero_filter_ns = Some(filter_start.elapsed().as_nanos());

    (hashes, metrics)
}

#[cfg(feature = "cuda")]
pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Vec<HashSet<u64>> {
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));
    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| {
            let fna_seqs = fastx_reader::read_merge_seq(&files[i]);

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(bp_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();
            let gpu_seq_nt4_table = stream.clone_htod(&SEQ_NT4_TABLE).unwrap();

            let n_hash_per_thread = bp_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_bit_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_bit_pack_mmhash").unwrap();
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
            builder.arg(&mut gpu_kmer_bit_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_bit_hash = stream.clone_dtoh(&gpu_kmer_bit_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_bit_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}

#[cfg(feature = "cuda")]
pub fn cuda_t1ha2_hash_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
    seed: u64,
) -> Vec<HashSet<u64>> {
    let files: Vec<_> = glob(Path::new(path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| {
            let fna_seqs = fastx_reader::read_merge_seq(files[*i].as_ref().unwrap());

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let kmer_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(kmer_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();

            let n_hash_per_thread = kmer_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_t1ha2").unwrap();
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
            builder.arg(&mut gpu_kmer_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}
