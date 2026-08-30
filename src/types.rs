use std::arch::x86_64::*;
use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use t1ha;

#[inline]
pub fn mm_hash(bytes: &[u8]) -> usize {
    let mut key = usize::from_ne_bytes(bytes.try_into().unwrap());
    key = !key.wrapping_add(key << 21);
    key = key ^ key >> 24;
    key = (key.wrapping_add(key << 3)).wrapping_add(key << 8);
    key = key ^ key >> 14;
    key = (key.wrapping_add(key << 2)).wrapping_add(key << 4);
    key = key ^ key >> 28;
    key = key.wrapping_add(key << 31);
    key
}

#[inline]
pub fn mm_hash64(kmer: u64) -> u64 {
    let mut key = kmer;
    key = !key + (key << 21);
    key = key ^ key >> 24;
    key = (key + (key << 3)) + (key << 8);
    key = key ^ key >> 14;
    key = (key + (key << 2)) + (key << 4);
    key = key ^ key >> 28;
    key = key + (key << 31);
    key
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn mm_hash64_avx2(kmer: __m256i) -> __m256i {
    let mut key = kmer;
    let s1 = _mm256_slli_epi64(key, 21);
    key = _mm256_add_epi64(key, s1);
    key = _mm256_xor_si256(key, _mm256_cmpeq_epi64(key, key));

    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 24));
    let s2 = _mm256_slli_epi64(key, 3);
    let s3 = _mm256_slli_epi64(key, 8);

    key = _mm256_add_epi64(key, s2);
    key = _mm256_add_epi64(key, s3);
    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 14));
    let s4 = _mm256_slli_epi64(key, 2);
    let s5 = _mm256_slli_epi64(key, 4);
    key = _mm256_add_epi64(key, s4);
    key = _mm256_add_epi64(key, s5);
    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 28));

    let s6 = _mm256_slli_epi64(key, 31);
    key = _mm256_add_epi64(key, s6);

    key
}

pub struct CliParams {
    pub mode: String,
    pub path: PathBuf,
    pub manifest: Option<PathBuf>,
    pub path_ref_sketch: PathBuf,
    pub path_query_sketch: PathBuf,
    pub out_file: PathBuf,

    pub ksize: u8,
    pub seed: u64,
    pub sketch_method: String,
    pub canonical: bool,
    pub device: String,
    pub scaled: u64,
    pub hv_d: usize,
    pub hv_quant_scale: f32,
    pub ani_threshold: f32,
    pub if_compressed: bool,

    pub threads: usize,
    pub cuda_dedup_strategy: CudaDedupStrategy,
    pub max_readers: Option<usize>,

    pub if_ull: bool,
    pub ull_p: u32,
    pub ull_out_file: PathBuf,
    pub path_ref_ull: PathBuf,
    pub path_query_ull: PathBuf,

    pub metrics_out: Option<PathBuf>,
}

pub struct SketchParams {
    pub path: PathBuf,
    pub manifest: Option<PathBuf>,
    pub out_file: PathBuf,
    pub sketch_method: String,
    pub canonical: bool,
    pub device: String,
    pub ksize: u8,
    pub seed: u64,
    pub scaled: u64,
    pub hv_d: usize,
    pub hv_quant_scale: f32,
    pub if_compressed: bool,
    pub threads: usize,
    pub cuda_dedup_strategy: CudaDedupStrategy,
    pub max_readers: Option<usize>,

    pub if_ull: bool,
    pub ull_p: u32,
    pub ull_out_file: PathBuf,
    pub metrics_out: Option<PathBuf>,
}

impl Default for SketchParams {
    fn default() -> Self {
        SketchParams {
            path: PathBuf::new(),
            manifest: None,
            out_file: PathBuf::new(),
            sketch_method: String::from("t1ha2"),
            canonical: true,
            device: String::from("cpu"),
            ksize: 21,
            seed: 123,
            scaled: 1500,
            hv_d: 4096,
            hv_quant_scale: 1.0,
            if_compressed: true,
            threads: 1,
            cuda_dedup_strategy: CudaDedupStrategy::SortUnstable,
            max_readers: None,

            if_ull: false,
            ull_p: 14,
            ull_out_file: PathBuf::new(),
            metrics_out: None,
        }
    }
}

impl SketchParams {
    pub fn new(params: &CliParams) -> SketchParams {
        let mut new_sketch = SketchParams::default();
        new_sketch.path = params.path.clone();
        new_sketch.manifest = params.manifest.clone();
        new_sketch.out_file = params.out_file.clone();
        new_sketch.sketch_method = params.sketch_method.clone();
        new_sketch.canonical = params.canonical;
        new_sketch.device = params.device.clone();
        new_sketch.ksize = params.ksize;
        new_sketch.seed = params.seed;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.hv_quant_scale = params.hv_quant_scale;
        new_sketch.if_compressed = params.if_compressed;
        new_sketch.threads = params.threads;
        new_sketch.cuda_dedup_strategy = params.cuda_dedup_strategy;
        new_sketch.max_readers = params.max_readers;

        new_sketch.if_ull = params.if_ull;
        new_sketch.ull_p = params.ull_p;
        new_sketch.ull_out_file = params.ull_out_file.clone();
        new_sketch.metrics_out = params.metrics_out.clone();

        new_sketch
    }
}

#[derive(Clone, Debug, Default)]
pub struct FileSketchMetrics {
    pub file: String,
    pub input_bases: usize,
    pub hashes_seen: usize,
    pub unique_hashes: usize,
    pub fasta_ns: u128,
    pub fasta_wait_ns: u128,
    pub hash_and_dedup_ns: u128,
    pub hd_encode_ns: u128,
    pub hv_norm_ns: u128,
    pub hd_compress_ns: u128,
    pub total_worker_ns: u128,
    pub sketch_wall_ns: Option<u128>,
    pub cuda_stream_lane: Option<usize>,
    pub cuda_device_id: Option<usize>,
    pub cuda_h2d_ns: Option<u128>,
    pub cuda_h2d_event_ns: Option<u128>,
    pub cuda_alloc_ns: Option<u128>,
    pub cuda_launch_ns: Option<u128>,
    pub cuda_kernel_event_ns: Option<u128>,
    pub cuda_d2h_ns: Option<u128>,
    pub cuda_d2h_event_ns: Option<u128>,
    pub cuda_compact_ns: Option<u128>,
    pub cuda_filter_ns: Option<u128>,
    pub cuda_hd_hash_h2d_ns: Option<u128>,
    pub cuda_hd_hash_h2d_event_ns: Option<u128>,
    pub cuda_hd_hv_h2d_ns: Option<u128>,
    pub cuda_hd_hv_h2d_event_ns: Option<u128>,
    pub cuda_hd_alloc_ns: Option<u128>,
    pub cuda_hd_kernel_launch_ns: Option<u128>,
    pub cuda_hd_kernel_event_ns: Option<u128>,
    pub cuda_hd_d2h_ns: Option<u128>,
    pub cuda_hd_d2h_event_ns: Option<u128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaDedupStrategy {
    HashSet,
    SortUnstable,
}

impl CudaDedupStrategy {
    pub fn from_cli_value(value: &str) -> Self {
        match value {
            "hashset" => Self::HashSet,
            "sort_unstable" => Self::SortUnstable,
            _ => panic!("invalid CUDA dedup strategy {value:?}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HashSet => "hashset",
            Self::SortUnstable => "sort_unstable",
        }
    }
}

pub(crate) fn hash_passes_threshold(hash: u64, threshold: u64) -> bool {
    threshold == u64::MAX || hash < threshold
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SketchInput {
    pub read_path: PathBuf,
    pub file_id: String,
}

pub struct Sketch {
    pub file_name: String,
    pub sketch_method: String,
    pub canonical: bool,
    pub ksize: u8,
    pub seed: u64,
    pub scaled: u64,
    pub threshold: u64,
    pub hash_set: HashSet<u64>,
    pub hv_quant_scale: f32,
    pub hv_quant_bits: u8,
    pub hv_d: usize,
    pub hv: Vec<i32>,
    pub hv_l2_norm_sq: i64,
}

impl Default for Sketch {
    fn default() -> Self {
        Sketch {
            file_name: String::from(""),
            sketch_method: String::from("xxh3"),
            canonical: true,
            ksize: 21,
            seed: 123,
            scaled: 2000,
            threshold: u64::MAX / 2000,
            hash_set: HashSet::default(),
            hv_quant_scale: 1.0,
            hv_quant_bits: 0,
            hv_d: 4096,
            hv: vec![],
            hv_l2_norm_sq: 0,
        }
    }
}

impl Sketch {
    pub fn new(file: String, params: &SketchParams) -> Sketch {
        let mut new_sketch = Sketch::default();
        new_sketch.file_name = file;
        new_sketch.sketch_method = params.sketch_method.clone();
        new_sketch.canonical = params.canonical;
        new_sketch.ksize = params.ksize;
        new_sketch.seed = params.seed;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.hv_quant_scale = params.hv_quant_scale;
        new_sketch.threshold = u64::MAX / params.scaled;
        new_sketch
    }

    pub fn insert_kmer(&mut self, kmer: &[u8]) {
        let h = match self.sketch_method.as_str() {
            "t1ha2" => t1ha::t1ha2_atonce(kmer, self.seed),
            "mmhash" => mm_hash(kmer) as u64,
            _ => t1ha::t1ha2_atonce(kmer, self.seed),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
        }
    }

    pub fn insert_kmer_u64(&mut self, kmer: u64) {
        let h = match self.sketch_method.as_str() {
            "t1ha2_64" => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
            "mmhash64" => mm_hash64(kmer),
            _ => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
        }
    }

    pub unsafe fn insert_kmer_u64_avx2(&mut self, kmer: __m256i) {
        let hash_256 = mm_hash64_avx2(kmer);

        let h1 = _mm256_extract_epi64(hash_256, 0) as u64;
        let h2 = _mm256_extract_epi64(hash_256, 1) as u64;
        let h3 = _mm256_extract_epi64(hash_256, 2) as u64;
        let h4 = _mm256_extract_epi64(hash_256, 3) as u64;

        for h in [h1, h2, h3, h4] {
            if h > 0 && h < self.threshold {
                self.hash_set.insert(h);
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct FileSketch {
    pub ksize: u8,
    pub scaled: u64,
    pub canonical: bool,
    pub seed: u64,
    pub hv_d: usize,
    pub hv_quant_bits: u8,
    pub hv_norm_2: i64,
    pub file_str: String,
    pub hv: Vec<i32>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct FileUllSketch {
    pub ksize: u8,
    pub canonical: bool,
    pub seed: u64,
    pub ull_p: u32,
    pub file_str: String,
    pub ull_state: Vec<u8>,
}

pub struct SketchDist {
    pub path_ref_sketch: PathBuf,
    pub path_query_sketch: PathBuf,
    pub path_ref_ull: PathBuf,
    pub path_query_ull: PathBuf,
    pub out_file: PathBuf,
    pub ksize: u8,
    pub hv_d: usize,
    pub ani_threshold: f32,
    pub threads: usize,
    pub file_ani: Vec<((String, String), f32)>,
}

impl Default for SketchDist {
    fn default() -> Self {
        SketchDist {
            path_ref_sketch: PathBuf::new(),
            path_query_sketch: PathBuf::new(),
            path_ref_ull: PathBuf::new(),
            path_query_ull: PathBuf::new(),
            out_file: PathBuf::new(),
            ksize: 21,
            hv_d: 1024,
            ani_threshold: 85.0,
            threads: 1,
            file_ani: Vec::<((String, String), f32)>::new(),
        }
    }
}

impl SketchDist {
    pub fn new(params: &CliParams) -> SketchDist {
        let mut new_dist = SketchDist::default();
        new_dist.path_ref_sketch = params.path_ref_sketch.clone();
        new_dist.path_query_sketch = params.path_query_sketch.clone();
        new_dist.path_ref_ull = params.path_ref_ull.clone();
        new_dist.path_query_ull = params.path_query_ull.clone();
        new_dist.out_file = params.out_file.clone();
        new_dist.ksize = params.ksize;
        new_dist.hv_d = params.hv_d;
        new_dist.ani_threshold = params.ani_threshold;
        new_dist.threads = params.threads;
        new_dist
    }
}
