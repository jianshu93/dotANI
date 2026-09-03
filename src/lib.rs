pub(crate) mod cardinality;
pub mod dist;
pub mod fastx_reader;
pub mod hd;
pub mod params;
pub mod sketch;
pub mod sketch_cuda;
pub mod types;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda_dot;
#[cfg(feature = "cuda")]
pub mod hd_cuda;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    use rand::{RngCore, SeedableRng};
    use wyhash::WyRng;

    use crate::types::{FileSketch, FileSketchMetrics};
    use crate::{dist, hd, utils};

    #[cfg(feature = "cuda")]
    const CUDA_KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

    const WY_P0: u64 = 0xa076_1d64_78bd_642f;
    const WY_P1: u64 = 0xe703_7ed1_a0b4_28db;

    fn sketch_for_hv(hv_d: usize) -> FileSketch {
        FileSketch {
            ksize: 16,
            scaled: 1,
            canonical: true,
            seed: 1447,
            hv_d,
            hv_quant_bits: 16,
            hv_norm_2: 0,
            file_str: String::from("test"),
            hv: Vec::new(),
        }
    }

    fn fixed_hashes() -> HashSet<u64> {
        [
            0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
        ]
        .into_iter()
        .collect()
    }

    fn test_wymum(a: u64, b: u64) -> u64 {
        let product = u128::from(a) * u128::from(b);
        ((product >> 64) ^ product) as u64
    }

    fn direct_seek_wyrng(hash: u64, chunk: usize) -> u64 {
        let chunk_offset = (chunk as u64).wrapping_add(1).wrapping_mul(WY_P0);
        let state = hash.wrapping_add(chunk_offset);
        test_wymum(state ^ WY_P1, state)
    }

    fn representative_wyrng_hashes() -> [u64; 6] {
        [
            0,
            1,
            0x1234_5678_9abc_def0,
            u64::MAX,
            0x8000_0000_0000_0000,
            0xfedc_ba98_7654_3210,
        ]
    }

    fn representative_wyrng_chunks() -> [usize; 7] {
        [0, 1, 63, 64, 127, (1024 / 64) - 1, (4096 / 64) - 1]
    }

    #[test]
    fn hd_encode_scalar_is_deterministic() {
        let sketch = sketch_for_hv(1024);
        let hashes = fixed_hashes();

        let first = hd::encode_hash_hd(&hashes, &sketch);
        let second = hd::encode_hash_hd(&hashes, &sketch);

        assert_eq!(first, second);
        assert_eq!(first.len(), 1024);
        assert_eq!(
            dist::compute_hv_l2_norm(&first),
            dist::compute_hv_l2_norm(&second)
        );
    }

    #[test]
    fn wyrng_direct_seek_matches_sequential_chunks() {
        let hashes = representative_wyrng_hashes();
        let chunks = representative_wyrng_chunks();

        for hash in hashes {
            let max_chunk = chunks.iter().copied().max().unwrap();
            let mut rng = WyRng::seed_from_u64(hash);
            let sequential: Vec<u64> = (0..=max_chunk).map(|_| rng.next_u64()).collect();

            for chunk in chunks {
                assert_eq!(
                    direct_seek_wyrng(hash, chunk),
                    sequential[chunk],
                    "direct seek mismatch for hash {hash:#018x}, chunk {chunk}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_wyrng_direct_seek_matches_rust_wyrng() {
        use cudarc::{
            driver::{CudaContext, LaunchConfig, PushKernelArg},
            nvrtc::Ptx,
        };

        let chunks = representative_wyrng_chunks();
        let mut host_hashes = Vec::new();
        let mut host_chunks = Vec::new();
        let mut expected = Vec::new();

        for hash in representative_wyrng_hashes() {
            let max_chunk = chunks.iter().copied().max().unwrap();
            let mut rng = WyRng::seed_from_u64(hash);
            let sequential: Vec<u64> = (0..=max_chunk).map(|_| rng.next_u64()).collect();

            for chunk in chunks {
                host_hashes.push(hash);
                host_chunks.push(chunk as i32);
                expected.push(sequential[chunk]);
            }
        }

        let ctx = CudaContext::new(0).unwrap();
        let module = ctx.load_module(Ptx::from_src(CUDA_KERNEL_PTX)).unwrap();
        let stream = ctx.default_stream();
        let gpu_hashes = stream.clone_htod(&host_hashes).unwrap();
        let gpu_chunks = stream.clone_htod(&host_chunks).unwrap();
        let mut gpu_out = stream.alloc_zeros::<u64>(expected.len()).unwrap();

        let f = module.load_function("cuda_test_wyrng_at_chunk").unwrap();
        let mut builder = stream.launch_builder(&f);
        builder.arg(&gpu_hashes);
        builder.arg(&gpu_chunks);
        builder.arg(&mut gpu_out);
        let n_outputs = expected.len() as i32;
        builder.arg(&n_outputs);

        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(expected.len() as u32))
                .unwrap();
        }

        let actual = stream.clone_dtoh(&gpu_out).unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_hash_threshold_preserves_both_u64_boundaries() {
        use cudarc::{
            driver::{CudaContext, LaunchConfig, PushKernelArg},
            nvrtc::Ptx,
        };

        let hashes = vec![0, u64::MAX - 1, u64::MAX, 0, u64::MAX - 1, u64::MAX];
        let thresholds = vec![
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX - 1,
        ];
        let expected = vec![1u8, 1, 1, 1, 0, 0];

        let ctx = CudaContext::new(0).unwrap();
        let module = ctx.load_module(Ptx::from_src(CUDA_KERNEL_PTX)).unwrap();
        let stream = ctx.default_stream();
        let gpu_hashes = stream.clone_htod(&hashes).unwrap();
        let gpu_thresholds = stream.clone_htod(&thresholds).unwrap();
        let mut gpu_out = stream.alloc_zeros::<u8>(expected.len()).unwrap();

        let f = module.load_function("cuda_test_hash_threshold").unwrap();
        let mut builder = stream.launch_builder(&f);
        builder.arg(&gpu_hashes);
        builder.arg(&gpu_thresholds);
        builder.arg(&mut gpu_out);
        let n = expected.len() as i32;
        builder.arg(&n);

        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(expected.len() as u32))
                .unwrap();
        }

        let actual = stream.clone_dtoh(&gpu_out).unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "cuda")]
    fn cuda_hd_test_module() -> (
        std::sync::Arc<cudarc::driver::CudaContext>,
        std::sync::Arc<cudarc::driver::CudaModule>,
    ) {
        use cudarc::{driver::CudaContext, nvrtc::Ptx};

        let ctx = CudaContext::new(0).unwrap();
        let module = ctx.load_module(Ptx::from_src(CUDA_KERNEL_PTX)).unwrap();
        (ctx, module)
    }

    #[cfg(feature = "cuda")]
    fn assert_cuda_hd_matches_cpu(input_hashes: &[u64], hv_d: usize) {
        use crate::hd_cuda;

        let sketch = sketch_for_hv(hv_d);
        let hash_set: HashSet<u64> = input_hashes.iter().copied().collect();
        assert_eq!(
            hash_set.len(),
            input_hashes.len(),
            "CUDA HD parity helper expects unique input hashes"
        );

        let cpu_hv = hd::encode_hash_hd(&hash_set, &sketch);
        let (ctx, module) = cuda_hd_test_module();
        let (gpu_hv, _metrics) =
            hd_cuda::encode_hash_hd_cuda(input_hashes, hv_d, &ctx, &module).unwrap();

        assert_eq!(cpu_hv, gpu_hv);
        assert_eq!(
            dist::compute_hv_l2_norm(&cpu_hv),
            dist::compute_hv_l2_norm(&gpu_hv)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_hd_encode_matches_cpu_edge_cases() {
        assert_cuda_hd_matches_cpu(&[], 1024);
        assert_cuda_hd_matches_cpu(&[0x1234_5678_9abc_def0], 1024);
        assert_cuda_hd_matches_cpu(&[0], 1024);
        assert_cuda_hd_matches_cpu(&[0], 4096);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_hd_encode_matches_cpu_representative_hash_sets() {
        let fixed: Vec<u64> = fixed_hashes().into_iter().collect();
        assert_cuda_hd_matches_cpu(&fixed, 1024);
        assert_cuda_hd_matches_cpu(&fixed, 4096);

        let mut rng = WyRng::seed_from_u64(0x5eed_cafe_dead_beef);
        let mut larger = HashSet::with_capacity(10_000);
        while larger.len() < 10_000 {
            larger.insert(rng.next_u64());
        }
        let larger: Vec<u64> = larger.into_iter().collect();
        assert_cuda_hd_matches_cpu(&larger, 1024);
        assert_cuda_hd_matches_cpu(&larger, 4096);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_hd_metrics_follow_gpu_work_applicability() {
        use crate::hd_cuda;

        let (ctx, module) = cuda_hd_test_module();

        let (_empty_hv, empty_metrics) =
            hd_cuda::encode_hash_hd_cuda(&[], 4096, &ctx, &module).unwrap();
        assert_eq!(empty_metrics, hd_cuda::GpuHdEncodeMetrics::default());

        assert!(hd_cuda::encode_hash_hd_cuda(&[0x1234_5678_9abc_def0], 63, &ctx, &module).is_err());

        let (_normal_hv, normal_metrics) =
            hd_cuda::encode_hash_hd_cuda(&[0x1234_5678_9abc_def0], 4096, &ctx, &module).unwrap();
        assert!(normal_metrics.cuda_hd_alloc_ns > 0);
        assert!(normal_metrics.cuda_hd_hash_h2d_ns > 0);
        assert!(normal_metrics.cuda_hd_hv_h2d_ns > 0);
        assert!(normal_metrics.cuda_hd_kernel_launch_ns > 0);
        assert!(normal_metrics.cuda_hd_d2h_ns > 0);
    }

    #[test]
    fn hd_encode_cpu_edge_cases_are_explicit() {
        let empty_hashes = HashSet::new();
        let empty_hv = hd::encode_hash_hd(&empty_hashes, &sketch_for_hv(256));
        assert_eq!(empty_hv, vec![0; 256]);

        let one_hash = HashSet::from([0x1234_5678_9abc_def0]);
        let one_hash_hv = hd::encode_hash_hd(&one_hash, &sketch_for_hv(256));
        assert_eq!(one_hash_hv.len(), 256);
        assert!(
            one_hash_hv
                .iter()
                .all(|&coordinate| coordinate == -1 || coordinate == 1)
        );

        let zero_hash = HashSet::from([0]);
        let zero_hash_hv = hd::encode_hash_hd(&zero_hash, &sketch_for_hv(256));
        let zero_hash_bits = direct_seek_wyrng(0, 0);
        let expected_zero_hash_hv: Vec<i32> = (0..256)
            .map(|bit| {
                let rnd = if bit < 64 {
                    zero_hash_bits
                } else {
                    direct_seek_wyrng(0, bit / 64)
                };
                -1 + (((rnd >> (bit % 64)) & 1) << 1) as i32
            })
            .collect();
        assert_eq!(zero_hash_hv, expected_zero_hash_hv);

        let multi_hashes =
            HashSet::from([0, 1, 0x1234_5678_9abc_def0, 0x8000_0000_0000_0000, u64::MAX]);
        let multi_hash_hv = hd::encode_hash_hd(&multi_hashes, &sketch_for_hv(256));
        assert_eq!(multi_hash_hv.len(), 256);

        for invalid in [0, 70, 257] {
            assert!(hd::validate_hv_dimension(invalid).is_err());
        }
    }

    #[test]
    fn hd_encode_simd_matches_scalar_when_available() {
        let hashes = fixed_hashes();

        for hv_d in [256, 1024] {
            let sketch = sketch_for_hv(hv_d);
            let scalar = hd::encode_hash_hd(&hashes, &sketch);

            if is_x86_feature_detected!("avx2") {
                let avx2 = unsafe { hd::encode_hash_hd_avx2(&hashes, &sketch) };
                assert_eq!(scalar, avx2);
            }

            if is_x86_feature_detected!("avx512f") {
                let avx512 = unsafe { hd::encode_hash_hd_avx512(&hashes, &sketch) };
                assert_eq!(scalar, avx512);
            }
        }
    }

    #[test]
    fn hd_compress_decompress_round_trips() {
        let mut sketch = sketch_for_hv(1024);
        let hv: Vec<i32> = (0..sketch.hv_d)
            .map(|i| ((i as i32 % 257) - 128) / 3)
            .collect();

        sketch.hv_quant_bits = hd::compress_hd_sketch(&mut sketch, &hv).unwrap();
        let decoded = hd::decompress_hd_sketch(&mut sketch).unwrap();

        assert_eq!(decoded, hv);
    }

    #[test]
    fn fasta_file_discovery_supports_expected_suffixes() {
        let dir = unique_test_dir("dotani_file_discovery");
        fs::create_dir_all(&dir).unwrap();

        for suffix in [
            "fna",
            "fa",
            "fasta",
            "fna.gz",
            "fa.gz",
            "fasta.gz",
            "fna.bz2",
            "fa.bz2",
            "fasta.bz2",
            "fna.xz",
            "fa.xz",
            "fasta.xz",
            "fna.zst",
            "fa.zst",
            "fasta.zst",
        ] {
            fs::write(dir.join(format!("sample.{suffix}")), b">s\nACGT\n").unwrap();
        }
        fs::write(dir.join("ignored.txt"), b">s\nACGT\n").unwrap();

        let files = utils::get_fasta_files(&dir).unwrap();
        let mut names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();

        assert_eq!(names.len(), 15);
        assert!(names.contains(&String::from("sample.fna")));
        assert!(names.contains(&String::from("sample.fasta.zst")));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sketch_metrics_tsv_preserves_cuda_hd_na_contract() {
        let dir = unique_test_dir("dotani_metrics_schema");
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("metrics");

        let cpu_metric = FileSketchMetrics {
            file: String::from("cpu.fna"),
            input_bases: 32,
            hashes_seen: 28,
            unique_hashes: 2,
            ..FileSketchMetrics::default()
        };
        let cuda_metric = FileSketchMetrics {
            file: String::from("cuda.fna"),
            input_bases: 32,
            hashes_seen: 28,
            unique_hashes: 2,
            cuda_stream_lane: Some(0),
            cuda_device_id: Some(2),
            cuda_h2d_ns: Some(11),
            cuda_h2d_event_ns: Some(12),
            cuda_alloc_ns: Some(22),
            cuda_launch_ns: Some(33),
            cuda_kernel_event_ns: Some(34),
            cuda_d2h_ns: Some(44),
            cuda_d2h_event_ns: Some(45),
            cuda_compact_ns: Some(55),
            cuda_filter_ns: Some(66),
            ..FileSketchMetrics::default()
        };

        utils::dump_sketch_metrics(&[cpu_metric, cuda_metric], &prefix, 1234).unwrap();

        let files_path = dir.join("metrics.files.tsv");
        let files_tsv = fs::read_to_string(&files_path).unwrap();
        let rows: Vec<Vec<&str>> = files_tsv
            .lines()
            .map(|line| line.split('\t').collect())
            .collect();

        assert_eq!(rows.len(), 3);
        let header = &rows[0];
        assert_eq!(header.len(), 32);
        assert_eq!(
            &header[23..],
            &[
                "cuda_hd_hash_h2d_ns",
                "cuda_hd_hash_h2d_event_ns",
                "cuda_hd_hv_h2d_ns",
                "cuda_hd_hv_h2d_event_ns",
                "cuda_hd_alloc_ns",
                "cuda_hd_kernel_launch_ns",
                "cuda_hd_kernel_event_ns",
                "cuda_hd_d2h_ns",
                "cuda_hd_d2h_event_ns"
            ]
        );

        let cpu_row = &rows[1];
        let cuda_row = &rows[2];
        assert_eq!(cpu_row.len(), header.len());
        assert_eq!(cuda_row.len(), header.len());
        assert_eq!(cpu_row[0], "cpu.fna");
        assert_eq!(cpu_row[1], "32");
        assert_eq!(&cpu_row[11..], &["NA"; 21]);
        assert_eq!(cuda_row[0], "cuda.fna");
        assert_eq!(cuda_row[1], "32");
        assert_eq!(&cuda_row[11..14], &["NA", "0", "2"]);
        assert_eq!(
            &cuda_row[14..23],
            &["11", "12", "22", "33", "34", "44", "45", "55", "66"]
        );
        assert_eq!(&cuda_row[23..], &["NA"; 9]);

        let summary_tsv = fs::read_to_string(dir.join("metrics.summary.tsv")).unwrap();
        let summary_rows: Vec<Vec<&str>> = summary_tsv
            .lines()
            .map(|line| line.split('\t').collect())
            .collect();
        assert_eq!(summary_rows.len(), 2);
        assert_eq!(summary_rows[0].len(), 32);
        assert_eq!(summary_rows[1].len(), 32);
        assert_eq!(summary_rows[1][0], "TOTAL");
        assert_eq!(summary_rows[1][1], "64");
        assert_eq!(summary_rows[1][11], "1234");
        assert_eq!(&summary_rows[1][12..14], &["NA", "NA"]);
        assert_eq!(
            &summary_rows[1][14..23],
            &["11", "12", "22", "33", "34", "44", "45", "55", "66"]
        );
        assert_eq!(&summary_rows[1][23..], &["NA"; 9]);

        fs::remove_dir_all(&dir).unwrap();
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "{name}_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        dir
    }
}
