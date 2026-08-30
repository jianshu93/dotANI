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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    use rand::{RngCore, SeedableRng};
    use wyhash::WyRng;

    use crate::types::FileSketch;
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
    fn hd_encode_simd_matches_scalar_when_available() {
        let sketch = sketch_for_hv(1024);
        let hashes = fixed_hashes();
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

    #[test]
    fn hd_compress_decompress_round_trips() {
        let mut sketch = sketch_for_hv(1024);
        let hv: Vec<i32> = (0..sketch.hv_d)
            .map(|i| ((i as i32 % 257) - 128) / 3)
            .collect();

        sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, &hv) };
        let decoded = unsafe { hd::decompress_hd_sketch(&mut sketch) };

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

        let files = utils::get_fasta_files(&dir);
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
