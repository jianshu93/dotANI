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

    use crate::types::FileSketch;
    use crate::{dist, hd, utils};

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
            0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
        ]
        .into_iter()
        .collect()
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
