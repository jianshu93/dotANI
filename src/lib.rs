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
    use std::fs;
    use std::path::PathBuf;

    use crate::utils;

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
