use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(label: &str) -> Self {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dotani_cli_{}_{}_{}",
            std::process::id(),
            label,
            id
        ));
        fs::create_dir_all(&path).expect("failed to create CLI test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(env!("CARGO_BIN_EXE_dotani"))
        .args(args)
        .output()
        .expect("failed to execute dotani")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "expected success, status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_failure_contains(output: &Output, needle: &str) {
    assert!(!output.status.success(), "unexpected success");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains(needle),
        "expected {needle:?} in command output:\n{text}"
    );
}

fn write_valid_fasta(dir: &Path) -> PathBuf {
    let path = dir.join("valid.fna");
    fs::write(
        &path,
        b">record-1\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n",
    )
    .expect("failed to write valid FASTA");
    path
}

#[test]
fn manifest_cli_preserves_row_order_and_file_ids() {
    let dir = TestDir::new("manifest-e2e");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    let alpha = inputs.join("alpha.fna");
    let zeta = inputs.join("zeta.fna");
    fs::write(
        &alpha,
        b">record-alpha\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n",
    )
    .unwrap();
    fs::write(
        &zeta,
        b">record-zeta\nTTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGG\n",
    )
    .unwrap();

    // Manifest row order (zeta first) deliberately differs from lexical path
    // order (alpha first), and file_id values are not filesystem paths.
    let manifest = dir.path().join("manifest.tsv");
    fs::write(
        &manifest,
        format!(
            "read_path\tfile_id\n{}\tgenome-zeta\n{}\tgenome-alpha\n",
            zeta.display(),
            alpha.display()
        ),
    )
    .unwrap();

    let sketch_path = dir.path().join("manifest.sketch");
    let sketch = run([
        "sketch",
        "--manifest",
        manifest.to_str().unwrap(),
        "--out",
        sketch_path.to_str().unwrap(),
        "--device",
        "cpu",
        "--ksize",
        "3",
        "--hv-d",
        "256",
    ]);
    assert_success(&sketch);

    let expected_ids = ["genome-zeta", "genome-alpha"];
    let hd_sketches = dotani::utils::load_sketch(&sketch_path);
    let hd_ids: Vec<&str> = hd_sketches
        .iter()
        .map(|sketch| sketch.file_str.as_str())
        .collect();
    assert_eq!(hd_ids, expected_ids);

    let ull_sidecar = PathBuf::from(format!("{}.ull", sketch_path.display()));
    let ull_sketches = dotani::utils::load_ull_sketch(&ull_sidecar);
    let ull_ids: Vec<&str> = ull_sketches
        .iter()
        .map(|sketch| sketch.file_str.as_str())
        .collect();
    assert_eq!(ull_ids, expected_ids);

    // Path-based input mode stays lexically ordered with path-as-id behavior.
    let path_sketch_path = dir.path().join("path.sketch");
    let path_sketch = run([
        "sketch",
        "--path",
        inputs.to_str().unwrap(),
        "--out",
        path_sketch_path.to_str().unwrap(),
        "--device",
        "cpu",
        "--ksize",
        "3",
        "--hv-d",
        "256",
    ]);
    assert_success(&path_sketch);

    let path_hd = dotani::utils::load_sketch(&path_sketch_path);
    let path_ids: Vec<String> = path_hd
        .iter()
        .map(|sketch| sketch.file_str.clone())
        .collect();
    let lexical_ids = [alpha.display().to_string(), zeta.display().to_string()];
    assert_eq!(path_ids, lexical_ids);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_only_binary_rejects_cuda_and_cpu_max_readers() {
    let dir = TestDir::new("device");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    write_valid_fasta(&inputs);

    let cuda = run([
        "sketch",
        "--path",
        inputs.to_str().unwrap(),
        "--out",
        dir.path().join("cuda.sketch").to_str().unwrap(),
        "--device",
        "cuda",
    ]);
    assert_failure_contains(&cuda, "requires a binary built with --features cuda");

    let max_readers = run([
        "sketch",
        "--path",
        inputs.to_str().unwrap(),
        "--out",
        dir.path().join("readers.sketch").to_str().unwrap(),
        "--device",
        "cpu",
        "--max-readers",
        "1",
    ]);
    assert_failure_contains(&max_readers, "only supported with --device cuda");
}

#[cfg(feature = "cuda")]
fn metric_counts(path: &Path) -> Vec<(String, usize, usize, usize)> {
    let metrics_path = PathBuf::from(format!("{}.files.tsv", path.display()));
    let text = fs::read_to_string(&metrics_path).expect("failed to read sketch metrics");
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .expect("metrics header is missing")
        .split('\t')
        .collect();
    assert_eq!(header.len(), 32);
    assert!(header.contains(&"cuda_compact_ns"));
    assert!(!header.contains(&"cuda_memset_ns"));
    let file_idx = header.iter().position(|&name| name == "file").unwrap();
    let input_idx = header
        .iter()
        .position(|&name| name == "input_bases")
        .unwrap();
    let seen_idx = header
        .iter()
        .position(|&name| name == "hashes_seen")
        .unwrap();
    let unique_idx = header
        .iter()
        .position(|&name| name == "unique_hashes")
        .unwrap();

    lines
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            (
                fields[file_idx].to_string(),
                fields[input_idx].parse().unwrap(),
                fields[seen_idx].parse().unwrap(),
                fields[unique_idx].parse().unwrap(),
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn cpu_and_cuda_sketches_match_for_concurrent_files_and_dedup_strategies() {
    let dir = TestDir::new("cuda-parity");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    let identical = b">record\nACGTTGCAAACGTTGCAAACGTTGCAAACGTTGCAAACGT\n";
    for name in ["a.fna", "b.fna", "c.fna"] {
        fs::write(inputs.join(name), identical).unwrap();
    }

    for canonical in ["true", "false"] {
        let cpu_output = dir.path().join(format!("cpu-{canonical}.sketch"));
        let cpu_metrics = dir.path().join(format!("cpu-{canonical}-metrics"));
        let cpu = run([
            "sketch",
            "--path",
            inputs.to_str().unwrap(),
            "--out",
            cpu_output.to_str().unwrap(),
            "--device",
            "cpu",
            "--ksize",
            "16",
            "--hv-d",
            "256",
            "--canonical",
            canonical,
            "--threads",
            "3",
            "--metrics-out",
            cpu_metrics.to_str().unwrap(),
        ]);
        assert_success(&cpu);
        let cpu_sketch = dotani::utils::load_sketch(&cpu_output);
        let cpu_ull =
            dotani::utils::load_ull_sketch(&PathBuf::from(format!("{}.ull", cpu_output.display())));
        let cpu_sketch_bytes = fs::read(&cpu_output).unwrap();
        let cpu_ull_bytes = fs::read(format!("{}.ull", cpu_output.display())).unwrap();
        let cpu_metrics_counts = metric_counts(&cpu_metrics);

        for dedup in ["hashset", "sort_unstable"] {
            for thread_count in [Some("1"), Some("3"), None] {
                let thread_label = thread_count.unwrap_or("default");
                let cuda_output = dir
                    .path()
                    .join(format!("cuda-{canonical}-{dedup}-t{thread_label}.sketch"));
                let cuda_metrics = dir
                    .path()
                    .join(format!("cuda-{canonical}-{dedup}-t{thread_label}-metrics"));
                let mut args = vec![
                    "sketch".to_string(),
                    "--path".to_string(),
                    inputs.display().to_string(),
                    "--out".to_string(),
                    cuda_output.display().to_string(),
                    "--device".to_string(),
                    "cuda".to_string(),
                    "--ksize".to_string(),
                    "16".to_string(),
                    "--hv-d".to_string(),
                    "256".to_string(),
                    "--canonical".to_string(),
                    canonical.to_string(),
                    "--cuda-dedup".to_string(),
                    dedup.to_string(),
                    "--metrics-out".to_string(),
                    cuda_metrics.display().to_string(),
                ];
                if let Some(thread_count) = thread_count {
                    args.push("--threads".to_string());
                    args.push(thread_count.to_string());
                }

                let cuda = run(args);
                assert_success(&cuda);

                let cuda_sketch = dotani::utils::load_sketch(&cuda_output);
                let cuda_ull = dotani::utils::load_ull_sketch(&PathBuf::from(format!(
                    "{}.ull",
                    cuda_output.display()
                )));
                assert_eq!(cuda_sketch, cpu_sketch);
                assert_eq!(cuda_ull, cpu_ull);
                assert_eq!(fs::read(&cuda_output).unwrap(), cpu_sketch_bytes);
                assert_eq!(
                    fs::read(format!("{}.ull", cuda_output.display())).unwrap(),
                    cpu_ull_bytes
                );
                assert_eq!(metric_counts(&cuda_metrics), cpu_metrics_counts);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cpu_and_cuda_sketches_match_for_multirecord_lowercase_and_ambiguous_input() {
    let dir = TestDir::new("cuda-parity-special-bases");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    let contents = b">first\nacgtacgtacgtacgtacgtNacgtacgtacgtacgtacgt\n>second\nttACG\n";
    for name in ["a.fna", "b.fna", "c.fna"] {
        fs::write(inputs.join(name), contents).unwrap();
    }

    for canonical in ["true", "false"] {
        let cpu_output = dir.path().join(format!("cpu-{canonical}.sketch"));
        let cpu_metrics = dir.path().join(format!("cpu-{canonical}-metrics"));
        let cpu = run([
            "sketch",
            "--path",
            inputs.to_str().unwrap(),
            "--out",
            cpu_output.to_str().unwrap(),
            "--device",
            "cpu",
            "--ksize",
            "3",
            "--hv-d",
            "256",
            "--canonical",
            canonical,
            "--threads",
            "3",
            "--metrics-out",
            cpu_metrics.to_str().unwrap(),
        ]);
        assert_success(&cpu);

        let cpu_sketch = dotani::utils::load_sketch(&cpu_output);
        let cpu_ull =
            dotani::utils::load_ull_sketch(&PathBuf::from(format!("{}.ull", cpu_output.display())));
        let cpu_metrics_counts = metric_counts(&cpu_metrics);

        for dedup in ["hashset", "sort_unstable"] {
            let cuda_output = dir.path().join(format!("cuda-{canonical}-{dedup}.sketch"));
            let cuda_metrics = dir.path().join(format!("cuda-{canonical}-{dedup}-metrics"));
            let cuda = run([
                "sketch",
                "--path",
                inputs.to_str().unwrap(),
                "--out",
                cuda_output.to_str().unwrap(),
                "--device",
                "cuda",
                "--ksize",
                "3",
                "--hv-d",
                "256",
                "--canonical",
                canonical,
                "--cuda-dedup",
                dedup,
                "--threads",
                "3",
                "--metrics-out",
                cuda_metrics.to_str().unwrap(),
            ]);
            assert_success(&cuda);

            let cuda_sketch = dotani::utils::load_sketch(&cuda_output);
            let cuda_ull = dotani::utils::load_ull_sketch(&PathBuf::from(format!(
                "{}.ull",
                cuda_output.display()
            )));
            assert_eq!(cuda_sketch, cpu_sketch);
            assert_eq!(cuda_ull, cpu_ull);
            assert_eq!(metric_counts(&cuda_metrics), cpu_metrics_counts);
        }
    }
}
