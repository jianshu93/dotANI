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
fn cli_help_defaults() {
    let sketch_help = run(["sketch", "--help"]);
    assert_success(&sketch_help);
    let sketch_text = String::from_utf8_lossy(&sketch_help.stdout);
    assert!(sketch_text.contains("--hv-d"));
    assert!(sketch_text.contains("sort_unstable"));

    let dist_help = run(["dist", "--help"]);
    assert_success(&dist_help);
    let dist_text = String::from_utf8_lossy(&dist_help.stdout);
    assert!(dist_text.contains("--output-mode"));
    assert!(dist_text.contains("rows"));
}

#[test]
fn ull_and_ell_cli_round_trip() {
    let dir = TestDir::new("cardinality-roundtrip");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    write_valid_fasta(&inputs);
    fs::write(
        inputs.join("second.fna"),
        b">record-2\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAAAA\n",
    )
    .unwrap();

    for (flag, extension) in [("--ull", "ull"), ("--ell", "ell")] {
        let sketch_path = dir.path().join(format!("{extension}.sketch"));
        let sketch = run([
            "sketch",
            "--path",
            inputs.to_str().unwrap(),
            "--out",
            sketch_path.to_str().unwrap(),
            "--device",
            "cpu",
            "--ksize",
            "3",
            "--hv-d",
            "256",
            flag,
        ]);
        assert_success(&sketch);

        let sidecar = PathBuf::from(format!("{}.{extension}", sketch_path.display()));
        match extension {
            "ull" => assert_eq!(dotani::utils::load_ull_sketch(&sidecar).unwrap().len(), 2),
            "ell" => assert_eq!(dotani::utils::load_ell_sketch(&sidecar).unwrap().len(), 2),
            _ => unreachable!(),
        }

        let ani_path = dir.path().join(format!("{extension}.tsv"));
        let dist = run([
            "dist",
            "--path-r",
            sketch_path.to_str().unwrap(),
            "--path-q",
            sketch_path.to_str().unwrap(),
            "--out",
            ani_path.to_str().unwrap(),
            "--ani-th",
            "0",
            "--threads",
            "1",
            flag,
        ]);
        assert_success(&dist);
        assert!(!fs::read_to_string(ani_path).unwrap().is_empty());
    }
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
    let hd_sketches = dotani::utils::load_sketch(&sketch_path).unwrap();
    let hd_ids: Vec<&str> = hd_sketches
        .iter()
        .map(|sketch| sketch.file_str.as_str())
        .collect();
    assert_eq!(hd_ids, expected_ids);

    let ull_sidecar = PathBuf::from(format!("{}.ull", sketch_path.display()));
    let ull_sketches = dotani::utils::load_ull_sketch(&ull_sidecar).unwrap();
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

    let path_hd = dotani::utils::load_sketch(&path_sketch_path).unwrap();
    let path_ids: Vec<String> = path_hd
        .iter()
        .map(|sketch| sketch.file_str.clone())
        .collect();
    let lexical_ids = [alpha.display().to_string(), zeta.display().to_string()];
    assert_eq!(path_ids, lexical_ids);
}

#[test]
fn cli_rejects_conflicting_paths_and_invalid_values() {
    let dir = TestDir::new("validation");
    let fasta_dir = dir.path().join("inputs");
    fs::create_dir_all(&fasta_dir).unwrap();
    write_valid_fasta(&fasta_dir);
    let manifest = dir.path().join("manifest.tsv");
    fs::write(&manifest, "read_path\tfile_id\n").unwrap();

    let conflict = run([
        "sketch",
        "--path",
        fasta_dir.to_str().unwrap(),
        "--manifest",
        manifest.to_str().unwrap(),
        "--out",
        dir.path().join("conflict.sketch").to_str().unwrap(),
    ]);
    assert_failure_contains(&conflict, "cannot be used with");

    for hv_d in ["0", "70", "257"] {
        let output = run([
            "sketch",
            "--path",
            fasta_dir.to_str().unwrap(),
            "--out",
            dir.path()
                .join(format!("hv-{hv_d}.sketch"))
                .to_str()
                .unwrap(),
            "--hv-d",
            hv_d,
            "--device",
            "cpu",
        ]);
        assert_failure_contains(&output, "hv_d");
    }

    let invalid_ull = run([
        "sketch",
        "--path",
        fasta_dir.to_str().unwrap(),
        "--out",
        dir.path().join("bad-ull.sketch").to_str().unwrap(),
        "--ull-p",
        "2",
        "--device",
        "cpu",
    ]);
    assert_failure_contains(&invalid_ull, "ULL precision");

    for threshold in ["NaN", "101"] {
        let output = run([
            "dist",
            "--path-r",
            "missing.sketch",
            "--path-q",
            "missing.sketch",
            "--out",
            dir.path()
                .join(format!("ani-{threshold}.tsv"))
                .to_str()
                .unwrap(),
            "--ani-th",
            threshold,
        ]);
        assert_failure_contains(&output, "ANI threshold");
    }

    let negative_threshold = run([
        "dist",
        "--path-r",
        "missing.sketch",
        "--path-q",
        "missing.sketch",
        "--out",
        dir.path().join("ani-negative.tsv").to_str().unwrap(),
        "--ani-th=-1",
    ]);
    assert_failure_contains(&negative_threshold, "ANI threshold");

    let zero_threads = run([
        "dist",
        "--path-r",
        "missing.sketch",
        "--path-q",
        "missing.sketch",
        "--out",
        dir.path().join("zero-threads.tsv").to_str().unwrap(),
        "--threads",
        "0",
    ]);
    assert_failure_contains(&zero_threads, "greater than zero");

    let zero_ksize = run([
        "sketch",
        "--path",
        fasta_dir.to_str().unwrap(),
        "--out",
        dir.path().join("zero-k.sketch").to_str().unwrap(),
        "--ksize",
        "0",
        "--device",
        "cpu",
    ]);
    assert_failure_contains(&zero_ksize, "ksize");

    #[cfg(feature = "cuda")]
    {
        let cuda_ksize = run([
            "sketch",
            "--path",
            fasta_dir.to_str().unwrap(),
            "--out",
            dir.path().join("cuda-large-k.sketch").to_str().unwrap(),
            "--device",
            "cuda",
            "--ksize",
            "33",
        ]);
        assert_failure_contains(&cuda_ksize, "up to 32");
    }
}

#[test]
fn valid_hv_dimensions_round_trip_and_count_output_mode() {
    let dir = TestDir::new("roundtrip");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    write_valid_fasta(&inputs);

    for hv_d in ["256", "1024", "4096"] {
        let output = dir.path().join(format!("{hv_d}.sketch"));
        let sketch = run([
            "sketch",
            "--path",
            inputs.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
            "--device",
            "cpu",
            "--ksize",
            "3",
            "--hv-d",
            hv_d,
            "--canonical",
            "false",
        ]);
        assert_success(&sketch);
        assert!(output.is_file());
        assert!(PathBuf::from(format!("{}.ull", output.display())).is_file());
    }

    let output = dir.path().join("256.sketch");
    let ani_output = dir.path().join("count.tsv");
    let dist = run([
        "dist",
        "--path-r",
        output.to_str().unwrap(),
        "--path-q",
        output.to_str().unwrap(),
        "--out",
        ani_output.to_str().unwrap(),
        "--output-mode",
        "count",
        "--ani-th",
        "0",
        "--threads",
        "1",
    ]);
    assert_success(&dist);
    assert!(
        fs::read_to_string(ani_output)
            .unwrap()
            .contains("mode\tcount")
    );
}

#[test]
fn cli_rejects_empty_and_bad_inputs() {
    let dir = TestDir::new("bad-inputs");

    let empty_inputs = dir.path().join("empty-inputs");
    fs::create_dir_all(&empty_inputs).unwrap();
    let empty_dir = run([
        "sketch",
        "--path",
        empty_inputs.to_str().unwrap(),
        "--out",
        dir.path().join("empty.sketch").to_str().unwrap(),
        "--device",
        "cpu",
    ]);
    assert_failure_contains(&empty_dir, "no supported FASTA files");

    let empty_manifest = dir.path().join("empty.tsv");
    fs::write(&empty_manifest, "read_path\tfile_id\n").unwrap();
    let manifest = run([
        "sketch",
        "--manifest",
        empty_manifest.to_str().unwrap(),
        "--out",
        dir.path().join("empty-manifest.sketch").to_str().unwrap(),
        "--device",
        "cpu",
    ]);
    assert_failure_contains(&manifest, "header but no data rows");

    let no_kmers = dir.path().join("no-kmers");
    fs::create_dir_all(&no_kmers).unwrap();
    fs::write(no_kmers.join("ambiguous.fna"), b">r\nNNNNNN\n").unwrap();
    let no_valid = run([
        "sketch",
        "--path",
        no_kmers.to_str().unwrap(),
        "--out",
        dir.path().join("no-kmers.sketch").to_str().unwrap(),
        "--device",
        "cpu",
        "--ksize",
        "3",
    ]);
    assert_failure_contains(&no_valid, "no valid");

    let malformed = dir.path().join("malformed");
    fs::create_dir_all(&malformed).unwrap();
    fs::write(malformed.join("broken.fna"), b"@r\nACGT\n+\n!!\n").unwrap();
    let malformed_output = run([
        "sketch",
        "--path",
        malformed.to_str().unwrap(),
        "--out",
        dir.path().join("malformed.sketch").to_str().unwrap(),
        "--device",
        "cpu",
        "--ksize",
        "3",
    ]);
    assert_failure_contains(&malformed_output, "parse");
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
        let cpu_sketch = dotani::utils::load_sketch(&cpu_output).unwrap();
        let cpu_ull =
            dotani::utils::load_ull_sketch(&PathBuf::from(format!("{}.ull", cpu_output.display())))
                .unwrap();
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

                let cuda_sketch = dotani::utils::load_sketch(&cuda_output).unwrap();
                let cuda_ull = dotani::utils::load_ull_sketch(&PathBuf::from(format!(
                    "{}.ull",
                    cuda_output.display()
                )))
                .unwrap();
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
fn cpu_and_cuda_ell_sketches_are_byte_identical_for_both_dedup_strategies() {
    let dir = TestDir::new("cuda-ell-parity");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    write_valid_fasta(&inputs);

    let cpu_output = dir.path().join("cpu.sketch");
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
        "--ell",
    ]);
    assert_success(&cpu);
    let cpu_sketch = fs::read(&cpu_output).unwrap();
    let cpu_ell = fs::read(format!("{}.ell", cpu_output.display())).unwrap();

    for dedup in ["hashset", "sort_unstable"] {
        let cuda_output = dir.path().join(format!("cuda-{dedup}.sketch"));
        let cuda = run([
            "sketch",
            "--path",
            inputs.to_str().unwrap(),
            "--out",
            cuda_output.to_str().unwrap(),
            "--device",
            "cuda",
            "--cuda-dedup",
            dedup,
            "--ksize",
            "3",
            "--hv-d",
            "256",
            "--ell",
        ]);
        assert_success(&cuda);
        assert_eq!(fs::read(&cuda_output).unwrap(), cpu_sketch);
        assert_eq!(
            fs::read(format!("{}.ell", cuda_output.display())).unwrap(),
            cpu_ell
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dist_failure_does_not_fall_back_to_cpu() {
    let dir = TestDir::new("cuda-dist-fail-fast");
    let inputs = dir.path().join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    write_valid_fasta(&inputs);

    let sketch_path = dir.path().join("input.sketch");
    let sketch = run([
        "sketch",
        "--path",
        inputs.to_str().unwrap(),
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

    let output = Command::new(env!("CARGO_BIN_EXE_dotani"))
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .args([
            "dist",
            "--path-r",
            sketch_path.to_str().unwrap(),
            "--path-q",
            sketch_path.to_str().unwrap(),
            "--out",
            dir.path().join("ani.tsv").to_str().unwrap(),
            "--ani-th",
            "0",
        ])
        .output()
        .expect("failed to execute dotani");
    assert!(!output.status.success(), "CUDA dist unexpectedly succeeded");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!text.contains("falling back to CPU"), "{text}");
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

        let cpu_sketch = dotani::utils::load_sketch(&cpu_output).unwrap();
        let cpu_ull =
            dotani::utils::load_ull_sketch(&PathBuf::from(format!("{}.ull", cpu_output.display())))
                .unwrap();
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

            let cuda_sketch = dotani::utils::load_sketch(&cuda_output).unwrap();
            let cuda_ull = dotani::utils::load_ull_sketch(&PathBuf::from(format!(
                "{}.ull",
                cuda_output.display()
            )))
            .unwrap();
            assert_eq!(cuda_sketch, cpu_sketch);
            assert_eq!(cuda_ull, cpu_ull);
            assert_eq!(metric_counts(&cuda_metrics), cpu_metrics_counts);
        }
    }
}
