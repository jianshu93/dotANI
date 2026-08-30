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
