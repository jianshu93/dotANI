use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};

use anyhow::{Result, anyhow, bail};
use log::{info, warn};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{hd, types::*};

pub fn get_fasta_files(path: &PathBuf) -> Vec<PathBuf> {
    let mut all_files = Vec::new();

    for pattern in [
        "*.fna",
        "*.fa",
        "*.fasta",
        "*.fna.gz",
        "*.fa.gz",
        "*.fasta.gz",
        "*.fna.bz2",
        "*.fa.bz2",
        "*.fasta.bz2",
        "*.fna.xz",
        "*.fa.xz",
        "*.fasta.xz",
        "*.fna.zst",
        "*.fa.zst",
        "*.fasta.zst",
    ] {
        let mut files: Vec<_> = glob(path.join(pattern).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .map(|f| f.unwrap())
            .collect();

        let mut recursive_files: Vec<_> = glob(path.join("**").join(pattern).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .map(|f| f.unwrap())
            .collect();

        all_files.append(&mut files);
        all_files.append(&mut recursive_files);
    }

    all_files.sort();
    all_files.dedup();
    all_files
}

pub fn get_sketch_inputs(params: &SketchParams) -> Result<Vec<SketchInput>> {
    if let Some(manifest) = &params.manifest {
        read_sketch_manifest(manifest)
    } else {
        Ok(get_fasta_files(&params.path)
            .into_iter()
            .map(|read_path| SketchInput {
                file_id: read_path.display().to_string(),
                read_path,
            })
            .collect())
    }
}

/// Parses a two-column TSV sketch manifest with `read_path` and `file_id` columns.
///
/// Row order is preserved and `file_id` values must be unique. `#` comment
/// lines are ignored and CRLF line endings are accepted. Relative `read_path`
/// values are interpreted relative to the process working directory, not the
/// manifest's directory; prefer absolute paths for portable or staged manifests.
pub fn read_sketch_manifest(path: &Path) -> Result<Vec<SketchInput>> {
    let contents = fs::read_to_string(path)
        .map_err(|e| anyhow!("failed to read manifest {}: {}", path.display(), e))?;

    if contents.as_bytes().contains(&0) {
        bail!("manifest {} contains NUL bytes", path.display());
    }

    let mut lines = contents.lines().enumerate();
    let (_, header_line) = lines
        .find(|(_, raw_line)| {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            !line.is_empty() && !line.starts_with('#')
        })
        .ok_or_else(|| anyhow!("manifest {} does not contain a header row", path.display()))?;
    let header_line = header_line.strip_suffix('\r').unwrap_or(header_line);
    let header: Vec<&str> = header_line.split('\t').collect();
    let read_path_idx = header
        .iter()
        .position(|field| *field == "read_path")
        .ok_or_else(|| {
            anyhow!(
                "manifest {} is missing required read_path column",
                path.display()
            )
        })?;
    let file_id_idx = header
        .iter()
        .position(|field| *field == "file_id")
        .ok_or_else(|| {
            anyhow!(
                "manifest {} is missing required file_id column",
                path.display()
            )
        })?;
    let header_len = header.len();
    let mut inputs = Vec::new();
    let mut seen_file_ids = HashSet::new();

    for (line_idx, raw_line) in lines {
        let line_no = line_idx + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);

        if line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != header_len {
            bail!(
                "manifest {} line {} has {} field(s), expected {}",
                path.display(),
                line_no,
                fields.len(),
                header_len
            );
        }

        let read_path = fields[read_path_idx];
        let file_id = fields[file_id_idx];

        if read_path.is_empty() {
            bail!(
                "manifest {} line {} has empty read_path",
                path.display(),
                line_no
            );
        }
        if file_id.is_empty() {
            bail!(
                "manifest {} line {} has empty file_id",
                path.display(),
                line_no
            );
        }
        if !seen_file_ids.insert(file_id.to_string()) {
            bail!(
                "manifest {} line {} duplicates file_id {:?}",
                path.display(),
                line_no,
                file_id
            );
        }

        let read_path = PathBuf::from(read_path);
        if !read_path.exists() {
            bail!(
                "manifest {} line {} read_path does not exist: {}",
                path.display(),
                line_no,
                read_path.display()
            );
        }
        if !read_path.is_file() {
            bail!(
                "manifest {} line {} read_path is not a file: {}",
                path.display(),
                line_no,
                read_path.display()
            );
        }

        inputs.push(SketchInput {
            read_path,
            file_id: file_id.to_string(),
        });
    }

    if inputs.is_empty() {
        bail!(
            "manifest {} contains a header but no data rows",
            path.display()
        );
    }

    Ok(inputs)
}

pub fn get_progress_bar(n_file: usize) -> ProgressBar {
    let pb = ProgressBar::new(n_file as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{wide_bar} {pos}/{len} ({percent}%) - Elapsed: {elapsed_precise}, ETA: {eta_precise}",
            )
            .unwrap(),
    );

    pb
}

pub fn dump_sketch(file_sketch: &Vec<FileSketch>, out_file_path: &PathBuf) {
    let out_filename = out_file_path.to_str().unwrap();

    let serialized = bincode::serialize::<Vec<FileSketch>>(file_sketch).unwrap();
    fs::write(out_filename, &serialized).expect("Dump sketch file failed!");

    let sketch_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    info!(
        "Dump sketch file to {} with size {:.2} MB",
        out_filename, sketch_size_mb
    );
}

pub fn load_sketch(path: &Path) -> Vec<FileSketch> {
    info!("Loading sketch from {}", path.to_str().unwrap());
    let serialized = fs::read(path).expect("Opening sketch file failed!");
    bincode::deserialize::<Vec<FileSketch>>(&serialized[..]).unwrap()
}

pub fn dump_ull_sketch(file_ull_sketch: &Vec<FileUllSketch>, out_file_path: &PathBuf) {
    let out_filename = out_file_path.to_str().unwrap();

    let serialized = bincode::serialize::<Vec<FileUllSketch>>(file_ull_sketch).unwrap();

    let n_threads = if serialized.len() < 8 * 1024 * 1024 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(32)
    };

    let mut encoder =
        zstd::stream::Encoder::new(Vec::new(), 3).expect("Failed to create zstd encoder");

    if n_threads > 1 {
        encoder
            .multithread(n_threads as u32)
            .expect("Failed to enable zstd multithreading");
    }

    encoder
        .write_all(serialized.as_slice())
        .expect("Failed to write ULL bytes into zstd encoder");

    let compressed = encoder.finish().expect("Failed to finalize zstd encoding");

    fs::write(out_filename, &compressed).expect("Dump ULL sketch file failed!");

    let raw_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    let compressed_size_mb = compressed.len() as f32 / 1024.0 / 1024.0;
    let ratio = if serialized.is_empty() {
        1.0
    } else {
        compressed.len() as f32 / serialized.len() as f32
    };

    info!(
        "Dump compressed ULL sketch file to {} with compressed size {:.2} MB (raw {:.2} MB, ratio {:.3}, zstd threads {})",
        out_filename, compressed_size_mb, raw_size_mb, ratio, n_threads
    );
}

pub fn dump_sketch_metrics(metrics: &[FileSketchMetrics], prefix: &Path, sketch_wall_ns: u128) {
    let summary_path = PathBuf::from(format!("{}.summary.tsv", prefix.to_string_lossy()));
    let files_path = PathBuf::from(format!("{}.files.tsv", prefix.to_string_lossy()));

    let mut files_tsv = String::new();
    files_tsv.push_str(metrics_header());
    files_tsv.push('\n');
    for metric in metrics {
        files_tsv.push_str(&metric_row(metric));
        files_tsv.push('\n');
    }
    fs::write(&files_path, files_tsv).expect("Dump sketch file metrics failed!");

    let mut summary = FileSketchMetrics::default();
    summary.file = String::from("TOTAL");
    summary.sketch_wall_ns = Some(sketch_wall_ns);
    for metric in metrics {
        summary.input_bases += metric.input_bases;
        summary.hashes_seen += metric.hashes_seen;
        summary.unique_hashes += metric.unique_hashes;
        summary.fasta_ns += metric.fasta_ns;
        summary.hash_and_dedup_ns += metric.hash_and_dedup_ns;
        summary.hd_encode_ns += metric.hd_encode_ns;
        summary.hv_norm_ns += metric.hv_norm_ns;
        summary.hd_compress_ns += metric.hd_compress_ns;
        summary.total_worker_ns += metric.total_worker_ns;
        summary.cuda_h2d_ns = add_optional_ns(summary.cuda_h2d_ns, metric.cuda_h2d_ns);
        summary.cuda_alloc_ns = add_optional_ns(summary.cuda_alloc_ns, metric.cuda_alloc_ns);
        summary.cuda_launch_ns = add_optional_ns(summary.cuda_launch_ns, metric.cuda_launch_ns);
        summary.cuda_d2h_ns = add_optional_ns(summary.cuda_d2h_ns, metric.cuda_d2h_ns);
        summary.cuda_zero_filter_ns =
            add_optional_ns(summary.cuda_zero_filter_ns, metric.cuda_zero_filter_ns);
        summary.cuda_filter_ns = add_optional_ns(summary.cuda_filter_ns, metric.cuda_filter_ns);
        summary.cuda_hd_hash_h2d_ns =
            add_optional_ns(summary.cuda_hd_hash_h2d_ns, metric.cuda_hd_hash_h2d_ns);
        summary.cuda_hd_hv_h2d_ns =
            add_optional_ns(summary.cuda_hd_hv_h2d_ns, metric.cuda_hd_hv_h2d_ns);
        summary.cuda_hd_alloc_ns =
            add_optional_ns(summary.cuda_hd_alloc_ns, metric.cuda_hd_alloc_ns);
        summary.cuda_hd_kernel_launch_ns = add_optional_ns(
            summary.cuda_hd_kernel_launch_ns,
            metric.cuda_hd_kernel_launch_ns,
        );
        summary.cuda_hd_d2h_ns = add_optional_ns(summary.cuda_hd_d2h_ns, metric.cuda_hd_d2h_ns);
    }

    let mut summary_tsv = String::new();
    summary_tsv.push_str(metrics_header());
    summary_tsv.push('\n');
    summary_tsv.push_str(&metric_row(&summary));
    summary_tsv.push('\n');
    fs::write(&summary_path, summary_tsv).expect("Dump sketch summary metrics failed!");

    info!(
        "Wrote sketch metrics to {} and {}",
        summary_path.display(),
        files_path.display()
    );
}

fn add_optional_ns(left: Option<u128>, right: Option<u128>) -> Option<u128> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn metrics_header() -> &'static str {
    "file\tinput_bases\thashes_seen\tunique_hashes\tfasta_ns\thash_and_dedup_ns\thd_encode_ns\thv_norm_ns\thd_compress_ns\ttotal_worker_ns\tsketch_wall_ns\tcuda_stream_lane\tcuda_h2d_ns\tcuda_alloc_ns\tcuda_launch_ns\tcuda_d2h_ns\tcuda_zero_filter_ns\tcuda_filter_ns\tcuda_hd_hash_h2d_ns\tcuda_hd_hv_h2d_ns\tcuda_hd_alloc_ns\tcuda_hd_kernel_launch_ns\tcuda_hd_d2h_ns"
}

fn metric_row(metric: &FileSketchMetrics) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        metric.file,
        metric.input_bases,
        metric.hashes_seen,
        metric.unique_hashes,
        metric.fasta_ns,
        metric.hash_and_dedup_ns,
        metric.hd_encode_ns,
        metric.hv_norm_ns,
        metric.hd_compress_ns,
        metric.total_worker_ns,
        optional_ns(metric.sketch_wall_ns),
        optional_usize(metric.cuda_stream_lane),
        optional_ns(metric.cuda_h2d_ns),
        optional_ns(metric.cuda_alloc_ns),
        optional_ns(metric.cuda_launch_ns),
        optional_ns(metric.cuda_d2h_ns),
        optional_ns(metric.cuda_zero_filter_ns),
        optional_ns(metric.cuda_filter_ns),
        optional_ns(metric.cuda_hd_hash_h2d_ns),
        optional_ns(metric.cuda_hd_hv_h2d_ns),
        optional_ns(metric.cuda_hd_alloc_ns),
        optional_ns(metric.cuda_hd_kernel_launch_ns),
        optional_ns(metric.cuda_hd_d2h_ns)
    )
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| String::from("NA"))
}

fn optional_ns(value: Option<u128>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| String::from("NA"))
}

pub fn load_ull_sketch(path: &Path) -> Vec<FileUllSketch> {
    info!("Loading ULL sketch from {}", path.to_str().unwrap());
    let bytes = fs::read(path).expect("Opening ULL sketch file failed!");

    // New format: zstd-compressed bincode
    if let Ok(serialized) = zstd::stream::decode_all(bytes.as_slice()) {
        if let Ok(v) = bincode::deserialize::<Vec<FileUllSketch>>(&serialized[..]) {
            return v;
        }
    }

    // Backward compatibility: old raw bincode format
    if let Ok(v) = bincode::deserialize::<Vec<FileUllSketch>>(&bytes[..]) {
        warn!(
            "ULL sketch file {} is in legacy uncompressed format",
            path.to_string_lossy()
        );
        return v;
    }

    panic!(
        "Failed to load ULL sketch file {} as either zstd-compressed or legacy uncompressed format",
        path.to_string_lossy()
    );
}

pub fn dump_ani_file(sketch_dist: &SketchDist) {
    let mut indices = (0..sketch_dist.file_ani.len()).collect::<Vec<_>>();
    indices.sort_by(|&i1, &i2| {
        sketch_dist.file_ani[i1]
            .1
            .partial_cmp(&sketch_dist.file_ani[i2].1)
            .unwrap()
    });
    indices.reverse();

    let mut csv_str = String::new();
    let mut cnt: f32 = 0.0;
    for i in 0..sketch_dist.file_ani.len() {
        if sketch_dist.file_ani[indices[i]].1 >= sketch_dist.ani_threshold {
            csv_str.push_str(&format!(
                "{}\t{}\t{:.3}\n",
                sketch_dist.file_ani[indices[i]].0.0,
                sketch_dist.file_ani[indices[i]].0.1,
                sketch_dist.file_ani[indices[i]].1
            ));
            cnt += 1.0;
        } else {
            break;
        }
    }

    fs::write(sketch_dist.out_file.to_str().unwrap(), csv_str.as_bytes())
        .expect("Dump ANI file failed!");

    let total_dist = sketch_dist.file_ani.len() as f32;
    let perc = cnt / total_dist * 100.0;
    if perc < 5.0 {
        warn!(
            "Output ANIs with threshold {:.1} are too divergent: {} of {} ({:.2}%) ANIs are reported",
            sketch_dist.ani_threshold, cnt, total_dist, perc
        );
    } else {
        info!(
            "Output {} of {} ANIs above threshold {:.1} to file {}",
            cnt,
            total_dist,
            sketch_dist.ani_threshold,
            sketch_dist.out_file.to_str().unwrap()
        )
    }
}

pub fn dump_distribution_to_txt(path: &Path) {
    let mut file_sketch = load_sketch(path);

    hd::decompress_file_sketch(&mut file_sketch);

    let data: Vec<Vec<i32>> = (0..file_sketch.len())
        .map(|i| file_sketch[i].hv.clone())
        .collect();

    let mut hist: HashMap<i32, u32> = HashMap::new();
    for row in &data {
        for v in row {
            if hist.get(v).is_none() {
                hist.insert(*v, 1);
            } else if let Some(c) = hist.get_mut(v) {
                *c += 1;
            }
        }
    }

    for kv in hist {
        println!("{}\t{}", kv.0, kv.1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "dotani_manifest_test_{}_{}_{}",
            std::process::id(),
            name,
            unique
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn manifest_preserves_row_order_and_ids() {
        let dir = test_dir("order");
        let a = dir.join("a.fna");
        let b = dir.join("b.fna");
        let manifest = dir.join("manifest.tsv");
        fs::write(&a, ">a\nACGT\n").unwrap();
        fs::write(&b, ">b\nACGT\n").unwrap();
        fs::write(
            &manifest,
            format!(
                "# version=1\nread_path\tfile_id\trel_path\n{}\tgenome-b\tb.fna\n{}\tgenome-a\ta.fna\n",
                b.display(),
                a.display()
            ),
        )
        .unwrap();

        let inputs = read_sketch_manifest(&manifest).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].read_path, b);
        assert_eq!(inputs[0].file_id, "genome-b");
        assert_eq!(inputs[1].read_path, a);
        assert_eq!(inputs[1].file_id, "genome-a");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_parses_crlf_and_skips_interleaved_comments() {
        let dir = test_dir("crlf");
        let a = dir.join("a.fna");
        let b = dir.join("b.fna");
        let manifest = dir.join("manifest.tsv");
        fs::write(&a, ">a\nACGT\n").unwrap();
        fs::write(&b, ">b\nACGT\n").unwrap();
        let body = format!(
            "# version=1\r\nread_path\tfile_id\r\n{}\tgenome-b\r\n# staged second\r\n{}\tgenome-a\r\n",
            b.display(),
            a.display()
        );
        fs::write(&manifest, body).unwrap();

        let inputs = read_sketch_manifest(&manifest).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].read_path, b);
        assert_eq!(inputs[0].file_id, "genome-b");
        assert_eq!(inputs[1].read_path, a);
        assert_eq!(inputs[1].file_id, "genome-a");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_rejects_duplicate_file_ids() {
        let dir = test_dir("duplicate");
        let a = dir.join("a.fna");
        let b = dir.join("b.fna");
        let manifest = dir.join("manifest.tsv");
        fs::write(&a, ">a\nACGT\n").unwrap();
        fs::write(&b, ">b\nACGT\n").unwrap();
        fs::write(
            &manifest,
            format!(
                "read_path\tfile_id\n{}\tgenome\n{}\tgenome\n",
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let err = read_sketch_manifest(&manifest).expect_err("duplicate IDs should fail");

        assert!(err.to_string().contains("duplicates file_id"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_rejects_malformed_rows() {
        let dir = test_dir("malformed");
        let a = dir.join("a.fna");
        let manifest = dir.join("manifest.tsv");
        fs::write(&a, ">a\nACGT\n").unwrap();
        fs::write(
            &manifest,
            format!("read_path\tfile_id\n{}\tgenome\textra\n", a.display()),
        )
        .unwrap();

        let err = read_sketch_manifest(&manifest).expect_err("malformed rows should fail");

        assert!(err.to_string().contains("expected 2"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_rejects_missing_read_paths() {
        let dir = test_dir("missing");
        let missing = dir.join("missing.fna");
        let manifest = dir.join("manifest.tsv");
        fs::write(
            &manifest,
            format!("read_path\tfile_id\n{}\tgenome\n", missing.display()),
        )
        .unwrap();

        let err = read_sketch_manifest(&manifest).expect_err("missing read_path should fail");

        assert!(err.to_string().contains("read_path does not exist"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_rejects_directory_read_paths() {
        let dir = test_dir("directory");
        let manifest = dir.join("manifest.tsv");
        fs::write(
            &manifest,
            format!("read_path\tfile_id\n{}\tgenome\n", dir.display()),
        )
        .unwrap();

        let err = read_sketch_manifest(&manifest).expect_err("directory read_path should fail");

        assert!(err.to_string().contains("read_path is not a file"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn path_mode_uses_path_as_file_id() {
        let dir = test_dir("path_mode");
        let a = dir.join("a.fna");
        fs::write(&a, ">a\nACGT\n").unwrap();
        let params = SketchParams {
            path: dir.clone(),
            ..SketchParams::default()
        };

        let inputs = get_sketch_inputs(&params).unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].read_path, a);
        assert_eq!(inputs[0].file_id, a.display().to_string());
        fs::remove_dir_all(dir).unwrap();
    }
}
