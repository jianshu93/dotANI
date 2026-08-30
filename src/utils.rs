use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};

use anyhow::{Context, Result, anyhow, bail};
use log::{info, warn};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{hd, types::*};

pub fn get_fasta_files(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.exists() {
        bail!("input directory does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("input path is not a directory: {}", path.display());
    }

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
        let direct_pattern = path.join(pattern).to_string_lossy().into_owned();
        let recursive_pattern = path.join("**").join(pattern).to_string_lossy().into_owned();
        let mut files = glob(&direct_pattern)
            .map_err(|e| anyhow!("failed to read FASTA glob pattern {direct_pattern:?}: {e}"))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("failed to inspect FASTA path: {e}"))?;

        let mut recursive_files = glob(&recursive_pattern)
            .map_err(|e| anyhow!("failed to read FASTA glob pattern {recursive_pattern:?}: {e}"))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("failed to inspect FASTA path: {e}"))?;

        all_files.append(&mut files);
        all_files.append(&mut recursive_files);
    }

    all_files.sort();
    all_files.dedup();
    if all_files.is_empty() {
        bail!(
            "input directory {} contains no supported FASTA files",
            path.display()
        );
    }

    Ok(all_files)
}

pub fn get_sketch_inputs(params: &SketchParams) -> Result<Vec<SketchInput>> {
    if let Some(manifest) = &params.manifest {
        read_sketch_manifest(manifest)
    } else {
        get_fasta_files(&params.path).map(|files| {
            files
                .into_iter()
                .map(|read_path| SketchInput {
                    file_id: read_path.display().to_string(),
                    read_path,
                })
                .collect()
        })
    }
}

pub(crate) fn validate_sketch_params(params: &SketchParams) -> Result<()> {
    if params.ksize == 0 {
        bail!("ksize must be greater than zero");
    }
    if params.scaled == 0 {
        bail!("scaled must be greater than zero");
    }
    if params.threads == 0 {
        bail!("thread count must be greater than zero");
    }
    if params.max_readers == Some(0) {
        bail!("max_readers must be greater than zero");
    }
    if params.device != "cpu" && params.device != "cuda" {
        bail!("unsupported sketch device {:?}", params.device);
    }
    if params.device == "cuda" && params.ksize > 32 {
        bail!(
            "CUDA sketching supports ksize up to 32 (found {})",
            params.ksize
        );
    }
    hd::validate_hv_dimension(params.hv_d)?;
    if !(3..=26).contains(&params.ull_p) {
        bail!("ULL precision must be in 3..=26 (found {})", params.ull_p);
    }
    if params.out_file.as_os_str().is_empty() {
        bail!("sketch output path must not be empty");
    }
    if params.manifest.is_some() && !params.path.as_os_str().is_empty() {
        bail!("path and manifest inputs are mutually exclusive");
    }
    if params.manifest.is_none() && params.path.as_os_str().is_empty() {
        bail!("either an input directory or manifest is required");
    }
    Ok(())
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
            .template("{wide_bar} {pos}/{len} ({percent}%) - Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .expect("progress bar template must be valid"),
    );
    pb
}

pub fn dump_sketch(file_sketch: &[FileSketch], out_file_path: &Path) -> Result<()> {
    let serialized = bincode::serialize(file_sketch).context("failed to serialize HD sketches")?;
    fs::write(out_file_path, &serialized)
        .with_context(|| format!("failed to write sketch file {}", out_file_path.display()))?;

    let sketch_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    info!(
        "Dump sketch file to {} with size {:.2} MB",
        out_file_path.display(),
        sketch_size_mb
    );
    Ok(())
}

pub fn load_sketch(path: &Path) -> Result<Vec<FileSketch>> {
    info!("Loading sketch from {}", path.display());
    let serialized =
        fs::read(path).with_context(|| format!("failed to read sketch file {}", path.display()))?;
    let sketches = bincode::deserialize::<Vec<FileSketch>>(&serialized)
        .with_context(|| format!("failed to decode sketch file {}", path.display()))?;
    validate_file_sketches(&sketches, &format!("HD sketch file {}", path.display()))?;
    Ok(sketches)
}

pub fn dump_ull_sketch(file_ull_sketch: &[FileUllSketch], out_file_path: &Path) -> Result<()> {
    dump_compressed_cardinality_sketch(file_ull_sketch, out_file_path, "ULL")
}

fn dump_compressed_cardinality_sketch<T: Serialize>(
    records: &[T],
    out_file_path: &Path,
    format: &str,
) -> Result<()> {
    let serialized = bincode::serialize(records)
        .with_context(|| format!("failed to serialize {format} sketches"))?;

    let n_threads = if serialized.len() < 8 * 1024 * 1024 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(32)
    };

    let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3)
        .with_context(|| format!("failed to create {format} zstd encoder"))?;

    if n_threads > 1 {
        encoder
            .multithread(n_threads as u32)
            .with_context(|| format!("failed to enable {format} zstd multithreading"))?;
    }

    encoder
        .write_all(serialized.as_slice())
        .with_context(|| format!("failed to write {format} bytes into zstd encoder"))?;

    let compressed = encoder
        .finish()
        .with_context(|| format!("failed to finalize {format} zstd encoding"))?;

    fs::write(out_file_path, &compressed).with_context(|| {
        format!(
            "failed to write {format} sketch file {}",
            out_file_path.display()
        )
    })?;

    let raw_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    let compressed_size_mb = compressed.len() as f32 / 1024.0 / 1024.0;
    let ratio = if serialized.is_empty() {
        1.0
    } else {
        compressed.len() as f32 / serialized.len() as f32
    };

    info!(
        "Dump compressed {format} sketch file to {} with compressed size {:.2} MB (raw {:.2} MB, ratio {:.3}, zstd threads {})",
        out_file_path.display(),
        compressed_size_mb,
        raw_size_mb,
        ratio,
        n_threads
    );
    Ok(())
}

pub fn dump_sketch_metrics(
    metrics: &[FileSketchMetrics],
    prefix: &Path,
    sketch_wall_ns: u128,
) -> Result<()> {
    let summary_path = PathBuf::from(format!("{}.summary.tsv", prefix.to_string_lossy()));
    let files_path = PathBuf::from(format!("{}.files.tsv", prefix.to_string_lossy()));

    let mut files_tsv = String::new();
    files_tsv.push_str(metrics_header());
    files_tsv.push('\n');
    for metric in metrics {
        files_tsv.push_str(&metric_row(metric));
        files_tsv.push('\n');
    }
    fs::write(&files_path, files_tsv).with_context(|| {
        format!(
            "failed to write sketch file metrics {}",
            files_path.display()
        )
    })?;

    let mut summary = FileSketchMetrics::default();
    summary.file = String::from("TOTAL");
    summary.sketch_wall_ns = Some(sketch_wall_ns);
    for metric in metrics {
        summary.input_bases += metric.input_bases;
        summary.hashes_seen += metric.hashes_seen;
        summary.unique_hashes += metric.unique_hashes;
        summary.fasta_ns += metric.fasta_ns;
        summary.fasta_wait_ns += metric.fasta_wait_ns;
        summary.hash_and_dedup_ns += metric.hash_and_dedup_ns;
        summary.hd_encode_ns += metric.hd_encode_ns;
        summary.hv_norm_ns += metric.hv_norm_ns;
        summary.hd_compress_ns += metric.hd_compress_ns;
        summary.total_worker_ns += metric.total_worker_ns;
        summary.cuda_h2d_ns = add_optional_ns(summary.cuda_h2d_ns, metric.cuda_h2d_ns);
        summary.cuda_h2d_event_ns =
            add_optional_ns(summary.cuda_h2d_event_ns, metric.cuda_h2d_event_ns);
        summary.cuda_alloc_ns = add_optional_ns(summary.cuda_alloc_ns, metric.cuda_alloc_ns);
        summary.cuda_launch_ns = add_optional_ns(summary.cuda_launch_ns, metric.cuda_launch_ns);
        summary.cuda_kernel_event_ns =
            add_optional_ns(summary.cuda_kernel_event_ns, metric.cuda_kernel_event_ns);
        summary.cuda_d2h_ns = add_optional_ns(summary.cuda_d2h_ns, metric.cuda_d2h_ns);
        summary.cuda_d2h_event_ns =
            add_optional_ns(summary.cuda_d2h_event_ns, metric.cuda_d2h_event_ns);
        summary.cuda_compact_ns = add_optional_ns(summary.cuda_compact_ns, metric.cuda_compact_ns);
        summary.cuda_filter_ns = add_optional_ns(summary.cuda_filter_ns, metric.cuda_filter_ns);
        summary.cuda_hd_hash_h2d_ns =
            add_optional_ns(summary.cuda_hd_hash_h2d_ns, metric.cuda_hd_hash_h2d_ns);
        summary.cuda_hd_hash_h2d_event_ns = add_optional_ns(
            summary.cuda_hd_hash_h2d_event_ns,
            metric.cuda_hd_hash_h2d_event_ns,
        );
        summary.cuda_hd_hv_h2d_ns =
            add_optional_ns(summary.cuda_hd_hv_h2d_ns, metric.cuda_hd_hv_h2d_ns);
        summary.cuda_hd_hv_h2d_event_ns = add_optional_ns(
            summary.cuda_hd_hv_h2d_event_ns,
            metric.cuda_hd_hv_h2d_event_ns,
        );
        summary.cuda_hd_alloc_ns =
            add_optional_ns(summary.cuda_hd_alloc_ns, metric.cuda_hd_alloc_ns);
        summary.cuda_hd_kernel_launch_ns = add_optional_ns(
            summary.cuda_hd_kernel_launch_ns,
            metric.cuda_hd_kernel_launch_ns,
        );
        summary.cuda_hd_kernel_event_ns = add_optional_ns(
            summary.cuda_hd_kernel_event_ns,
            metric.cuda_hd_kernel_event_ns,
        );
        summary.cuda_hd_d2h_ns = add_optional_ns(summary.cuda_hd_d2h_ns, metric.cuda_hd_d2h_ns);
        summary.cuda_hd_d2h_event_ns =
            add_optional_ns(summary.cuda_hd_d2h_event_ns, metric.cuda_hd_d2h_event_ns);
    }

    let mut summary_tsv = String::new();
    summary_tsv.push_str(metrics_header());
    summary_tsv.push('\n');
    summary_tsv.push_str(&metric_row(&summary));
    summary_tsv.push('\n');
    fs::write(&summary_path, summary_tsv).with_context(|| {
        format!(
            "failed to write sketch summary metrics {}",
            summary_path.display()
        )
    })?;

    info!(
        "Wrote sketch metrics to {} and {}",
        summary_path.display(),
        files_path.display()
    );
    Ok(())
}

pub(crate) fn validate_file_sketches(sketches: &[FileSketch], label: &str) -> Result<()> {
    if sketches.is_empty() {
        bail!("{label} collection is empty");
    }

    let first = &sketches[0];
    validate_file_sketch(first, &format!("{label} record 0"))?;

    for (index, sketch) in sketches.iter().enumerate().skip(1) {
        validate_file_sketch(sketch, &format!("{label} record {index}"))?;
        for (field, expected, actual) in [
            ("ksize", first.ksize as u64, sketch.ksize as u64),
            ("scaled", first.scaled, sketch.scaled),
            ("seed", first.seed, sketch.seed),
            ("hv_d", first.hv_d as u64, sketch.hv_d as u64),
        ] {
            if expected != actual {
                bail!(
                    "{label} record {index} has incompatible {field}: expected {expected}, found {actual}"
                );
            }
        }
        if first.canonical != sketch.canonical {
            bail!(
                "{label} record {index} has incompatible canonical flag: expected {}, found {}",
                first.canonical,
                sketch.canonical
            );
        }
    }

    Ok(())
}

fn validate_file_sketch(sketch: &FileSketch, label: &str) -> Result<()> {
    if sketch.file_str.is_empty() {
        bail!("{label} has an empty file identifier");
    }
    if sketch.ksize == 0 {
        bail!("{label} has invalid ksize 0");
    }
    if sketch.scaled == 0 {
        bail!("{label} has invalid scaled value 0");
    }
    hd::validate_hv_dimension(sketch.hv_d).map_err(|e| anyhow!("{label} has invalid hv_d: {e}"))?;
    if sketch.hv_norm_2 < 0 {
        bail!("{label} has negative hv_norm_2 {}", sketch.hv_norm_2);
    }

    if sketch.hv_quant_bits == 0 {
        if sketch.hv.len() != sketch.hv_d {
            bail!(
                "{label} raw vector length {} does not match hv_d {}",
                sketch.hv.len(),
                sketch.hv_d
            );
        }
        return Ok(());
    }

    if !(6..=32).contains(&sketch.hv_quant_bits) {
        bail!(
            "{label} has invalid quantization bit width {}",
            sketch.hv_quant_bits
        );
    }
    let expected_len = sketch
        .hv_d
        .checked_mul(sketch.hv_quant_bits as usize)
        .and_then(|bits| bits.checked_div(32))
        .ok_or_else(|| anyhow!("{label} compressed vector length overflows"))?;
    if sketch.hv.len() != expected_len {
        bail!(
            "{label} compressed vector length {} does not match hv_d {} and quantization bits {} (expected {})",
            sketch.hv.len(),
            sketch.hv_d,
            sketch.hv_quant_bits,
            expected_len
        );
    }

    Ok(())
}

pub(crate) fn validate_ull_sketches(sketches: &[FileUllSketch], label: &str) -> Result<()> {
    if sketches.is_empty() {
        bail!("{label} collection is empty");
    }

    let first = &sketches[0];
    validate_ull_sketch(first, &format!("{label} record 0"))?;
    for (index, sketch) in sketches.iter().enumerate().skip(1) {
        validate_ull_sketch(sketch, &format!("{label} record {index}"))?;
        if sketch.ksize != first.ksize
            || sketch.canonical != first.canonical
            || sketch.seed != first.seed
            || sketch.ull_p != first.ull_p
        {
            bail!("{label} record {index} has inconsistent ksize, canonical, seed, or ull_p");
        }
    }

    Ok(())
}

fn validate_ull_sketch(sketch: &FileUllSketch, label: &str) -> Result<()> {
    if sketch.file_str.is_empty() {
        bail!("{label} has an empty file identifier");
    }
    if sketch.ksize == 0 {
        bail!("{label} has invalid ksize 0");
    }
    if !(3..=26).contains(&sketch.ull_p) {
        bail!("{label} has invalid ULL precision {}", sketch.ull_p);
    }
    let ull = ultraloglog::UltraLogLog::wrap(sketch.ull_state.clone())
        .map_err(|e| anyhow!("{label} has invalid ULL state: {e}"))?;
    if ull.get_p() != sketch.ull_p {
        bail!(
            "{label} ULL state precision {} does not match metadata {}",
            ull.get_p(),
            sketch.ull_p
        );
    }
    Ok(())
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
    "file\tinput_bases\thashes_seen\tunique_hashes\tfasta_ns\tfasta_wait_ns\thash_and_dedup_ns\thd_encode_ns\thv_norm_ns\thd_compress_ns\ttotal_worker_ns\tsketch_wall_ns\tcuda_stream_lane\tcuda_device_id\tcuda_h2d_ns\tcuda_h2d_event_ns\tcuda_alloc_ns\tcuda_launch_ns\tcuda_kernel_event_ns\tcuda_d2h_ns\tcuda_d2h_event_ns\tcuda_compact_ns\tcuda_filter_ns\tcuda_hd_hash_h2d_ns\tcuda_hd_hash_h2d_event_ns\tcuda_hd_hv_h2d_ns\tcuda_hd_hv_h2d_event_ns\tcuda_hd_alloc_ns\tcuda_hd_kernel_launch_ns\tcuda_hd_kernel_event_ns\tcuda_hd_d2h_ns\tcuda_hd_d2h_event_ns"
}

fn metric_row(metric: &FileSketchMetrics) -> String {
    [
        metric.file.clone(),
        metric.input_bases.to_string(),
        metric.hashes_seen.to_string(),
        metric.unique_hashes.to_string(),
        metric.fasta_ns.to_string(),
        metric.fasta_wait_ns.to_string(),
        metric.hash_and_dedup_ns.to_string(),
        metric.hd_encode_ns.to_string(),
        metric.hv_norm_ns.to_string(),
        metric.hd_compress_ns.to_string(),
        metric.total_worker_ns.to_string(),
        optional_ns(metric.sketch_wall_ns),
        optional_usize(metric.cuda_stream_lane),
        optional_usize(metric.cuda_device_id),
        optional_ns(metric.cuda_h2d_ns),
        optional_ns(metric.cuda_h2d_event_ns),
        optional_ns(metric.cuda_alloc_ns),
        optional_ns(metric.cuda_launch_ns),
        optional_ns(metric.cuda_kernel_event_ns),
        optional_ns(metric.cuda_d2h_ns),
        optional_ns(metric.cuda_d2h_event_ns),
        optional_ns(metric.cuda_compact_ns),
        optional_ns(metric.cuda_filter_ns),
        optional_ns(metric.cuda_hd_hash_h2d_ns),
        optional_ns(metric.cuda_hd_hash_h2d_event_ns),
        optional_ns(metric.cuda_hd_hv_h2d_ns),
        optional_ns(metric.cuda_hd_hv_h2d_event_ns),
        optional_ns(metric.cuda_hd_alloc_ns),
        optional_ns(metric.cuda_hd_kernel_launch_ns),
        optional_ns(metric.cuda_hd_kernel_event_ns),
        optional_ns(metric.cuda_hd_d2h_ns),
        optional_ns(metric.cuda_hd_d2h_event_ns),
    ]
    .join("\t")
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

pub fn load_ull_sketch(path: &Path) -> Result<Vec<FileUllSketch>> {
    info!("Loading ULL sketch from {}", path.display());
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read ULL sketch file {}", path.display()))?;

    // New format: zstd-compressed bincode
    if let Ok(serialized) = zstd::stream::decode_all(bytes.as_slice())
        && let Ok(v) = bincode::deserialize::<Vec<FileUllSketch>>(&serialized[..])
    {
        validate_ull_sketches(&v, &format!("ULL sketch file {}", path.display()))?;
        return Ok(v);
    }

    // Backward compatibility: old raw bincode format
    if let Ok(v) = bincode::deserialize::<Vec<FileUllSketch>>(&bytes[..]) {
        warn!(
            "ULL sketch file {} is in legacy uncompressed format",
            path.display()
        );
        validate_ull_sketches(&v, &format!("ULL sketch file {}", path.display()))?;
        return Ok(v);
    }

    bail!(
        "failed to decode ULL sketch file {} as zstd-compressed or legacy uncompressed bincode",
        path.display()
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
    let mut file_sketch = load_sketch(path).expect("failed to load HD sketch");
    hd::decompress_file_sketch(&mut file_sketch).expect("failed to decompress HD sketch");

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

    fn valid_hd(file: &str) -> FileSketch {
        FileSketch {
            ksize: 3,
            scaled: 1,
            canonical: true,
            seed: 1447,
            hv_d: 256,
            hv_quant_bits: 0,
            hv_norm_2: 0,
            file_str: file.to_string(),
            hv: vec![0; 256],
        }
    }

    fn valid_ull(file: &str) -> FileUllSketch {
        let ull = ultraloglog::UltraLogLog::new(14).unwrap();
        FileUllSketch {
            ksize: 3,
            canonical: true,
            seed: 1447,
            ull_p: 14,
            file_str: file.to_string(),
            ull_state: ull.get_state().to_vec(),
        }
    }

    #[test]
    fn loaded_sketch_validation_rejects_empty_corrupt_and_inconsistent_records() {
        let dir = test_dir("loaded_validation");
        let empty_hd_path = dir.join("empty.sketch");
        fs::write(
            &empty_hd_path,
            bincode::serialize(&Vec::<FileSketch>::new()).unwrap(),
        )
        .unwrap();
        assert!(load_sketch(&empty_hd_path).is_err());

        let mut bad_length = valid_hd("bad-length");
        bad_length.hv.pop();
        let bad_length_path = dir.join("bad-length.sketch");
        fs::write(
            &bad_length_path,
            bincode::serialize(&vec![bad_length]).unwrap(),
        )
        .unwrap();
        assert!(load_sketch(&bad_length_path).is_err());

        let mut bad_quantization = valid_hd("bad-quantization");
        bad_quantization.hv_quant_bits = 5;
        bad_quantization.hv = vec![0; 40];
        let bad_quantization_path = dir.join("bad-quantization.sketch");
        fs::write(
            &bad_quantization_path,
            bincode::serialize(&vec![bad_quantization]).unwrap(),
        )
        .unwrap();
        assert!(load_sketch(&bad_quantization_path).is_err());

        let mut inconsistent = valid_hd("second");
        inconsistent.canonical = false;
        let inconsistent_path = dir.join("inconsistent.sketch");
        fs::write(
            &inconsistent_path,
            bincode::serialize(&vec![valid_hd("first"), inconsistent]).unwrap(),
        )
        .unwrap();
        assert!(load_sketch(&inconsistent_path).is_err());

        let empty_ull_path = dir.join("empty.ull");
        fs::write(
            &empty_ull_path,
            bincode::serialize(&Vec::<FileUllSketch>::new()).unwrap(),
        )
        .unwrap();
        assert!(load_ull_sketch(&empty_ull_path).is_err());

        let mut bad_state = valid_ull("bad-state");
        bad_state.ull_state = vec![0];
        let bad_state_path = dir.join("bad-state.ull");
        fs::write(
            &bad_state_path,
            bincode::serialize(&vec![bad_state]).unwrap(),
        )
        .unwrap();
        assert!(load_ull_sketch(&bad_state_path).is_err());

        let mut inconsistent_ull = valid_ull("second");
        inconsistent_ull.ull_p = 15;
        inconsistent_ull.ull_state = ultraloglog::UltraLogLog::new(15)
            .unwrap()
            .get_state()
            .to_vec();
        let inconsistent_ull_path = dir.join("inconsistent.ull");
        fs::write(
            &inconsistent_ull_path,
            bincode::serialize(&vec![valid_ull("first"), inconsistent_ull]).unwrap(),
        )
        .unwrap();
        assert!(load_ull_sketch(&inconsistent_ull_path).is_err());

        fs::remove_dir_all(dir).unwrap();
    }
}
