use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use chrono::Local;
use clap::{Arg, ArgAction, Command, value_parser};
use env_logger::{Builder, Env, Target};

#[cfg(feature = "cuda")]
use dotani::sketch_cuda;
use dotani::{dist, params, sketch, types};

fn init_log() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d-%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .parse_env(Env::default().default_filter_or("info"))
        .target(Target::Stdout)
        .init();
}

fn cardinality_path_from_sketch_path(p: &Path, estimator: types::CardinalityEstimator) -> PathBuf {
    PathBuf::from(format!("{}.{}", p.to_string_lossy(), estimator.as_str()))
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn main() {
    init_log();
    println!("\n ************** initializing logger *****************\n");
    log::info!("\nLogger initialized\n");

    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{value:?} is not a valid positive integer"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn parse_ksize(value: &str) -> Result<u8, String> {
    let parsed = value
        .parse::<u8>()
        .map_err(|_| format!("{value:?} is not a valid k-mer size"))?;
    if parsed == 0 {
        return Err("ksize must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn parse_ull_p(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{value:?} is not a valid ULL precision"))?;
    if !(3..=26).contains(&parsed) {
        return Err("ULL precision must be in 3..=26".to_string());
    }
    Ok(parsed)
}

fn parse_hv_d(value: &str) -> Result<usize, String> {
    let parsed = parse_positive_usize(value)?;
    if parsed % 256 != 0 {
        return Err("hv_d must be divisible by 256".to_string());
    }
    Ok(parsed)
}

fn parse_ani_threshold(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{value:?} is not a valid ANI threshold"))?;
    if !parsed.is_finite() || !(0.0..=100.0).contains(&parsed) {
        return Err("ANI threshold must be finite and in 0..=100".to_string());
    }
    Ok(parsed)
}

fn run() -> Result<()> {
    let sketch_cmd = Command::new(params::CMD_SKETCH)
        .version("0.3.0")
        .about("Sketch genome FASTA files into DotHash and ULL or ELL sketches")
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .help("Input folder path containing .fna/.fa/.fasta files (gzip/bzip2/xz/zstd compressed files supported, e.g., .fna.gz, .fa.bz2, .fasta.xz, .fna.zst)")
                .required_unless_present("manifest")
                .conflicts_with("manifest")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("manifest")
                .long("manifest")
                .help("Input TSV manifest with read_path and file_id columns; row order is preserved; relative read_path values resolve against the current working directory")
                .required_unless_present("path")
                .conflicts_with("path")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .help("Output DotHash sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('T')
                .help("Number of threads, default all logical cores")
                .value_parser(parse_positive_usize)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("canonical")
                .short('C')
                .long("canonical")
                .help("Whether to use canonical k-mers")
                .default_value("true")
                .value_parser(value_parser!(bool))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ksize")
                .short('k')
                .long("ksize")
                .help("k-mer size for sketching")
                .default_value("16")
                .value_parser(parse_ksize)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .short('S')
                .long("seed")
                .help("Hash seed")
                .default_value("1447")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ull")
                .long("ull")
                .help("Use UltraLogLog cardinality estimation (default)")
                .conflicts_with("ell")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ell")
                .long("ell")
                .help("Use ExaLogLog cardinality estimation")
                .conflicts_with("ull")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ull_p")
                .long("ull-p")
                .help("UltraLogLog precision parameter")
                .default_value("14")
                .conflicts_with("ell")
                .value_parser(parse_ull_p)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ell_t")
                .long("ell-t")
                .help("ExaLogLog t parameter (default: 2)")
                .requires("ell")
                .value_parser(value_parser!(u32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ell_d")
                .long("ell-d")
                .help("ExaLogLog d parameter (default: 24)")
                .requires("ell")
                .value_parser(value_parser!(u32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ell_p")
                .long("ell-p")
                .help("ExaLogLog precision parameter (default: 12)")
                .requires("ell")
                .value_parser(value_parser!(u32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("hv_d")
                .short('d')
                .long("hv-d")
                .help("Dimension for hypervector")
                .default_value("4096")
                .value_parser(parse_hv_d)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("quant_scale")
                .short('Q')
                .long("quant-scale")
                .help("Scaling factor for HV quantization")
                .default_value("1.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("device")
                .long("device")
                .help("Sketch execution device. CUDA builds default to cuda; CPU-only builds default to cpu.")
                .value_parser(["cpu", "cuda"])
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("cuda_dedup")
                .long("cuda-dedup")
                .help("CUDA host-side dedup strategy")
                .default_value("sort_unstable")
                .value_parser(["hashset", "sort_unstable"])
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("metrics_out")
                .long("metrics-out")
                .help("Write sketch metrics to <prefix>.summary.tsv and <prefix>.files.tsv")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("max_readers")
                .long("max-readers")
                .help("CUDA only: maximum concurrent FASTA read/decompress/merge workers")
                .value_parser(parse_positive_usize)
                .action(ArgAction::Set),
        );

    let dist_cmd = Command::new(params::CMD_DIST)
        .about("Estimate ANI from reference and query sketch files")
        .arg(
            Arg::new("ull")
                .long("ull")
                .help("Load UltraLogLog cardinality sidecars (default)")
                .conflicts_with("ell")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ell")
                .long("ell")
                .help("Load ExaLogLog cardinality sidecars")
                .conflicts_with("ull")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("path_r")
                .short('r')
                .long("path-r")
                .help("Path to reference DotHash sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("path_q")
                .short('q')
                .long("path-q")
                .help("Path to query DotHash sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .help("Output ANI results file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('T')
                .help("Number of threads, default all logical cores; distance postprocessing uses one worker at -T1 and at most 128 workers")
                .value_parser(parse_positive_usize)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ani_th")
                .short('a')
                .long("ani-th")
                .help("ANI threshold")
                .default_value("85.0")
                .value_parser(parse_ani_threshold)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("output_mode")
                .long("output-mode")
                .help("Dist output mode")
                .default_value("rows")
                .value_parser(["rows", "count"])
                .action(ArgAction::Set),
        );

    let matches = Command::new("dotani")
        .version(params::VERSION)
        .about("DotANI: Ultra-fast and memory-efficient ANI estimation in hyperdimensional space via DotHash and ULL or ELL, with GPU acceleration")
        .arg_required_else_help(true)
        .subcommand_required(true)
        .subcommand(sketch_cmd)
        .subcommand(dist_cmd)
        .get_matches();

    if let Some(sketch_m) = matches.subcommand_matches(params::CMD_SKETCH) {
        let out_file = sketch_m
            .get_one::<PathBuf>("out")
            .cloned()
            .expect("clap guarantees --out is required");
        let threads = sketch_m
            .get_one::<usize>("threads")
            .copied()
            .unwrap_or_else(default_threads);
        let device = sketch_m
            .get_one::<String>("device")
            .cloned()
            .unwrap_or_else(default_sketch_device);
        let max_readers = sketch_m.get_one::<usize>("max_readers").copied();
        let cardinality_estimator = if sketch_m.get_flag("ell") {
            types::CardinalityEstimator::Ell
        } else {
            types::CardinalityEstimator::Ull
        };

        if device == "cuda" && !cuda_enabled() {
            bail!("--device cuda requires a binary built with --features cuda");
        }
        if device == "cpu" && max_readers.is_some() {
            bail!("--max-readers is only supported with --device cuda");
        }

        let cli_params = types::CliParams {
            mode: params::CMD_SKETCH.to_string(),
            path: sketch_m
                .get_one::<PathBuf>("path")
                .cloned()
                .unwrap_or_default(),
            manifest: sketch_m.get_one::<PathBuf>("manifest").cloned(),
            path_ref_sketch: PathBuf::new(),
            path_query_sketch: PathBuf::new(),
            out_file: out_file.clone(),
            ksize: *sketch_m
                .get_one::<u8>("ksize")
                .expect("clap guarantees ksize has default"),
            sketch_method: String::from("t1ha2"),
            canonical: *sketch_m
                .get_one::<bool>("canonical")
                .expect("clap guarantees canonical has default"),
            seed: *sketch_m
                .get_one::<u64>("seed")
                .expect("clap guarantees seed has default"),
            scaled: 1u64,
            hv_d: *sketch_m
                .get_one::<usize>("hv_d")
                .expect("clap guarantees hv-d has default"),
            hv_quant_scale: *sketch_m
                .get_one::<f32>("quant_scale")
                .expect("clap guarantees quant-scale has default"),
            ani_threshold: 0.0,
            if_compressed: true,
            threads,
            cuda_dedup_strategy: types::CudaDedupStrategy::from_cli_value(
                sketch_m
                    .get_one::<String>("cuda_dedup")
                    .expect("clap guarantees cuda-dedup has default"),
            ),
            max_readers,
            device,
            cardinality_estimator,
            ull_p: *sketch_m
                .get_one::<u32>("ull_p")
                .expect("clap guarantees ull-p has default"),
            ell_t: sketch_m.get_one::<u32>("ell_t").copied().unwrap_or(2),
            ell_d: sketch_m.get_one::<u32>("ell_d").copied().unwrap_or(24),
            ell_p: sketch_m.get_one::<u32>("ell_p").copied().unwrap_or(12),
            cardinality_out_file: cardinality_path_from_sketch_path(
                &out_file,
                cardinality_estimator,
            ),
            path_ref_cardinality: PathBuf::new(),
            path_query_cardinality: PathBuf::new(),
            metrics_out: sketch_m.get_one::<PathBuf>("metrics_out").cloned(),
            dist_output_mode: types::DistOutputMode::Rows,
        };

        let sketch_params = types::SketchParams::new(&cli_params);

        let sketch_result = if sketch_params.device == "cuda" {
            #[cfg(feature = "cuda")]
            {
                sketch_cuda::sketch_cuda(sketch_params)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!(
                    "--device cuda requires a binary built with --features cuda"
                ));
            }
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(cli_params.threads)
                .build_global()
                .map_err(|e| anyhow!("failed to configure Rayon thread pool: {e}"))?;

            sketch::sketch(sketch_params)
        };

        sketch_result?;
    } else if let Some(dist_m) = matches.subcommand_matches(params::CMD_DIST) {
        let path_ref_sketch = dist_m
            .get_one::<PathBuf>("path_r")
            .cloned()
            .expect("clap guarantees --path-r is required");
        let path_query_sketch = dist_m
            .get_one::<PathBuf>("path_q")
            .cloned()
            .expect("clap guarantees --path-q is required");
        let cardinality_estimator = if dist_m.get_flag("ell") {
            types::CardinalityEstimator::Ell
        } else {
            types::CardinalityEstimator::Ull
        };
        let threads = dist_m
            .get_one::<usize>("threads")
            .copied()
            .unwrap_or_else(default_threads);

        let cli_params = types::CliParams {
            mode: params::CMD_DIST.to_string(),
            path: PathBuf::new(),
            manifest: None,
            path_ref_sketch: path_ref_sketch.clone(),
            path_query_sketch: path_query_sketch.clone(),
            out_file: dist_m
                .get_one::<PathBuf>("out")
                .cloned()
                .expect("clap guarantees --out is required"),
            ksize: 0,
            sketch_method: String::new(),
            canonical: true,
            seed: 0,
            scaled: 1u64,
            hv_d: 0,
            hv_quant_scale: 1.0,
            ani_threshold: *dist_m
                .get_one::<f32>("ani_th")
                .expect("clap guarantees ani-th has default"),
            if_compressed: true,
            threads,
            cuda_dedup_strategy: types::CudaDedupStrategy::HashSet,
            max_readers: None,
            device: String::from("cpu"),
            cardinality_estimator,
            ull_p: 0,
            ell_t: 0,
            ell_d: 0,
            ell_p: 0,
            cardinality_out_file: PathBuf::new(),
            path_ref_cardinality: cardinality_path_from_sketch_path(
                &path_ref_sketch,
                cardinality_estimator,
            ),
            path_query_cardinality: cardinality_path_from_sketch_path(
                &path_query_sketch,
                cardinality_estimator,
            ),
            metrics_out: None,
            dist_output_mode: types::DistOutputMode::from_cli_value(
                dist_m
                    .get_one::<String>("output_mode")
                    .expect("clap guarantees output-mode has default"),
            ),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads)
            .build_global()
            .map_err(|e| anyhow!("failed to configure Rayon thread pool: {e}"))?;

        let mut sketch_dist = types::SketchDist::new(&cli_params);
        dist::dist(&mut sketch_dist)?;
    } else {
        bail!("no supported subcommand was selected");
    }

    Ok(())
}

fn default_sketch_device() -> String {
    if cuda_enabled() {
        String::from("cuda")
    } else {
        String::from("cpu")
    }
}

fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
