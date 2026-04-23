use std::io::Write;
use std::path::PathBuf;

use chrono::Local;
use clap::{value_parser, Arg, ArgAction, Command};
use env_logger::{Builder, Target};
use log::LevelFilter;

use dotani::{dist, params, sketch, sketch_cuda, types};

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
        .filter(None, LevelFilter::Info)
        .target(Target::Stdout)
        .init();
}

fn ull_path_from_sketch_path(p: &PathBuf) -> PathBuf {
    PathBuf::from(format!("{}.ull", p.to_string_lossy()))
}

fn default_threads_u8() -> u8 {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(u8::MAX as usize) as u8
}

fn main() {
    init_log();
    println!("\n ************** initializing logger *****************\n");
    log::info!("\nLogger initialized\n");

    let sketch_cmd = Command::new(params::CMD_SKETCH)
        .version("0.3.0")
        .about("Sketch genome FASTA files into DotHash and UltraLogLog sketches")
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .help("Input folder path containing .fna/.fa/.fasta files (gzip/bzip2/xz/zstd compressed files supported, e.g., .fna.gz, .fa.bz2, .fasta.xz, .fna.zst)")
                .required(true)
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
                .value_parser(value_parser!(usize))
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
                .value_parser(value_parser!(u8))
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
            Arg::new("ull_p")
                .long("ull-p")
                .help("UltraLogLog precision parameter")
                .default_value("14")
                .value_parser(value_parser!(u32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("hv_d")
                .short('d')
                .long("hv-d")
                .help("Dimension for hypervector")
                .default_value("4096")
                .value_parser(value_parser!(usize))
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
        );

    let dist_cmd = Command::new(params::CMD_DIST)
        .about("Estimate ANI from reference and query sketch files")
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
                .help("Number of threads, default all logical cores")
                .value_parser(value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ani_th")
                .short('a')
                .long("ani-th")
                .help("ANI threshold")
                .default_value("85.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        );

    let matches = Command::new("dotani")
        .version(params::VERSION)
        .about("DotANI: Ultra-fast and memory-efficient ANI estimation in hyperdimensional space via DotHash and UltraLogLog, with GPU acceleration")
        .arg_required_else_help(true)
        .subcommand_required(true)
        .subcommand(sketch_cmd)
        .subcommand(dist_cmd)
        .subcommand(search_cmd)
        .get_matches();

    if let Some(sketch_m) = matches.subcommand_matches(params::CMD_SKETCH) {
        let out_file = sketch_m.get_one::<PathBuf>("out").cloned().unwrap();
        let threads = sketch_m
            .get_one::<usize>("threads")
            .copied()
            .unwrap_or_else(|| default_threads_u8() as usize)
            .min(u8::MAX as usize) as u8;

        let cli_params = types::CliParams {
            mode: params::CMD_SKETCH.to_string(),
            path: sketch_m.get_one::<PathBuf>("path").cloned().unwrap(),
            path_ref_sketch: PathBuf::new(),
            path_query_sketch: PathBuf::new(),
            out_file: out_file.clone(),
            ksize: *sketch_m.get_one::<u8>("ksize").unwrap(),
            sketch_method: String::from("t1ha2"),
            canonical: *sketch_m.get_one::<bool>("canonical").unwrap(),
            seed: *sketch_m.get_one::<u64>("seed").unwrap(),
            scaled: 1u64,
            hv_d: *sketch_m.get_one::<usize>("hv_d").unwrap(),
            hv_quant_scale: *sketch_m.get_one::<f32>("quant_scale").unwrap(),
            ani_threshold: 0.0,
            if_compressed: true,
            threads,
            device: String::new(),
            if_ull: true,
            ull_p: *sketch_m.get_one::<u32>("ull_p").unwrap(),
            ull_out_file: ull_path_from_sketch_path(&out_file),
            path_ref_ull: PathBuf::new(),
            path_query_ull: PathBuf::new(),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads as usize)
            .build_global()
            .unwrap();

        let sketch_params = types::SketchParams::new(&cli_params);

        #[cfg(feature = "cuda")]
        {
            sketch_cuda::sketch_cuda(sketch_params);
        }

        #[cfg(not(feature = "cuda"))]
        {
            sketch::sketch(sketch_params);
        }
            } else if let Some(dist_m) = matches.subcommand_matches(params::CMD_DIST) {
        let path_ref_sketch = dist_m.get_one::<PathBuf>("path_r").cloned().unwrap();
        let path_query_sketch = dist_m.get_one::<PathBuf>("path_q").cloned().unwrap();
        let threads = dist_m
            .get_one::<usize>("threads")
            .copied()
            .unwrap_or_else(|| default_threads_u8() as usize)
            .min(u8::MAX as usize) as u8;

        let cli_params = types::CliParams {
            mode: params::CMD_DIST.to_string(),
            path: PathBuf::new(),
            path_ref_sketch: path_ref_sketch.clone(),
            path_query_sketch: path_query_sketch.clone(),
            out_file: dist_m.get_one::<PathBuf>("out").cloned().unwrap(),
            ksize: 0,
            sketch_method: String::new(),
            canonical: true,
            seed: 0,
            scaled: 1u64,
            hv_d: 0,
            hv_quant_scale: 1.0,
            ani_threshold: *dist_m.get_one::<f32>("ani_th").unwrap(),
            if_compressed: true,
            threads,
            device: String::from("cpu"),
            if_ull: true,
            ull_p: 0,
            ull_out_file: PathBuf::new(),
            path_ref_ull: ull_path_from_sketch_path(&path_ref_sketch),
            path_query_ull: ull_path_from_sketch_path(&path_query_sketch),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads as usize)
            .build_global()
            .unwrap();

        let mut sketch_dist = types::SketchDist::new(&cli_params);
        dist::dist(&mut sketch_dist);
    } else {
        unreachable!("clap should ensure we don't get here");
    }
}