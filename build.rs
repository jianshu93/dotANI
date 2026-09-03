use std::{env, path::PathBuf, process::Command};

use bindgen::CargoCallbacks;
use regex::Regex;

fn main() {
    if !cfg!(feature = "cuda") {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernel.h");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let cuda_src = PathBuf::from("src/cuda_kernel.cu");
    let ptx_file = out_dir.join("cuda_kmer_hash.ptx");

    // PTX is generally forward-compatible, so choose the lowest modern arch
    // you want to support. Default: Turing+.
    //
    // Examples:
    //   CUDA_ARCH=75  -> Turing+
    //   CUDA_ARCH=80  -> Ampere+
    //   CUDA_ARCH=86  -> RTX 30xx / Ampere consumer
    //   CUDA_ARCH=89  -> Ada
    //   CUDA_ARCH=90  -> Hopper
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "75".to_string());
    let arch_flag = format!("-arch=compute_{cuda_arch}");

    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cuda_src)
        .arg(&arch_flag)
        .status()
        .expect("Failed to run nvcc");

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX with nvcc."
    );

    let bindings = bindgen::Builder::default()
        .header("src/cuda_kernel.h")
        .parse_callbacks(Box::new(CargoCallbacks))
        .no_copy("*")
        .no_debug("*")
        .generate()
        .expect("Unable to generate bindings");

    let generated_bindings = bindings.to_string();

    let pointer_regex = Regex::new(r"\*mut f32").expect("Failed to compile regex");
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    std::fs::write(out_dir.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}
