use {
    crate::hd,
    anyhow::{Result, bail},
    cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg},
    std::sync::Arc,
    std::time::Instant,
};

pub(crate) const HASH_TILE: usize = 256;
pub(crate) const MAX_KERNEL_WARPS: usize = 8;

const _: () = assert!(
    HASH_TILE.is_multiple_of(32) && HASH_TILE / 32 <= MAX_KERNEL_WARPS,
    "HASH_TILE must be a whole number of warps and within CUDA shared-memory layout"
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuHdEncodeMetrics {
    pub cuda_hd_alloc_ns: u128,
    pub cuda_hd_hash_h2d_ns: u128,
    pub cuda_hd_hash_h2d_event_ns: u128,
    pub cuda_hd_hv_h2d_ns: u128,
    pub cuda_hd_hv_h2d_event_ns: u128,
    pub cuda_hd_kernel_launch_ns: u128,
    pub cuda_hd_kernel_event_ns: u128,
    pub cuda_hd_d2h_ns: u128,
    pub cuda_hd_d2h_event_ns: u128,
}

pub fn encode_hash_hd_cuda(
    hashes: &[u64],
    hv_d: usize,
    ctx: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
) -> Result<(Vec<i32>, GpuHdEncodeMetrics)> {
    hd::validate_hv_dimension(hv_d)?;
    if hashes.len() > i32::MAX as usize {
        bail!("too many hashes for i32 HD count vector: {}", hashes.len());
    }
    if hashes.is_empty() {
        return Ok((vec![0; hv_d], GpuHdEncodeMetrics::default()));
    }

    let num_chunks = hv_d / 64;
    let mut metrics = GpuHdEncodeMetrics::default();
    let mut hv_host = vec![-(hashes.len() as i32); hv_d];

    if num_chunks == 0 {
        return Ok((hv_host, metrics));
    }

    let stream = ctx.default_stream();

    let alloc_start = Instant::now();
    let mut d_hv = unsafe { stream.alloc::<i32>(hv_d)? };
    metrics.cuda_hd_alloc_ns = alloc_start.elapsed().as_nanos();

    let hash_h2d_start = Instant::now();
    let d_hashes = stream.clone_htod(hashes)?;
    metrics.cuda_hd_hash_h2d_ns = hash_h2d_start.elapsed().as_nanos();

    let hv_h2d_start = Instant::now();
    stream.memcpy_htod(&hv_host, &mut d_hv)?;
    metrics.cuda_hd_hv_h2d_ns = hv_h2d_start.elapsed().as_nanos();

    let function = module.load_function("cuda_hd_encode_counts_direct")?;
    let num_hashes = hashes.len() as i32;
    let hv_d_i32 = hv_d as i32;
    let cfg = LaunchConfig {
        grid_dim: (
            num_chunks as u32,
            hashes.len().div_ceil(HASH_TILE) as u32,
            1,
        ),
        block_dim: (HASH_TILE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&function);
    launch.arg(&d_hashes);
    launch.arg(&num_hashes);
    launch.arg(&hv_d_i32);
    launch.arg(&mut d_hv);

    let kernel_launch_start = Instant::now();
    unsafe {
        launch.launch(cfg)?;
    }
    metrics.cuda_hd_kernel_launch_ns = kernel_launch_start.elapsed().as_nanos();

    let d2h_start = Instant::now();
    hv_host = stream.clone_dtoh(&d_hv)?;
    metrics.cuda_hd_d2h_ns = d2h_start.elapsed().as_nanos();

    Ok((hv_host, metrics))
}
