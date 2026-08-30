use anyhow::{Context, Result, bail};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaModule, CudaSlice, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#ifndef BLOCK_M
#define BLOCK_M 64
#endif

#ifndef BLOCK_N
#define BLOCK_N 32
#endif

#ifndef BLOCK_K
#define BLOCK_K 32
#endif

#ifndef THREAD_M
#define THREAD_M 4
#endif

#ifndef THREAD_N
#define THREAD_N 2
#endif

#ifndef PAD_A
#define PAD_A 1
#endif

#ifndef PAD_B
#define PAD_B 1
#endif

struct __align__(16) Int4 {
    int x, y, z, w;
};

extern "C" __global__
void dot_rect_i32_i64_tiled_rb(
    const int* __restrict__ query_hv,   // [nq * d]
    const int* __restrict__ ref_hv,     // [nr * d]
    int nq,
    int nr,
    int d,
    long long* __restrict__ out         // [nq * nr], row-major
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int local_row0 = ty * THREAD_M;
    const int local_col0 = tx * THREAD_N;

    const int global_row0 = block_row + local_row0;
    const int global_col0 = block_col + local_col0;

    const int STRIDE_A = BLOCK_K + PAD_A;
    const int STRIDE_B = BLOCK_K + PAD_B;

    extern __shared__ int smem[];
    int* As = smem;
    int* Bs = As + (BLOCK_M * STRIDE_A);

    long long acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0;
        }
    }

    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < d; k0 += BLOCK_K) {
        const int bk = ((BLOCK_K < (d - k0)) ? BLOCK_K : (d - k0));

        // Load A tile
        const int vec_bk = bk / 4;
        const int total_a_vec = BLOCK_M * vec_bk;

        for (int idx = tid; idx < total_a_vec; idx += nthreads) {
            const int row = idx / vec_bk;
            const int kk4 = (idx - row * vec_bk) * 4;

            const int g_row = block_row + row;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;

            if (g_row < nq) {
                const Int4* gptr = (const Int4*)(query_hv + (size_t)g_row * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = As + row * STRIDE_A + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_a = vec_bk * 4;
        if (bk > kk_tail_a) {
            const int tail_cols = bk - kk_tail_a;
            const int total_a_tail = BLOCK_M * tail_cols;
            for (int idx = tid; idx < total_a_tail; idx += nthreads) {
                const int row = idx / tail_cols;
                const int kk  = kk_tail_a + (idx - row * tail_cols);

                const int g_row = block_row + row;
                int v = 0;
                if (g_row < nq) {
                    v = query_hv[(size_t)g_row * (size_t)d + (size_t)(k0 + kk)];
                }
                As[row * STRIDE_A + kk] = v;
            }
        }

        // Load B tile
        const int total_b_vec = BLOCK_N * vec_bk;

        for (int idx = tid; idx < total_b_vec; idx += nthreads) {
            const int col = idx / vec_bk;
            const int kk4 = (idx - col * vec_bk) * 4;

            const int g_col = block_col + col;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;

            if (g_col < nr) {
                const Int4* gptr = (const Int4*)(ref_hv + (size_t)g_col * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = Bs + col * STRIDE_B + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_b = vec_bk * 4;
        if (bk > kk_tail_b) {
            const int tail_cols = bk - kk_tail_b;
            const int total_b_tail = BLOCK_N * tail_cols;
            for (int idx = tid; idx < total_b_tail; idx += nthreads) {
                const int col = idx / tail_cols;
                const int kk  = kk_tail_b + (idx - col * tail_cols);

                const int g_col = block_col + col;
                int v = 0;
                if (g_col < nr) {
                    v = ref_hv[(size_t)g_col * (size_t)d + (size_t)(k0 + kk)];
                }
                Bs[col * STRIDE_B + kk] = v;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            if (kk >= bk) break;

            int a_frag[THREAD_M];
            int b_frag[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                a_frag[i] = As[(local_row0 + i) * STRIDE_A + kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                b_frag[j] = Bs[(local_col0 + j) * STRIDE_B + kk];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += (long long)a_frag[i] * (long long)b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int g_row = global_row0 + i;
        if (g_row >= nq) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int g_col = global_col0 + j;
            if (g_col >= nr) continue;

            out[(size_t)g_row * (size_t)nr + (size_t)g_col] = acc[i][j];
        }
    }
}

extern "C" __global__
void dot_rect_count_symmetric_resident(
    const int* __restrict__ query_hv,
    const int* __restrict__ ref_hv,
    const double* __restrict__ query_cards,
    const double* __restrict__ ref_cards,
    int q0,
    int r0,
    int nq,
    int nr,
    int d,
    int ksize,
    float ani_threshold,
    float transformed_threshold_low,
    float transformed_threshold_high,
    unsigned int borderline_capacity,
    long long* __restrict__ borderline_dots,
    unsigned int* __restrict__ borderline_refs,
    unsigned int* __restrict__ borderline_queries,
    unsigned long long* __restrict__ stats
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int local_row0 = ty * THREAD_M;
    const int local_col0 = tx * THREAD_N;

    const int global_row0 = block_row + local_row0;
    const int global_col0 = block_col + local_col0;

    const int STRIDE_A = BLOCK_K + PAD_A;
    const int STRIDE_B = BLOCK_K + PAD_B;

    extern __shared__ int smem[];
    int* As = smem;
    int* Bs = As + (BLOCK_M * STRIDE_A);

    long long acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0;
        }
    }

    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < d; k0 += BLOCK_K) {
        const int bk = ((BLOCK_K < (d - k0)) ? BLOCK_K : (d - k0));

        const int vec_bk = bk / 4;
        const int total_a_vec = BLOCK_M * vec_bk;

        for (int idx = tid; idx < total_a_vec; idx += nthreads) {
            const int row = idx / vec_bk;
            const int kk4 = (idx - row * vec_bk) * 4;
            const int g_row = block_row + row;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;
            if (g_row < nq) {
                const Int4* gptr = (const Int4*)(query_hv + (size_t)g_row * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = As + row * STRIDE_A + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_a = vec_bk * 4;
        if (bk > kk_tail_a) {
            const int tail_cols = bk - kk_tail_a;
            const int total_a_tail = BLOCK_M * tail_cols;
            for (int idx = tid; idx < total_a_tail; idx += nthreads) {
                const int row = idx / tail_cols;
                const int kk = kk_tail_a + (idx - row * tail_cols);
                const int g_row = block_row + row;
                int v = 0;
                if (g_row < nq) {
                    v = query_hv[(size_t)g_row * (size_t)d + (size_t)(k0 + kk)];
                }
                As[row * STRIDE_A + kk] = v;
            }
        }

        const int total_b_vec = BLOCK_N * vec_bk;

        for (int idx = tid; idx < total_b_vec; idx += nthreads) {
            const int col = idx / vec_bk;
            const int kk4 = (idx - col * vec_bk) * 4;
            const int g_col = block_col + col;

            Int4 v;
            v.x = 0; v.y = 0; v.z = 0; v.w = 0;
            if (g_col < nr) {
                const Int4* gptr = (const Int4*)(ref_hv + (size_t)g_col * (size_t)d + (size_t)(k0 + kk4));
                v = *gptr;
            }

            int* sptr = Bs + col * STRIDE_B + kk4;
            sptr[0] = v.x;
            sptr[1] = v.y;
            sptr[2] = v.z;
            sptr[3] = v.w;
        }

        const int kk_tail_b = vec_bk * 4;
        if (bk > kk_tail_b) {
            const int tail_cols = bk - kk_tail_b;
            const int total_b_tail = BLOCK_N * tail_cols;
            for (int idx = tid; idx < total_b_tail; idx += nthreads) {
                const int col = idx / tail_cols;
                const int kk = kk_tail_b + (idx - col * tail_cols);
                const int g_col = block_col + col;
                int v = 0;
                if (g_col < nr) {
                    v = ref_hv[(size_t)g_col * (size_t)d + (size_t)(k0 + kk)];
                }
                Bs[col * STRIDE_B + kk] = v;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            if (kk >= bk) break;

            int a_frag[THREAD_M];
            int b_frag[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                a_frag[i] = As[(local_row0 + i) * STRIDE_A + kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                b_frag[j] = Bs[(local_col0 + j) * STRIDE_B + kk];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += (long long)a_frag[i] * (long long)b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    unsigned long long local_pairs = 0;
    unsigned long long local_hits = 0;
    unsigned long long local_ani_evals = 0;
    unsigned long long local_nonpositive = 0;

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int g_row = global_row0 + i;
        if (g_row >= nq) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int g_col = global_col0 + j;
            if (g_col >= nr) continue;

            const int global_ref = r0 + g_col;
            const int global_query = q0 + g_row;
            if (global_ref >= global_query) continue;

            local_pairs += 1;
            const double inter_hat = (double)acc[i][j] / (double)d;
            if (inter_hat <= 0.0 && ani_threshold > 0.0f) {
                local_nonpositive += 1;
                continue;
            }

            local_ani_evals += 1;
            bool hit = false;
            bool borderline = false;
            const double union_hat = ref_cards[g_col] + query_cards[g_row] - inter_hat;
            if (inter_hat > 0.0 && union_hat > 0.0) {
                const double jaccard = inter_hat / union_hat;
                if (isfinite(jaccard) && jaccard > 0.0) {
                    if (jaccard > 1.0) {
                        hit = ani_threshold <= 100.0f;
                    } else {
                        const float jf = (float)jaccard;
                        const float transformed = (2.0f * jf) / (1.0f + jf);
                        if (transformed >= transformed_threshold_high) {
                            hit = true;
                        } else if (transformed >= transformed_threshold_low) {
                            borderline = true;
                        }
                    }
                }
            }

            if (ani_threshold <= 0.0f || hit) {
                local_hits += 1;
            } else if (borderline) {
                const unsigned long long border_idx = atomicAdd(&stats[4], 1ULL);
                if (border_idx < (unsigned long long)borderline_capacity) {
                    borderline_dots[border_idx] = acc[i][j];
                    borderline_refs[border_idx] = (unsigned int)global_ref;
                    borderline_queries[border_idx] = (unsigned int)global_query;
                }
            }
        }
    }

    __shared__ unsigned long long block_stats[4 * 256];
    block_stats[tid] = local_pairs;
    block_stats[256 + tid] = local_hits;
    block_stats[512 + tid] = local_ani_evals;
    block_stats[768 + tid] = local_nonpositive;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_stats[tid] += block_stats[tid + stride];
            block_stats[256 + tid] += block_stats[256 + tid + stride];
            block_stats[512 + tid] += block_stats[512 + tid + stride];
            block_stats[768 + tid] += block_stats[768 + tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&stats[0], block_stats[0]);
        atomicAdd(&stats[1], block_stats[256]);
        atomicAdd(&stats[2], block_stats[512]);
        atomicAdd(&stats[3], block_stats[768]);
    }
}
"#;

pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

#[inline]
fn mib(x: usize) -> f64 {
    x as f64 / (1024.0 * 1024.0)
}

#[inline]
fn shared_mem_bytes_i32(block_m: usize, block_n: usize, block_k: usize) -> u32 {
    let stride_a = block_k + 1;
    let stride_b = block_k + 1;
    ((block_m * stride_a + block_n * stride_b) * std::mem::size_of::<i32>()) as u32
}

pub struct GpuDotExecutor {
    gpu_id: usize,
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,

    d_query: Option<CudaSlice<i32>>,
    d_ref: Option<CudaSlice<i32>>,
    d_out: Option<CudaSlice<i64>>,
    d_count_stats: Option<CudaSlice<u64>>,
    d_count_borderline_dots: Option<CudaSlice<i64>>,
    d_count_borderline_refs: Option<CudaSlice<u32>>,
    d_count_borderline_queries: Option<CudaSlice<u32>>,
    kernel_start: CudaEvent,
    kernel_end: CudaEvent,
    d2h_start: CudaEvent,
    d2h_end: CudaEvent,

    cap_query: usize,
    cap_ref: usize,
    cap_out: usize,
    cap_count_borderline: usize,
}

pub struct GpuResidentMatrix {
    d_hv: CudaSlice<i32>,
    rows: usize,
    hv_d: usize,
}

pub struct GpuResidentCards {
    d_cards: CudaSlice<f64>,
    rows: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GpuTileTimings {
    pub query_h2d_ns: u128,
    pub ref_h2d_ns: u128,
    pub compute_d2h_ns: u128,
    pub kernel_event_ns: u128,
    pub d2h_event_ns: u128,
    pub total_ns: u128,
    pub query_h2d_bytes: usize,
    pub ref_h2d_bytes: usize,
    pub out_d2h_bytes: usize,
    pub ref_upload_performed: bool,
}

#[derive(Debug, Default, Clone)]
pub struct GpuCountTileResult {
    pub timings: GpuTileTimings,
    pub pairs: usize,
    pub hits: usize,
    pub ani_evals: usize,
    pub nonpositive_skipped: usize,
    pub borderline_pairs: Vec<GpuCountBorderlinePair>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GpuCountBorderlinePair {
    pub ref_index: usize,
    pub query_index: usize,
    pub dot: i64,
}

impl GpuDotExecutor {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let t0 = Instant::now();

        let ctx = CudaContext::new(gpu_id)?;
        let ptx = compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;
        let event_flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let kernel_start = ctx.new_event(event_flags)?;
        let kernel_end = ctx.new_event(event_flags)?;
        let d2h_start = ctx.new_event(event_flags)?;
        let d2h_end = ctx.new_event(event_flags)?;

        log::debug!(
            "Initialized GPU dot executor on gpu_id={} in {:.3}s",
            gpu_id,
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            gpu_id,
            ctx,
            module,
            d_query: None,
            d_ref: None,
            d_out: None,
            d_count_stats: None,
            d_count_borderline_dots: None,
            d_count_borderline_refs: None,
            d_count_borderline_queries: None,
            kernel_start,
            kernel_end,
            d2h_start,
            d2h_end,
            cap_query: 0,
            cap_ref: 0,
            cap_out: 0,
            cap_count_borderline: 0,
        })
    }

    fn ensure_query_capacity(&mut self, len: usize) -> Result<bool> {
        if self.cap_query < len {
            let stream = self.ctx.default_stream();
            self.d_query = Some(stream.alloc_zeros::<i32>(len)?);
            self.cap_query = len;
            log::debug!(
                "gpu_id={} grow d_query to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i32>())
            );
            return Ok(true);
        }
        Ok(false)
    }

    fn ensure_ref_capacity(&mut self, len: usize) -> Result<bool> {
        if self.cap_ref < len {
            let stream = self.ctx.default_stream();
            self.d_ref = Some(stream.alloc_zeros::<i32>(len)?);
            self.cap_ref = len;
            log::debug!(
                "gpu_id={} grow d_ref to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i32>())
            );
            return Ok(true);
        }
        Ok(false)
    }

    fn ensure_out_capacity(&mut self, len: usize) -> Result<bool> {
        if self.cap_out < len {
            let stream = self.ctx.default_stream();
            self.d_out = Some(stream.alloc_zeros::<i64>(len)?);
            self.cap_out = len;
            log::debug!(
                "gpu_id={} grow d_out to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * std::mem::size_of::<i64>())
            );
            return Ok(true);
        }
        Ok(false)
    }

    fn ensure_count_stats(&mut self) -> Result<()> {
        if self.d_count_stats.is_none() {
            let stream = self.ctx.default_stream();
            self.d_count_stats = Some(stream.alloc_zeros::<u64>(5)?);
            log::debug!("gpu_id={} allocate count stats scratch", self.gpu_id);
        }
        Ok(())
    }

    fn ensure_count_borderline_capacity(&mut self, len: usize) -> Result<()> {
        if self.cap_count_borderline < len {
            let stream = self.ctx.default_stream();
            self.d_count_borderline_dots = Some(stream.alloc_zeros::<i64>(len)?);
            self.d_count_borderline_refs = Some(stream.alloc_zeros::<u32>(len)?);
            self.d_count_borderline_queries = Some(stream.alloc_zeros::<u32>(len)?);
            self.cap_count_borderline = len;
            log::debug!(
                "gpu_id={} grow count borderline scratch to {} elems ({:.2} MiB)",
                self.gpu_id,
                len,
                mib(len * (std::mem::size_of::<i64>() + 2 * std::mem::size_of::<u32>()))
            );
        }
        Ok(())
    }

    pub fn free_memory_bytes(&self) -> Result<usize> {
        let (free, _total) = self.ctx.mem_get_info()?;
        Ok(free)
    }

    pub fn upload_resident_matrix(
        &mut self,
        flat_hv: &[i32],
        rows: usize,
        hv_d: usize,
    ) -> Result<GpuResidentMatrix> {
        if flat_hv.len() != rows * hv_d {
            bail!(
                "resident matrix length mismatch: got {}, expected {}",
                flat_hv.len(),
                rows * hv_d
            );
        }
        if hv_d == 0 {
            bail!("hv_d must be > 0");
        }

        let stream = self.ctx.default_stream();
        let d_hv = stream.clone_htod(flat_hv)?;
        Ok(GpuResidentMatrix { d_hv, rows, hv_d })
    }

    pub fn upload_resident_cards(&mut self, cards: &[f64]) -> Result<GpuResidentCards> {
        if cards.is_empty() {
            bail!("resident cards must be non-empty");
        }

        let stream = self.ctx.default_stream();
        let d_cards = stream.clone_htod(cards)?;
        Ok(GpuResidentCards {
            d_cards,
            rows: cards.len(),
        })
    }

    pub fn compute_tile(
        &mut self,
        query_hv: &[i32],
        nq: usize,
        ref_hv: &[i32],
        nr: usize,
        d: usize,
        out: &mut [i64],
        upload_ref: bool,
    ) -> Result<GpuTileTimings> {
        if query_hv.len() != nq * d {
            bail!(
                "query_hv length mismatch: got {}, expected {}",
                query_hv.len(),
                nq * d
            );
        }
        if ref_hv.len() != nr * d {
            bail!(
                "ref_hv length mismatch: got {}, expected {}",
                ref_hv.len(),
                nr * d
            );
        }
        if out.len() != nq * nr {
            bail!(
                "out length mismatch: got {}, expected {}",
                out.len(),
                nq * nr
            );
        }
        if d == 0 {
            bail!("d must be > 0");
        }

        let t0 = Instant::now();

        self.ensure_query_capacity(query_hv.len())?;
        let ref_capacity_grew = self.ensure_ref_capacity(ref_hv.len())?;
        self.ensure_out_capacity(out.len())?;

        let stream = self.ctx.default_stream();
        let func = self
            .module
            .load_function("dot_rect_i32_i64_tiled_rb")
            .context("load function dot_rect_i32_i64_tiled_rb")?;

        let d_query = self
            .d_query
            .as_mut()
            .context("internal error: d_query missing after allocation")?;
        let d_ref = self
            .d_ref
            .as_mut()
            .context("internal error: d_ref missing after allocation")?;
        let d_out = self
            .d_out
            .as_mut()
            .context("internal error: d_out missing after allocation")?;

        // Reuse persistent buffers. Ref upload is skipped on cache hits unless
        // the device buffer grew and therefore cannot contain the cached tile.
        let query_h2d_start = Instant::now();
        stream.memcpy_htod(query_hv, d_query)?;
        let query_h2d_ns = query_h2d_start.elapsed().as_nanos();

        let ref_upload_performed = upload_ref || ref_capacity_grew;
        let mut ref_h2d_ns = 0u128;
        if ref_upload_performed {
            let ref_h2d_start = Instant::now();
            stream.memcpy_htod(ref_hv, d_ref)?;
            ref_h2d_ns = ref_h2d_start.elapsed().as_nanos();
        }

        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 32;
        const BLOCK_K: usize = 32;
        const THREAD_M: usize = 4;
        const THREAD_N: usize = 2;

        const BLK_X: usize = BLOCK_N / THREAD_N; // 16
        const BLK_Y: usize = BLOCK_M / THREAD_M; // 16

        let smem_bytes = shared_mem_bytes_i32(BLOCK_M, BLOCK_N, BLOCK_K);

        let cfg = LaunchConfig {
            grid_dim: (nr.div_ceil(BLOCK_N) as u32, nq.div_ceil(BLOCK_M) as u32, 1),
            block_dim: (BLK_X as u32, BLK_Y as u32, 1),
            shared_mem_bytes: smem_bytes,
        };

        let nq_i32 = nq as i32;
        let nr_i32 = nr as i32;
        let d_i32 = d as i32;

        let mut launch = stream.launch_builder(&func);
        launch.arg(d_query);
        launch.arg(d_ref);
        launch.arg(&nq_i32);
        launch.arg(&nr_i32);
        launch.arg(&d_i32);
        launch.arg(&mut *d_out);

        let compute_d2h_start = Instant::now();
        self.kernel_start.record(&stream)?;
        unsafe { launch.launch(cfg) }?;
        self.kernel_end.record(&stream)?;

        // Fast common path: full-size tile matches current capacity.
        self.d2h_start.record(&stream)?;
        let out_d2h_elems;
        if self.cap_out == out.len() {
            stream.memcpy_dtoh(d_out, out)?;
            out_d2h_elems = out.len();
        } else {
            // Edge tiles: copy full persistent buffer to temp, then take prefix.
            let mut tmp = vec![0i64; self.cap_out];
            stream.memcpy_dtoh(d_out, &mut tmp)?;
            out.copy_from_slice(&tmp[..out.len()]);
            out_d2h_elems = self.cap_out;
        }
        self.d2h_end.record(&stream)?;
        let kernel_event_ns = event_ms_to_ns(self.kernel_start.elapsed_ms(&self.kernel_end)?);
        let d2h_event_ns = event_ms_to_ns(self.d2h_start.elapsed_ms(&self.d2h_end)?);
        let compute_d2h_ns = compute_d2h_start.elapsed().as_nanos();

        log::debug!(
            "GPU dot tile done on gpu_id={} nq={} nr={} d={} query={:.2} MiB ref={:.2} MiB out={:.2} MiB in {:.3}s",
            self.gpu_id,
            nq,
            nr,
            d,
            mib(query_hv.len() * std::mem::size_of::<i32>()),
            mib(ref_hv.len() * std::mem::size_of::<i32>()),
            mib(out.len() * std::mem::size_of::<i64>()),
            t0.elapsed().as_secs_f64()
        );

        Ok(GpuTileTimings {
            query_h2d_ns,
            ref_h2d_ns,
            compute_d2h_ns,
            kernel_event_ns,
            d2h_event_ns,
            total_ns: t0.elapsed().as_nanos(),
            query_h2d_bytes: std::mem::size_of_val(query_hv),
            ref_h2d_bytes: if ref_upload_performed {
                std::mem::size_of_val(ref_hv)
            } else {
                0
            },
            out_d2h_bytes: out_d2h_elems * std::mem::size_of::<i64>(),
            ref_upload_performed,
        })
    }

    pub fn compute_tile_resident(
        &mut self,
        query: &GpuResidentMatrix,
        q0: usize,
        nq: usize,
        refs: &GpuResidentMatrix,
        r0: usize,
        nr: usize,
        out: &mut [i64],
    ) -> Result<GpuTileTimings> {
        if query.hv_d != refs.hv_d {
            bail!(
                "resident matrix dimension mismatch: query d={} ref d={}",
                query.hv_d,
                refs.hv_d
            );
        }
        if q0.checked_add(nq).is_none_or(|end| end > query.rows) {
            bail!(
                "resident query range out of bounds: q0={} nq={} rows={}",
                q0,
                nq,
                query.rows
            );
        }
        if r0.checked_add(nr).is_none_or(|end| end > refs.rows) {
            bail!(
                "resident ref range out of bounds: r0={} nr={} rows={}",
                r0,
                nr,
                refs.rows
            );
        }
        if out.len() != nq * nr {
            bail!(
                "out length mismatch: got {}, expected {}",
                out.len(),
                nq * nr
            );
        }
        if query.hv_d == 0 {
            bail!("d must be > 0");
        }

        let t0 = Instant::now();

        self.ensure_out_capacity(out.len())?;

        let stream = self.ctx.default_stream();
        let func = self
            .module
            .load_function("dot_rect_i32_i64_tiled_rb")
            .context("load function dot_rect_i32_i64_tiled_rb")?;

        let d_out = self
            .d_out
            .as_mut()
            .context("internal error: d_out missing after allocation")?;

        let q_start = q0 * query.hv_d;
        let q_end = q_start + nq * query.hv_d;
        let r_start = r0 * refs.hv_d;
        let r_end = r_start + nr * refs.hv_d;
        let d_query = query.d_hv.slice(q_start..q_end);
        let d_ref = refs.d_hv.slice(r_start..r_end);

        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 32;
        const BLOCK_K: usize = 32;
        const THREAD_M: usize = 4;
        const THREAD_N: usize = 2;

        const BLK_X: usize = BLOCK_N / THREAD_N;
        const BLK_Y: usize = BLOCK_M / THREAD_M;

        let smem_bytes = shared_mem_bytes_i32(BLOCK_M, BLOCK_N, BLOCK_K);

        let cfg = LaunchConfig {
            grid_dim: (nr.div_ceil(BLOCK_N) as u32, nq.div_ceil(BLOCK_M) as u32, 1),
            block_dim: (BLK_X as u32, BLK_Y as u32, 1),
            shared_mem_bytes: smem_bytes,
        };

        let nq_i32 = nq as i32;
        let nr_i32 = nr as i32;
        let d_i32 = query.hv_d as i32;

        let mut launch = stream.launch_builder(&func);
        launch.arg(&d_query);
        launch.arg(&d_ref);
        launch.arg(&nq_i32);
        launch.arg(&nr_i32);
        launch.arg(&d_i32);
        launch.arg(&mut *d_out);

        let compute_d2h_start = Instant::now();
        self.kernel_start.record(&stream)?;
        unsafe { launch.launch(cfg) }?;
        self.kernel_end.record(&stream)?;

        self.d2h_start.record(&stream)?;
        let out_d2h_elems;
        if self.cap_out == out.len() {
            stream.memcpy_dtoh(d_out, out)?;
            out_d2h_elems = out.len();
        } else {
            let mut tmp = vec![0i64; self.cap_out];
            stream.memcpy_dtoh(d_out, &mut tmp)?;
            out.copy_from_slice(&tmp[..out.len()]);
            out_d2h_elems = self.cap_out;
        }
        self.d2h_end.record(&stream)?;
        let kernel_event_ns = event_ms_to_ns(self.kernel_start.elapsed_ms(&self.kernel_end)?);
        let d2h_event_ns = event_ms_to_ns(self.d2h_start.elapsed_ms(&self.d2h_end)?);
        let compute_d2h_ns = compute_d2h_start.elapsed().as_nanos();

        log::debug!(
            "GPU resident dot tile done on gpu_id={} q0={} nq={} r0={} nr={} d={} out={:.2} MiB in {:.3}s",
            self.gpu_id,
            q0,
            nq,
            r0,
            nr,
            query.hv_d,
            mib(out.len() * std::mem::size_of::<i64>()),
            t0.elapsed().as_secs_f64()
        );

        Ok(GpuTileTimings {
            query_h2d_ns: 0,
            ref_h2d_ns: 0,
            compute_d2h_ns,
            kernel_event_ns,
            d2h_event_ns,
            total_ns: t0.elapsed().as_nanos(),
            query_h2d_bytes: 0,
            ref_h2d_bytes: 0,
            out_d2h_bytes: out_d2h_elems * std::mem::size_of::<i64>(),
            ref_upload_performed: false,
        })
    }

    pub fn compute_count_tile_symmetric_resident(
        &mut self,
        query: &GpuResidentMatrix,
        query_cards: &GpuResidentCards,
        q0: usize,
        nq: usize,
        refs: &GpuResidentMatrix,
        ref_cards: &GpuResidentCards,
        r0: usize,
        nr: usize,
        ksize: u8,
        ani_threshold: f32,
    ) -> Result<GpuCountTileResult> {
        if query.hv_d != refs.hv_d {
            bail!(
                "resident matrix dimension mismatch: query d={} ref d={}",
                query.hv_d,
                refs.hv_d
            );
        }
        if q0.checked_add(nq).is_none_or(|end| end > query.rows) {
            bail!(
                "resident query range out of bounds: q0={} nq={} rows={}",
                q0,
                nq,
                query.rows
            );
        }
        if r0.checked_add(nr).is_none_or(|end| end > refs.rows) {
            bail!(
                "resident ref range out of bounds: r0={} nr={} rows={}",
                r0,
                nr,
                refs.rows
            );
        }
        if q0.checked_add(nq).is_none_or(|end| end > query_cards.rows) {
            bail!(
                "resident query card range out of bounds: q0={} nq={} rows={}",
                q0,
                nq,
                query_cards.rows
            );
        }
        if r0.checked_add(nr).is_none_or(|end| end > ref_cards.rows) {
            bail!(
                "resident ref card range out of bounds: r0={} nr={} rows={}",
                r0,
                nr,
                ref_cards.rows
            );
        }
        if query.hv_d == 0 {
            bail!("d must be > 0");
        }

        let t0 = Instant::now();
        let stream = self.ctx.default_stream();
        let func = self
            .module
            .load_function("dot_rect_count_symmetric_resident")
            .context("load function dot_rect_count_symmetric_resident")?;

        let q_start = q0 * query.hv_d;
        let q_end = q_start + nq * query.hv_d;
        let r_start = r0 * refs.hv_d;
        let r_end = r_start + nr * refs.hv_d;
        let d_query = query.d_hv.slice(q_start..q_end);
        let d_ref = refs.d_hv.slice(r_start..r_end);
        let d_query_cards = query_cards.d_cards.slice(q0..q0 + nq);
        let d_ref_cards = ref_cards.d_cards.slice(r0..r0 + nr);

        self.ensure_count_stats()?;
        self.ensure_count_borderline_capacity(nq * nr)?;
        let d_stats = self
            .d_count_stats
            .as_mut()
            .context("internal error: d_count_stats missing after allocation")?;
        let d_borderline_dots = self
            .d_count_borderline_dots
            .as_mut()
            .context("internal error: d_count_borderline_dots missing after allocation")?;
        let d_borderline_refs = self
            .d_count_borderline_refs
            .as_mut()
            .context("internal error: d_count_borderline_refs missing after allocation")?;
        let d_borderline_queries = self
            .d_count_borderline_queries
            .as_mut()
            .context("internal error: d_count_borderline_queries missing after allocation")?;
        stream.memset_zeros(&mut *d_stats)?;

        const BLOCK_M: usize = 64;
        const BLOCK_N: usize = 32;
        const BLOCK_K: usize = 32;
        const THREAD_M: usize = 4;
        const THREAD_N: usize = 2;

        const BLK_X: usize = BLOCK_N / THREAD_N;
        const BLK_Y: usize = BLOCK_M / THREAD_M;

        let smem_bytes = shared_mem_bytes_i32(BLOCK_M, BLOCK_N, BLOCK_K);

        let cfg = LaunchConfig {
            grid_dim: (nr.div_ceil(BLOCK_N) as u32, nq.div_ceil(BLOCK_M) as u32, 1),
            block_dim: (BLK_X as u32, BLK_Y as u32, 1),
            shared_mem_bytes: smem_bytes,
        };

        let q0_i32 = q0 as i32;
        let r0_i32 = r0 as i32;
        let nq_i32 = nq as i32;
        let nr_i32 = nr as i32;
        let d_i32 = query.hv_d as i32;
        let ksize_i32 = ksize as i32;
        let transformed_threshold = if ani_threshold <= 0.0 {
            0.0
        } else {
            (ani_threshold / 100.0).powf(ksize as f32)
        };
        const COUNT_BORDERLINE_BAND: f32 = 1.0e-6;
        let transformed_threshold_low = (transformed_threshold - COUNT_BORDERLINE_BAND).max(0.0);
        let transformed_threshold_high = transformed_threshold + COUNT_BORDERLINE_BAND;
        let borderline_capacity = (nq * nr) as u32;

        let mut launch = stream.launch_builder(&func);
        launch.arg(&d_query);
        launch.arg(&d_ref);
        launch.arg(&d_query_cards);
        launch.arg(&d_ref_cards);
        launch.arg(&q0_i32);
        launch.arg(&r0_i32);
        launch.arg(&nq_i32);
        launch.arg(&nr_i32);
        launch.arg(&d_i32);
        launch.arg(&ksize_i32);
        launch.arg(&ani_threshold);
        launch.arg(&transformed_threshold_low);
        launch.arg(&transformed_threshold_high);
        launch.arg(&borderline_capacity);
        launch.arg(&mut *d_borderline_dots);
        launch.arg(&mut *d_borderline_refs);
        launch.arg(&mut *d_borderline_queries);
        launch.arg(&mut *d_stats);

        let compute_d2h_start = Instant::now();
        self.kernel_start.record(&stream)?;
        unsafe { launch.launch(cfg) }?;
        self.kernel_end.record(&stream)?;

        let mut stats = [0u64; 5];
        self.d2h_start.record(&stream)?;
        stream.memcpy_dtoh(&*d_stats, &mut stats)?;
        let borderline_count = stats[4] as usize;
        if borderline_count > nq * nr {
            bail!(
                "resident count borderline overflow: count={} capacity={}",
                borderline_count,
                nq * nr
            );
        }
        let mut borderline_dots = vec![0_i64; borderline_count];
        let mut borderline_refs = vec![0_u32; borderline_count];
        let mut borderline_queries = vec![0_u32; borderline_count];
        if borderline_count > 0 {
            stream.memcpy_dtoh(
                &d_borderline_dots.slice(0..borderline_count),
                &mut borderline_dots,
            )?;
            stream.memcpy_dtoh(
                &d_borderline_refs.slice(0..borderline_count),
                &mut borderline_refs,
            )?;
            stream.memcpy_dtoh(
                &d_borderline_queries.slice(0..borderline_count),
                &mut borderline_queries,
            )?;
        }
        self.d2h_end.record(&stream)?;
        let kernel_event_ns = event_ms_to_ns(self.kernel_start.elapsed_ms(&self.kernel_end)?);
        let d2h_event_ns = event_ms_to_ns(self.d2h_start.elapsed_ms(&self.d2h_end)?);
        let compute_d2h_ns = compute_d2h_start.elapsed().as_nanos();
        let borderline_pairs = borderline_dots
            .into_iter()
            .zip(borderline_refs)
            .zip(borderline_queries)
            .map(|((dot, ref_index), query_index)| GpuCountBorderlinePair {
                ref_index: ref_index as usize,
                query_index: query_index as usize,
                dot,
            })
            .collect::<Vec<_>>();

        log::debug!(
            "GPU resident count tile done on gpu_id={} q0={} nq={} r0={} nr={} d={} in {:.3}s",
            self.gpu_id,
            q0,
            nq,
            r0,
            nr,
            query.hv_d,
            t0.elapsed().as_secs_f64()
        );

        Ok(GpuCountTileResult {
            timings: GpuTileTimings {
                query_h2d_ns: 0,
                ref_h2d_ns: 0,
                compute_d2h_ns,
                kernel_event_ns,
                d2h_event_ns,
                total_ns: t0.elapsed().as_nanos(),
                query_h2d_bytes: 0,
                ref_h2d_bytes: 0,
                out_d2h_bytes: stats.len() * std::mem::size_of::<u64>()
                    + borderline_count
                        * (std::mem::size_of::<i64>() + 2 * std::mem::size_of::<u32>()),
                ref_upload_performed: false,
            },
            pairs: stats[0] as usize,
            hits: stats[1] as usize,
            ani_evals: stats[2] as usize,
            nonpositive_skipped: stats[3] as usize,
            borderline_pairs,
        })
    }
}

#[inline]
fn event_ms_to_ns(ms: f32) -> u128 {
    (ms.max(0.0) as f64 * 1_000_000.0).round() as u128
}

#[cfg(test)]
mod tests {
    use super::GpuDotExecutor;

    fn compute(
        gpu: &mut GpuDotExecutor,
        query_hv: &[i32],
        nq: usize,
        ref_hv: &[i32],
        nr: usize,
        d: usize,
        upload_ref: bool,
    ) -> (Vec<i64>, super::GpuTileTimings) {
        let mut out = vec![0i64; nq * nr];
        let timings = gpu
            .compute_tile(query_hv, nq, ref_hv, nr, d, &mut out, upload_ref)
            .expect("GPU tile should compute");
        (out, timings)
    }

    fn resident_count(
        gpu: &mut GpuDotExecutor,
        matrix: &[i32],
        cards: &[f64],
        rows: usize,
        d: usize,
        q0: usize,
        nq: usize,
        r0: usize,
        nr: usize,
        threshold: f32,
    ) -> super::GpuCountTileResult {
        resident_count_with_ksize(gpu, matrix, cards, rows, d, q0, nq, r0, nr, 1, threshold)
    }

    fn resident_count_with_ksize(
        gpu: &mut GpuDotExecutor,
        matrix: &[i32],
        cards: &[f64],
        rows: usize,
        d: usize,
        q0: usize,
        nq: usize,
        r0: usize,
        nr: usize,
        ksize: u8,
        threshold: f32,
    ) -> super::GpuCountTileResult {
        let resident = gpu
            .upload_resident_matrix(matrix, rows, d)
            .expect("resident matrix upload");
        let resident_cards = gpu
            .upload_resident_cards(cards)
            .expect("resident cards upload");
        gpu.compute_count_tile_symmetric_resident(
            &resident,
            &resident_cards,
            q0,
            nq,
            &resident,
            &resident_cards,
            r0,
            nr,
            ksize,
            threshold,
        )
        .expect("resident count tile should compute")
    }

    #[test]
    fn compute_tile_upload_ref_true_matches_small_matrix() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let query = [1, 2, 3, 4, 0, 1];
        let refs = [10, 0, 0, 0, 10, 0, 1, 1, 1];

        let (out, timings) = compute(&mut gpu, &query, 2, &refs, 3, 3, true);

        assert_eq!(out, vec![10, 20, 6, 40, 0, 5]);
        assert!(timings.ref_upload_performed);
        assert_eq!(
            timings.query_h2d_bytes,
            query.len() * std::mem::size_of::<i32>()
        );
        assert_eq!(
            timings.ref_h2d_bytes,
            refs.len() * std::mem::size_of::<i32>()
        );
        assert_eq!(
            timings.out_d2h_bytes,
            out.len() * std::mem::size_of::<i64>()
        );
    }

    #[test]
    fn compute_tile_upload_ref_false_reuses_previous_ref_with_new_query() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let refs = [1, 2, 3, 4];
        let bogus_refs = [9, 9, 9, 9];

        let (first, first_timings) = compute(&mut gpu, &[1, 0], 1, &refs, 2, 2, true);
        let (second, second_timings) = compute(&mut gpu, &[0, 1], 1, &bogus_refs, 2, 2, false);

        assert_eq!(first, vec![1, 3]);
        assert_eq!(second, vec![2, 4]);
        assert!(first_timings.ref_upload_performed);
        assert!(!second_timings.ref_upload_performed);
        assert_eq!(second_timings.ref_h2d_bytes, 0);
    }

    #[test]
    fn compute_tile_upload_ref_true_replaces_previous_ref() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");

        let (first, _) = compute(&mut gpu, &[1, 1], 1, &[1, 2, 3, 4], 2, 2, true);
        let (second, timings) = compute(&mut gpu, &[1, 1], 1, &[5, 6, 7, 8], 2, 2, true);

        assert_eq!(first, vec![3, 7]);
        assert_eq!(second, vec![11, 15]);
        assert!(timings.ref_upload_performed);
    }

    #[test]
    fn compute_tile_forces_ref_upload_after_ref_capacity_grows() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");

        let (first, _) = compute(&mut gpu, &[1, 0], 1, &[1, 2], 1, 2, true);
        let (second, timings) = compute(&mut gpu, &[1, 0], 1, &[5, 6, 7, 8], 2, 2, false);

        assert_eq!(first, vec![1]);
        assert_eq!(second, vec![5, 7]);
        assert!(timings.ref_upload_performed);
        assert_eq!(timings.ref_h2d_bytes, 4 * std::mem::size_of::<i32>());
    }

    #[test]
    fn resident_full_matrix_tile_matches_tiled_compute_tile() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [1, 2, 3, 4, 0, 1, 2, 0, 2];
        let resident = gpu
            .upload_resident_matrix(&matrix, 3, 3)
            .expect("resident upload");

        let mut resident_out = vec![0i64; 9];
        gpu.compute_tile_resident(&resident, 0, 3, &resident, 0, 3, &mut resident_out)
            .expect("resident tile should compute");

        let mut tiled_out = vec![0i64; 9];
        gpu.compute_tile(&matrix, 3, &matrix, 3, 3, &mut tiled_out, true)
            .expect("tiled tile should compute");

        assert_eq!(resident_out, tiled_out);
    }

    #[test]
    fn resident_slicing_with_nonzero_offsets_returns_expected_submatrix() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [
            1, 0, 0, 0, // row 0
            0, 1, 0, 0, // row 1
            1, 1, 0, 0, // row 2
            2, 0, 1, 0, // row 3
        ];
        let resident = gpu
            .upload_resident_matrix(&matrix, 4, 4)
            .expect("resident upload");

        let mut out = vec![0i64; 4];
        gpu.compute_tile_resident(&resident, 1, 2, &resident, 2, 2, &mut out)
            .expect("resident tile should compute");

        assert_eq!(out, vec![1, 0, 2, 2]);
    }

    #[test]
    fn symmetric_resident_mode_can_reuse_same_matrix_for_query_and_ref() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [2, 0, 0, 0, 3, 1];
        let resident = gpu
            .upload_resident_matrix(&matrix, 2, 3)
            .expect("resident upload");

        let mut out = vec![0i64; 4];
        let timings = gpu
            .compute_tile_resident(&resident, 0, 2, &resident, 0, 2, &mut out)
            .expect("resident tile should compute");

        assert_eq!(out, vec![4, 0, 0, 10]);
        assert_eq!(timings.query_h2d_bytes, 0);
        assert_eq!(timings.ref_h2d_bytes, 0);
        assert!(!timings.ref_upload_performed);
    }

    #[test]
    fn resident_count_threshold_zero_keeps_zero_ani_pairs() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [4, 0, 0, 0, 90, 0, 0, 0, 0, 90, 0, 0];
        let cards = [100.0, 100.0, 100.0];

        let result = resident_count(&mut gpu, &matrix, &cards, 3, 4, 0, 3, 0, 3, 0.0);

        assert_eq!(result.pairs, 3);
        assert_eq!(result.hits, 3);
        assert_eq!(result.ani_evals, 3);
        assert_eq!(result.nonpositive_skipped, 0);
        assert_eq!(result.timings.out_d2h_bytes, 5 * std::mem::size_of::<u64>());
    }

    #[test]
    fn resident_count_positive_threshold_skips_nonpositive_pairs() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [4, 0, 0, 0, 90, 0, 0, 0, 0, 90, 0, 0];
        let cards = [100.0, 100.0, 100.0];

        let result = resident_count(&mut gpu, &matrix, &cards, 3, 4, 0, 3, 0, 3, 85.0);

        assert_eq!(result.pairs, 3);
        assert_eq!(result.hits, 1);
        assert_eq!(result.ani_evals, 1);
        assert_eq!(result.nonpositive_skipped, 2);
    }

    #[test]
    fn resident_count_edge_offsets_skip_diagonal_and_lower_triangle() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [
            4, 0, 0, 0, //
            4, 0, 0, 0, //
            0, 4, 0, 0, //
            0, 0, 4, 0,
        ];
        let cards = [100.0, 100.0, 100.0, 100.0];

        let result = resident_count(&mut gpu, &matrix, &cards, 4, 4, 1, 3, 0, 2, 0.0);

        assert_eq!(result.pairs, 5);
        assert_eq!(result.hits, 5);
        assert_eq!(result.ani_evals, 5);
        assert_eq!(result.nonpositive_skipped, 0);
    }

    #[test]
    fn resident_count_clamps_estimated_jaccard_overshoot_to_100() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let matrix = [4, 0, 0, 0, 4, 0, 0, 0];
        let cards = [3.0, 3.0];

        let result = resident_count(&mut gpu, &matrix, &cards, 2, 4, 0, 2, 0, 2, 100.0);

        assert_eq!(result.pairs, 1);
        assert_eq!(result.hits, 1);
        assert_eq!(result.ani_evals, 1);
        assert_eq!(result.nonpositive_skipped, 0);
    }

    #[test]
    fn resident_count_matches_host_near_ani_threshold_with_realistic_ksize() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let ksize = 21;
        let threshold = 85.0;
        let dot = 4096_i64;
        let d = 4_usize;
        let inter_hat = dot as f64 / d as f64;
        let target = 0.85_f64.powi(ksize as i32);
        let mut card = inter_hat / target;

        for step in 0..10_000 {
            let candidate = card * (1.0 + step as f64 * 1.0e-8);
            let ani = crate::dist::ani_from_intersection_and_cardinalities(
                inter_hat, candidate, candidate, ksize,
            );
            if (threshold..=threshold + 0.0001).contains(&ani) {
                card = candidate;
                break;
            }
        }

        let host_ani =
            crate::dist::ani_from_intersection_and_cardinalities(inter_hat, card, card, ksize);
        assert!(
            host_ani >= threshold,
            "test setup should produce a host hit, got {host_ani}"
        );

        let matrix = [4096, 0, 0, 0, 1, 0, 0, 0];
        let cards = [card, card];
        let result = resident_count_with_ksize(
            &mut gpu, &matrix, &cards, 2, d, 0, 2, 0, 2, ksize, threshold,
        );

        assert_eq!(result.pairs, 1);
        let host_rechecked_hits = result.hits
            + result
                .borderline_pairs
                .iter()
                .filter(|pair| {
                    let inter_hat = pair.dot as f64 / d as f64;
                    crate::dist::ani_from_intersection_and_cardinalities(
                        inter_hat,
                        cards[pair.ref_index],
                        cards[pair.query_index],
                        ksize,
                    ) >= threshold
                })
                .count();
        assert_eq!(host_rechecked_hits, 1, "host ANI was {host_ani}");
    }

    #[test]
    fn fallback_tiled_path_remains_correct() {
        let mut gpu = GpuDotExecutor::new(0).expect("GPU executor");
        let query = [1, 2, 0, 1];
        let refs = [3, 0, 0, 4, 5, 6];

        let (out, timings) = compute(&mut gpu, &query, 2, &refs, 3, 2, true);

        assert_eq!(out, vec![3, 8, 17, 0, 4, 6]);
        assert_eq!(
            timings.query_h2d_bytes,
            query.len() * std::mem::size_of::<i32>()
        );
        assert_eq!(
            timings.ref_h2d_bytes,
            refs.len() * std::mem::size_of::<i32>()
        );
    }
}
