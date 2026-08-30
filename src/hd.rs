use log::info;
use std::collections::HashSet;

use crate::types::FileSketch;
use anyhow::{Result, bail};
use rand::{RngCore, SeedableRng};
use wyhash::WyRng;

extern crate bitpacking;
use bitpacking::{BitPacker, BitPacker8x};

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar reference implementation.
/// Math:
/// - hv starts at -N for every coordinate
/// - each seed contributes +2 when the corresponding random bit is 1
///
/// So final coordinate = (#ones * 2) - N.
pub(crate) fn validate_hv_dimension(hv_d: usize) -> Result<()> {
    if hv_d == 0 {
        bail!("hv_d must be greater than zero");
    }
    if !hv_d.is_multiple_of(256) {
        bail!("hv_d must be divisible by 256 (found {hv_d})");
    }
    Ok(())
}

pub fn encode_hash_hd(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    assert!(sketch.hv_d != 0 && sketch.hv_d.is_multiple_of(256));
    assert!(kmer_hash_set.len() <= i32::MAX as usize);
    let hv_d = sketch.hv_d;
    let seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let mut hv = vec![-(kmer_hash_set.len() as i32); hv_d];

    for hash in seed_vec {
        let mut rng = WyRng::seed_from_u64(hash);

        for i in 0..(hv_d / 64) {
            let rnd_bits = rng.next_u64();

            for j in 0..64 {
                hv[i * 64 + j] += (((rnd_bits >> j) & 1) << 1) as i32;
            }
        }
    }

    hv
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// The caller must execute this function only when AVX2 is available.
pub unsafe fn encode_hash_hd_avx2(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    assert!(sketch.hv_d != 0 && sketch.hv_d.is_multiple_of(256));
    assert!(kmer_hash_set.len() <= i32::MAX as usize);
    let hv_d = sketch.hv_d;
    let num_seed = kmer_hash_set.len();

    let mut hv = vec![-(num_seed as i32); hv_d];

    let mut seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let num_tail = num_seed % 4;
    let num_seed_round_4 = num_seed + if num_tail == 0 { 0 } else { 4 - num_tail };
    let num_batch_round_4 = num_seed_round_4 / 4;
    let num_chunk = hv_d / 64;

    seed_vec.resize(num_seed_round_4, 0);

    let mut rng_vec = [
        WyRng::default(),
        WyRng::default(),
        WyRng::default(),
        WyRng::default(),
    ];
    let mut rnd_vec = [0u64; 4];

    for b_i in 0..num_batch_round_4 {
        for lane in 0..4 {
            rng_vec[lane] = WyRng::seed_from_u64(seed_vec[b_i * 4 + lane]);
        }

        for chunk_i in 0..num_chunk {
            for lane in 0..4 {
                rnd_vec[lane] = rng_vec[lane].next_u64();
            }

            if b_i == num_batch_round_4 - 1 && num_tail > 0 {
                for lane in num_tail..4 {
                    rnd_vec[lane] = 0;
                }
            }

            let vrnd = _mm256_set_epi64x(
                rnd_vec[3] as i64,
                rnd_vec[2] as i64,
                rnd_vec[1] as i64,
                rnd_vec[0] as i64,
            );

            let base = chunk_i * 64;

            for bit in 0..64 {
                let mask64 = 1u64 << bit;
                let vmask = _mm256_set1_epi64x(mask64 as i64);
                let m = _mm256_cmpeq_epi64(_mm256_and_si256(vrnd, vmask), vmask);
                let lane_mask = _mm256_movemask_pd(_mm256_castsi256_pd(m)) as u32;
                let ones = lane_mask.count_ones() as i32;
                hv[base + bit] += ones << 1;
            }
        }
    }

    hv
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// # Safety
///
/// The caller must execute this function only when AVX-512F is available.
pub unsafe fn encode_hash_hd_avx512(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i32> {
    assert!(sketch.hv_d != 0 && sketch.hv_d.is_multiple_of(256));
    assert!(kmer_hash_set.len() <= i32::MAX as usize);
    let hv_d = sketch.hv_d;
    let num_seed = kmer_hash_set.len();

    let mut hv = vec![-(num_seed as i32); hv_d];

    // Batch 16 seeds at once using two zmm registers.
    let mut seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let num_tail = num_seed % 16;
    let num_seed_round_16 = num_seed + if num_tail == 0 { 0 } else { 16 - num_tail };
    let num_batch_round_16 = num_seed_round_16 / 16;
    let num_chunk = hv_d / 64;

    seed_vec.resize(num_seed_round_16, 0);

    let mut rng_vec = std::array::from_fn::<WyRng, 16, _>(|_| WyRng::default());
    let mut rnd_vec = [0u64; 16];

    for b_i in 0..num_batch_round_16 {
        let batch_base = b_i * 16;

        for lane in 0..16 {
            rng_vec[lane] = WyRng::seed_from_u64(seed_vec[batch_base + lane]);
        }

        for chunk_i in 0..num_chunk {
            for lane in 0..16 {
                rnd_vec[lane] = rng_vec[lane].next_u64();
            }

            if b_i == num_batch_round_16 - 1 && num_tail > 0 {
                for lane in num_tail..16 {
                    rnd_vec[lane] = 0;
                }
            }

            let vrnd0 = _mm512_set_epi64(
                rnd_vec[7] as i64,
                rnd_vec[6] as i64,
                rnd_vec[5] as i64,
                rnd_vec[4] as i64,
                rnd_vec[3] as i64,
                rnd_vec[2] as i64,
                rnd_vec[1] as i64,
                rnd_vec[0] as i64,
            );

            let vrnd1 = _mm512_set_epi64(
                rnd_vec[15] as i64,
                rnd_vec[14] as i64,
                rnd_vec[13] as i64,
                rnd_vec[12] as i64,
                rnd_vec[11] as i64,
                rnd_vec[10] as i64,
                rnd_vec[9] as i64,
                rnd_vec[8] as i64,
            );

            let base = chunk_i * 64;

            // Still one update per bit position, but now over 16 seeds per pass.
            for bit in 0..64 {
                let mask64 = 1u64 << bit;
                let vmask = _mm512_set1_epi64(mask64 as i64);

                let m0 = _mm512_cmpeq_epi64_mask(_mm512_and_si512(vrnd0, vmask), vmask);
                let m1 = _mm512_cmpeq_epi64_mask(_mm512_and_si512(vrnd1, vmask), vmask);

                let ones = (m0.count_ones() + m1.count_ones()) as i32;
                hv[base + bit] += ones << 1;
            }
        }
    }

    hv
}

#[cfg(target_arch = "x86_64")]
pub fn compress_hd_sketch(sketch: &mut FileSketch, hv: &[i32]) -> Result<u8> {
    validate_hv_dimension(sketch.hv_d)?;
    if hv.len() != sketch.hv_d {
        bail!(
            "HD vector length {} does not match hv_d {}",
            hv.len(),
            sketch.hv_d
        );
    }
    let hv_d = sketch.hv_d;

    let min_hv = hv
        .iter()
        .min()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("HD vector is empty"))?;
    let max_hv = hv
        .iter()
        .max()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("HD vector is empty"))?;

    let mut quant_bit: u8 = 6;
    loop {
        let quant_min: i64 = -(1i64 << (quant_bit - 1));
        let quant_max: i64 = (1i64 << (quant_bit - 1)) - 1;

        if quant_min <= min_hv as i64 && quant_max >= max_hv as i64 {
            break;
        }

        if quant_bit == 32 {
            break;
        }
        quant_bit += 1;
    }

    if is_x86_feature_detected!("avx2") {
        let offset: i64 = 1i64 << (quant_bit - 1);
        let hv_u32: Vec<u32> = hv.iter().map(|&x| (x as i64 + offset) as u32).collect();

        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let mut hv_compress_bits = vec![0u8; quant_bit as usize * (hv_d >> 3)];
        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.compress(
                &hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                &mut hv_compress_bits[(bits_per_block * i)..(bits_per_block * (i + 1))],
                quant_bit,
            );
        }

        let packed = hv_compress_bits
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|bytes| i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        sketch.hv.clone_from(&packed);
    } else {
        let total_bits = quant_bit as usize * hv_d;
        let len_bit_vec_u32 = total_bits.div_ceil(32);
        let mut hv_compress_bits: Vec<i32> = vec![0; len_bit_vec_u32];

        let offset: i64 = 1i64 << (quant_bit - 1);
        let hv_u32: Vec<u32> = hv.iter().map(|&x| (x as i64 + offset) as u32).collect();

        for bit_idx in 0..total_bits {
            let src_idx = bit_idx / quant_bit as usize;
            let src_bit = bit_idx % quant_bit as usize;
            let bit = ((hv_u32[src_idx] >> src_bit) & 1) as i32;
            hv_compress_bits[bit_idx / 32] |= bit << (bit_idx % 32);
        }

        sketch.hv.clone_from(&hv_compress_bits);
    }

    Ok(quant_bit)
}

pub fn decompress_file_sketch(file_sketch: &mut [FileSketch]) -> Result<()> {
    if file_sketch.is_empty() {
        bail!("HD sketch collection is empty");
    }
    let hv_dim = file_sketch[0].hv_d;
    info!("Decompressing sketch with HV dim={}", hv_dim);

    file_sketch.par_iter_mut().try_for_each(|sketch| {
        let hv_decompressed = decompress_hd_sketch(sketch)?;
        sketch.hv = hv_decompressed;
        Ok::<(), anyhow::Error>(())
    })
}

#[cfg(target_arch = "x86_64")]
pub fn decompress_hd_sketch(sketch: &mut FileSketch) -> Result<Vec<i32>> {
    validate_hv_dimension(sketch.hv_d)?;
    let hv_d = sketch.hv_d;
    let quant_bit = sketch.hv_quant_bits;

    if quant_bit == 0 {
        if sketch.hv.len() != hv_d {
            bail!(
                "raw HD vector length {} does not match hv_d {}",
                sketch.hv.len(),
                hv_d
            );
        }
        return Ok(sketch.hv.clone());
    }
    if !(6..=32).contains(&quant_bit) {
        bail!("invalid HD quantization bit width {quant_bit}");
    }
    let expected_len = hv_d
        .checked_mul(quant_bit as usize)
        .and_then(|bits| bits.checked_div(32))
        .ok_or_else(|| anyhow::anyhow!("compressed HD vector length overflows"))?;
    if sketch.hv.len() != expected_len {
        bail!(
            "compressed HD vector length {} does not match expected {}",
            sketch.hv.len(),
            expected_len
        );
    }

    let mut hv_decompressed: Vec<i32> = vec![0; hv_d];

    if is_x86_feature_detected!("avx2") {
        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let hv_u8 = sketch
            .hv
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut hv_u32: Vec<u32> = vec![0; hv_d];

        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.decompress(
                &hv_u8[(bits_per_block * i)..(bits_per_block * (i + 1))],
                &mut hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                quant_bit,
            );
        }

        let offset: i64 = 1i64 << (quant_bit - 1);
        hv_decompressed = hv_u32
            .into_iter()
            .map(|x| (x as i64 - offset) as i32)
            .collect();
    } else {
        let total_bits = quant_bit as usize * hv_d;
        let mut hv_u32: Vec<u32> = vec![0; hv_d];

        for bit_idx in 0..total_bits {
            let bit = ((sketch.hv[bit_idx / 32] as u32) >> (bit_idx % 32)) & 1;
            hv_u32[bit_idx / quant_bit as usize] |= bit << (bit_idx % quant_bit as usize);
        }

        let offset: i64 = 1i64 << (quant_bit - 1);
        for i in 0..hv_d {
            hv_decompressed[i] = (hv_u32[i] as i64 - offset) as i32;
        }
    }

    Ok(hv_decompressed)
}
