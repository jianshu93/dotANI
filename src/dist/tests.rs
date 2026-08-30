use super::{ani_from_intersection_and_cardinalities, dist};
use crate::types::{FileSketch, SketchDist};

#[test]
fn ani_clamps_estimated_jaccard_overshoot_to_100() {
    let ani = ani_from_intersection_and_cardinalities(120.0, 100.0, 100.0, 16);
    assert!(
        (ani - 100.0).abs() < f32::EPSILON,
        "expected overshot Jaccard estimate to produce 100 ANI, got {ani}"
    );
}

#[test]
fn dist_validates_public_boundary_parameters_before_loading_files() {
    let cases = [
        (
            SketchDist {
                ani_threshold: f32::NAN,
                ..SketchDist::default()
            },
            "ANI threshold",
        ),
        (
            SketchDist {
                threads: 0,
                ..SketchDist::default()
            },
            "thread count",
        ),
        (SketchDist::default(), "output path"),
    ];

    for (mut params, expected) in cases {
        let error = dist(&mut params).unwrap_err().to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

fn test_filesketch(file_str: &str, hv: Vec<i32>) -> FileSketch {
    FileSketch {
        ksize: 1,
        scaled: 1,
        canonical: true,
        seed: 123,
        hv_d: hv.len(),
        hv_quant_bits: 0,
        hv_norm_2: 0,
        file_str: file_str.to_string(),
        hv,
    }
}

// --- lazy resident host-matrix flattening regression tests ---

#[cfg(feature = "cuda")]
#[test]
fn resident_matrix_bytes_checked_and_overflow() {
    let (upload_bytes, required_bytes) =
        super::resident_matrix_bytes(3, 4, 2, 512, 512).expect("small plan must not overflow");
    assert_eq!(
        upload_bytes,
        3 * 4 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f64>()
    );
    assert_eq!(
        required_bytes,
        upload_bytes + 512 * 512 * std::mem::size_of::<i64>() + super::RESIDENT_SAFETY_BYTES
    );
    // Any overflowed component must report the resident path as infeasible.
    assert!(super::resident_matrix_bytes(usize::MAX, usize::MAX, 1, 512, 512).is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn resident_flat_if_feasible_gates_flatten_and_shares_one_matrix() {
    let sketches = vec![
        test_filesketch("s0", vec![1, 2, 3, 4]),
        test_filesketch("s1", vec![5, 6, 7, 8]),
    ];
    let (_, required_bytes) = super::resident_matrix_bytes(sketches.len(), 4, 2, 512, 512)
        .expect("small plan must not overflow");
    let cell = std::sync::OnceLock::<(Vec<i32>, u128)>::new();

    // Equal and insufficient capacity leave the shared cell empty.
    assert!(
        super::resident_flat_if_feasible(required_bytes, required_bytes, &cell, &sketches)
            .is_none()
    );
    assert!(super::resident_flat_if_feasible(0, required_bytes, &cell, &sketches).is_none());
    assert!(cell.get().is_none());

    // Feasible scoped workers all share the one initialized matrix.
    const WORKERS: usize = 4;
    let mut observed: Vec<&[i32]> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let cell = &cell;
                let sketches = &sketches;
                scope.spawn(move || {
                    super::resident_flat_if_feasible(
                        required_bytes + 1,
                        required_bytes,
                        cell,
                        sketches,
                    )
                })
            })
            .collect();
        for handle in handles {
            observed.push(
                handle
                    .join()
                    .expect("resident worker panicked")
                    .expect("feasible worker must see the resident matrix"),
            );
        }
    });

    assert_eq!(observed.len(), WORKERS);
    let expected: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(observed[0], expected);
    assert!(
        observed.iter().all(|flat| std::ptr::eq(*flat, observed[0])),
        "every feasible worker must share the one flattened matrix"
    );
}
