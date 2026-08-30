use super::{ani_from_intersection_and_cardinalities, dist};
use crate::types::SketchDist;

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
