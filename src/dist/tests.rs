use super::ani_from_intersection_and_cardinalities;

#[test]
fn ani_clamps_estimated_jaccard_overshoot_to_100() {
    let ani = ani_from_intersection_and_cardinalities(120.0, 100.0, 100.0, 16);
    assert!(
        (ani - 100.0).abs() < f32::EPSILON,
        "expected overshot Jaccard estimate to produce 100 ANI, got {ani}"
    );
}
