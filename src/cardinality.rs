use anyhow::{Result, anyhow};
use exaloglog::ExaLogLog;
use ultraloglog::UltraLogLog;

use crate::types::{
    CardinalityEstimator, FileCardinalitySketch, FileEllSketch, FileUllSketch, SketchParams,
};

pub(crate) enum CardinalitySketch {
    Ull(UltraLogLog),
    Ell(ExaLogLog),
}

impl CardinalitySketch {
    pub(crate) fn new(params: &SketchParams) -> Result<Self> {
        match params.cardinality_estimator {
            CardinalityEstimator::Ull => UltraLogLog::new(params.ull_p)
                .map(Self::Ull)
                .map_err(|error| anyhow!("invalid ULL parameters: {error}")),
            CardinalityEstimator::Ell => ExaLogLog::new(params.ell_t, params.ell_d, params.ell_p)
                .map(Self::Ell)
                .map_err(|error| anyhow!("invalid ELL parameters: {error}")),
        }
    }

    pub(crate) fn into_record(
        self,
        params: &SketchParams,
        file_str: String,
    ) -> FileCardinalitySketch {
        match self {
            Self::Ull(ull) => FileCardinalitySketch::Ull(FileUllSketch {
                ksize: params.ksize,
                canonical: params.canonical,
                seed: params.seed,
                ull_p: params.ull_p,
                file_str,
                ull_state: ull.get_state().to_vec(),
            }),
            Self::Ell(ell) => FileCardinalitySketch::Ell(FileEllSketch {
                ksize: params.ksize,
                canonical: params.canonical,
                seed: params.seed,
                ell_t: params.ell_t,
                ell_d: params.ell_d,
                file_str,
                ell_state: ell.into_state(),
            }),
        }
    }
}
