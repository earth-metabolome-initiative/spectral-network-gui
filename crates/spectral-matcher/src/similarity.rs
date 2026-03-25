use std::sync::Arc;

use mass_spectrometry::prelude::{
    GenericSpectrum, GreedyCosine, HungarianCosine, LinearEntropy, ModifiedGreedyCosine,
    ModifiedHungarianCosine, ModifiedLinearEntropy, MsEntropyCleanSpectrum, ScalarSimilarity,
    SiriusMergeClosePeaks, SpectralProcessor, SpectrumAlloc, SpectrumMut,
};
use serde::{Deserialize, Serialize};

use crate::model::SpectrumRecord;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    CosineHungarian,
    #[default]
    CosineGreedy,
    ModifiedCosine,
    ModifiedGreedyCosine,
    LinearEntropyWeighted,
    LinearEntropyUnweighted,
    ModifiedLinearEntropyWeighted,
    ModifiedLinearEntropyUnweighted,
}

impl SimilarityMetric {
    pub const ALL: [Self; 8] = [
        Self::CosineHungarian,
        Self::CosineGreedy,
        Self::ModifiedCosine,
        Self::ModifiedGreedyCosine,
        Self::LinearEntropyWeighted,
        Self::LinearEntropyUnweighted,
        Self::ModifiedLinearEntropyWeighted,
        Self::ModifiedLinearEntropyUnweighted,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::CosineHungarian => "CosineHungarian",
            Self::CosineGreedy => "CosineGreedy",
            Self::ModifiedCosine => "ModifiedCosine",
            Self::ModifiedGreedyCosine => "ModifiedGreedyCosine",
            Self::LinearEntropyWeighted => "LinearEntropyWeighted",
            Self::LinearEntropyUnweighted => "LinearEntropyUnweighted",
            Self::ModifiedLinearEntropyWeighted => "ModifiedLinearEntropyWeighted",
            Self::ModifiedLinearEntropyUnweighted => "ModifiedLinearEntropyUnweighted",
        }
    }

    fn needs_linear_entropy_preprocessing(self) -> bool {
        matches!(
            self,
            Self::LinearEntropyWeighted
                | Self::LinearEntropyUnweighted
                | Self::ModifiedLinearEntropyWeighted
                | Self::ModifiedLinearEntropyUnweighted
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ComputeParams {
    pub metric: SimilarityMetric,
    pub tolerance: f64,
    pub mz_power: f64,
    pub intensity_power: f64,
    #[serde(default)]
    pub top_n_peaks: Option<usize>,
}

enum MetricScorerInner {
    CosineHungarian(HungarianCosine<f64, f64>),
    CosineGreedy(GreedyCosine<f64, f64>),
    ModifiedCosine(ModifiedHungarianCosine<f64, f64>),
    ModifiedGreedyCosine(ModifiedGreedyCosine<f64, f64>),
    LinearEntropyWeighted(LinearEntropy<f64, f64>),
    LinearEntropyUnweighted(LinearEntropy<f64, f64>),
    ModifiedLinearEntropyWeighted(ModifiedLinearEntropy<f64, f64>),
    ModifiedLinearEntropyUnweighted(ModifiedLinearEntropy<f64, f64>),
}

pub struct MetricScorer {
    metric: SimilarityMetric,
    inner: MetricScorerInner,
}

impl MetricScorer {
    pub fn new(params: ComputeParams) -> Result<Self, String> {
        let inner = match params.metric {
            SimilarityMetric::CosineHungarian => HungarianCosine::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
            )
            .map(MetricScorerInner::CosineHungarian),
            SimilarityMetric::CosineGreedy => GreedyCosine::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
            )
            .map(MetricScorerInner::CosineGreedy),
            SimilarityMetric::ModifiedCosine => ModifiedHungarianCosine::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
            )
            .map(MetricScorerInner::ModifiedCosine),
            SimilarityMetric::ModifiedGreedyCosine => ModifiedGreedyCosine::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
            )
            .map(MetricScorerInner::ModifiedGreedyCosine),
            SimilarityMetric::LinearEntropyWeighted => LinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                true,
            )
            .map(MetricScorerInner::LinearEntropyWeighted),
            SimilarityMetric::LinearEntropyUnweighted => LinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                false,
            )
            .map(MetricScorerInner::LinearEntropyUnweighted),
            SimilarityMetric::ModifiedLinearEntropyWeighted => ModifiedLinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                true,
            )
            .map(MetricScorerInner::ModifiedLinearEntropyWeighted),
            SimilarityMetric::ModifiedLinearEntropyUnweighted => ModifiedLinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                false,
            )
            .map(MetricScorerInner::ModifiedLinearEntropyUnweighted),
        }
        .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label()))?;

        Ok(Self {
            metric: params.metric,
            inner,
        })
    }

    pub fn similarity(
        &self,
        left: &GenericSpectrum<f64, f64>,
        right: &GenericSpectrum<f64, f64>,
        left_idx: usize,
        right_idx: usize,
    ) -> Result<(f64, usize), String> {
        let result = match &self.inner {
            MetricScorerInner::CosineHungarian(sim) => sim.similarity(left, right),
            MetricScorerInner::CosineGreedy(sim) => sim.similarity(left, right),
            MetricScorerInner::ModifiedCosine(sim) => sim.similarity(left, right),
            MetricScorerInner::ModifiedGreedyCosine(sim) => sim.similarity(left, right),
            MetricScorerInner::LinearEntropyWeighted(sim) => sim.similarity(left, right),
            MetricScorerInner::LinearEntropyUnweighted(sim) => sim.similarity(left, right),
            MetricScorerInner::ModifiedLinearEntropyWeighted(sim) => sim.similarity(left, right),
            MetricScorerInner::ModifiedLinearEntropyUnweighted(sim) => sim.similarity(left, right),
        };
        result.map_err(|err| {
            format!(
                "{} failed for pair ({left_idx},{right_idx}): {err:?}",
                self.metric.label()
            )
        })
    }
}

pub fn preprocess_spectra_for_metric<T>(
    spectra: Vec<SpectrumRecord<T>>,
    params: ComputeParams,
) -> Result<Vec<SpectrumRecord<T>>, String> {
    let spectra = apply_top_n_peak_filter(spectra, params.top_n_peaks)?;

    if !params.metric.needs_linear_entropy_preprocessing() {
        return Ok(spectra);
    }

    let cleaner = MsEntropyCleanSpectrum::<f64>::builder()
        .build()
        .map_err(|err| format!("failed to configure ms_entropy cleaner: {err:?}"))?;
    let merger = SiriusMergeClosePeaks::new(params.tolerance)
        .map_err(|err| format!("failed to configure close-peak merger: {err:?}"))?;

    Ok(spectra
        .into_iter()
        .map(|mut record| {
            let cleaned = cleaner.process(record.spectrum.as_ref());
            let merged = merger.process(&cleaned);
            record.spectrum = Arc::new(merged);
            record
        })
        .collect())
}

fn apply_top_n_peak_filter<T>(
    spectra: Vec<SpectrumRecord<T>>,
    top_n_peaks: Option<usize>,
) -> Result<Vec<SpectrumRecord<T>>, String> {
    let Some(limit) = top_n_peaks.filter(|limit| *limit > 0) else {
        return Ok(spectra);
    };

    spectra
        .into_iter()
        .map(|mut record| {
            if record.peaks.len() <= limit {
                return Ok(record);
            }

            let mut selected = record.peaks.as_ref().clone();
            selected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.total_cmp(&b.0)));
            selected.truncate(limit);
            selected.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut spectrum =
                GenericSpectrum::<f64, f64>::with_capacity(record.meta.precursor_mz, selected.len())
                    .map_err(|err| {
                        format!(
                            "failed to allocate top-{limit} filtered spectrum for node {}: {err:?}",
                            record.meta.id
                        )
                    })?;
            for (mz, intensity) in &selected {
                spectrum.add_peak(*mz, *intensity).map_err(|err| {
                    format!(
                        "failed to add top-{limit} filtered peak for node {}: {err:?}",
                        record.meta.id
                    )
                })?;
            }
            record.spectrum = Arc::new(spectrum);
            Ok(record)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

    use super::{ComputeParams, MetricScorer, SimilarityMetric, preprocess_spectra_for_metric};
    use crate::model::{SpectrumMetadata, SpectrumRecord};

    fn record(id: usize, peaks: &[(f64, f64)]) -> SpectrumRecord {
        let mut sorted_peaks = peaks.to_vec();
        sorted_peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut spectrum = GenericSpectrum::<f64, f64>::with_capacity(100.0, sorted_peaks.len())
            .expect("alloc test spectrum");
        for (mz, intensity) in &sorted_peaks {
            spectrum.add_peak(*mz, *intensity).expect("add test peak");
        }
        SpectrumRecord {
            meta: SpectrumMetadata {
                id,
                label: format!("s{id}"),
                raw_name: format!("raw{id}"),
                feature_id: None,
                scans: None,
                filename: None,
                source_scan_usi: None,
                featurelist_feature_id: None,
                headers: BTreeMap::new(),
                precursor_mz: 100.0,
                num_peaks: peaks.len(),
            },
            peaks: Arc::new(peaks.to_vec()),
            spectrum: Arc::new(spectrum),
            payload: (),
        }
    }

    #[test]
    fn top_n_peak_filter_keeps_most_intense_peaks_for_similarity_only() {
        let spectra = vec![record(0, &[(50.0, 0.4), (10.0, 1.0), (30.0, 0.8), (20.0, 0.2)])];
        let processed = preprocess_spectra_for_metric(
            spectra,
            ComputeParams {
                metric: SimilarityMetric::CosineGreedy,
                tolerance: 0.2,
                mz_power: 0.0,
                intensity_power: 1.0,
                top_n_peaks: Some(2),
            },
        )
        .expect("preprocess");
        let reference = record(1, &[(10.0, 1.0), (30.0, 0.8)]);
        let scorer = MetricScorer::new(ComputeParams {
            metric: SimilarityMetric::CosineGreedy,
            tolerance: 0.2,
            mz_power: 0.0,
            intensity_power: 1.0,
            top_n_peaks: None,
        })
        .expect("scorer");
        let (score, matches) = scorer
            .similarity(processed[0].spectrum.as_ref(), reference.spectrum.as_ref(), 0, 1)
            .expect("similarity");

        assert_eq!(processed[0].peaks.as_ref().len(), 4);
        assert_eq!(matches, 2);
        assert!((score - 1.0).abs() < 1e-9);
    }
}
