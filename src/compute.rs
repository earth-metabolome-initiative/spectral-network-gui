use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};

use mass_spectrometry::prelude::{
    EntropySimilarity, GenericSpectrum, GreedyCosine, HungarianCosine, ModifiedGreedyCosine,
    ModifiedHungarianCosine, ScalarSimilarity,
};

use crate::io::SpectrumRecord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarityMetric {
    CosineHungarian,
    CosineGreedy,
    ModifiedCosine,
    ModifiedGreedyCosine,
    EntropySimilarityWeighted,
    EntropySimilarityUnweighted,
}

impl SimilarityMetric {
    pub const ALL: [Self; 6] = [
        Self::CosineHungarian,
        Self::CosineGreedy,
        Self::ModifiedCosine,
        Self::ModifiedGreedyCosine,
        Self::EntropySimilarityWeighted,
        Self::EntropySimilarityUnweighted,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::CosineHungarian => "CosineHungarian",
            Self::CosineGreedy => "CosineGreedy",
            Self::ModifiedCosine => "ModifiedCosine",
            Self::ModifiedGreedyCosine => "ModifiedGreedyCosine",
            Self::EntropySimilarityWeighted => "EntropySimilarityWeighted",
            Self::EntropySimilarityUnweighted => "EntropySimilarityUnweighted",
        }
    }
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::CosineGreedy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ComputeParams {
    pub metric: SimilarityMetric,
    pub tolerance: f64,
    pub mz_power: f64,
    pub intensity_power: f64,
}

enum MetricScorer {
    CosineHungarian(HungarianCosine<f64, f64>),
    CosineGreedy(GreedyCosine<f64, f64>),
    ModifiedCosine(ModifiedHungarianCosine<f64, f64>),
    ModifiedGreedyCosine(ModifiedGreedyCosine<f64, f64>),
    EntropySimilarityWeighted(EntropySimilarity<f64>),
    EntropySimilarityUnweighted(EntropySimilarity<f64>),
}

impl MetricScorer {
    fn new(params: ComputeParams) -> Result<Self, String> {
        match params.metric {
            SimilarityMetric::CosineHungarian => {
                HungarianCosine::new(params.mz_power, params.intensity_power, params.tolerance)
                    .map(Self::CosineHungarian)
                    .map_err(|err| {
                        format!("failed to configure {}: {err:?}", params.metric.label())
                    })
            }
            SimilarityMetric::CosineGreedy => {
                GreedyCosine::new(params.mz_power, params.intensity_power, params.tolerance)
                    .map(Self::CosineGreedy)
                    .map_err(|err| {
                        format!("failed to configure {}: {err:?}", params.metric.label())
                    })
            }
            SimilarityMetric::ModifiedCosine => ModifiedHungarianCosine::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
            )
            .map(Self::ModifiedCosine)
            .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label())),
            SimilarityMetric::ModifiedGreedyCosine => {
                ModifiedGreedyCosine::new(params.mz_power, params.intensity_power, params.tolerance)
                    .map(Self::ModifiedGreedyCosine)
                    .map_err(|err| {
                        format!("failed to configure {}: {err:?}", params.metric.label())
                    })
            }
            SimilarityMetric::EntropySimilarityWeighted => {
                EntropySimilarity::weighted(params.tolerance)
                    .map(Self::EntropySimilarityWeighted)
                    .map_err(|err| {
                        format!("failed to configure {}: {err:?}", params.metric.label())
                    })
            }
            SimilarityMetric::EntropySimilarityUnweighted => {
                EntropySimilarity::unweighted(params.tolerance)
                    .map(Self::EntropySimilarityUnweighted)
                    .map_err(|err| {
                        format!("failed to configure {}: {err:?}", params.metric.label())
                    })
            }
        }
    }

    fn similarity(
        &self,
        left: &GenericSpectrum<f64, f64>,
        right: &GenericSpectrum<f64, f64>,
        metric: SimilarityMetric,
        left_idx: usize,
        right_idx: usize,
    ) -> Result<(f64, usize), String> {
        let result = match self {
            Self::CosineHungarian(sim) => sim.similarity(left, right),
            Self::CosineGreedy(sim) => sim.similarity(left, right),
            Self::ModifiedCosine(sim) => sim.similarity(left, right),
            Self::ModifiedGreedyCosine(sim) => sim.similarity(left, right),
            Self::EntropySimilarityWeighted(sim) => sim.similarity(left, right),
            Self::EntropySimilarityUnweighted(sim) => sim.similarity(left, right),
        };
        result.map_err(|err| {
            format!(
                "{} failed for pair ({left_idx},{right_idx}): {err:?}",
                metric.label()
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct PairScore {
    pub left: usize,
    pub right: usize,
    pub score: f64,
    pub matches: usize,
}

#[derive(Clone, Debug)]
pub struct ComputeResult {
    pub pairs: Vec<PairScore>,
}

#[derive(Debug)]
pub enum ComputeMessage {
    Finished(ComputeResult),
    Cancelled,
    Failed(String),
}

pub struct NativeComputeHandle {
    total: usize,
    done: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<ComputeMessage>,
}

impl NativeComputeHandle {
    pub fn total(&self) -> usize {
        self.total
    }

    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn try_recv(&self) -> Option<ComputeMessage> {
        self.rx.try_recv().ok()
    }
}

pub fn total_pairs(n: usize) -> usize {
    n.saturating_mul(n.saturating_add(1)) / 2
}

pub fn start_native_compute(
    spectra: Vec<SpectrumRecord>,
    params: ComputeParams,
) -> NativeComputeHandle {
    let total = total_pairs(spectra.len());
    let done = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    let done_for_thread = Arc::clone(&done);
    #[cfg(not(target_arch = "wasm32"))]
    let cancel_for_thread = Arc::clone(&cancel);

    std::thread::spawn(move || {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let done_worker = Arc::clone(&done_for_thread);
            let cancel_worker = Arc::clone(&cancel_for_thread);
            use rayon::prelude::*;

            let scorer = match MetricScorer::new(params) {
                Ok(sim) => sim,
                Err(err) => {
                    let _ = tx.send(ComputeMessage::Failed(err));
                    return;
                }
            };

            let mut pair_indices = Vec::with_capacity(total);
            for i in 0..spectra.len() {
                for j in i..spectra.len() {
                    pair_indices.push((i, j));
                }
            }

            let error = Arc::new(Mutex::new(None::<String>));
            let error_worker = Arc::clone(&error);
            let cancel_for_iter = Arc::clone(&cancel_worker);
            let done_for_iter = Arc::clone(&done_worker);

            let pairs: Vec<PairScore> = pair_indices
                .into_par_iter()
                .filter_map(|(i, j)| {
                    if cancel_for_iter.load(Ordering::Relaxed) {
                        return None;
                    }

                    let left = spectra[i].spectrum.as_ref();
                    let right = spectra[j].spectrum.as_ref();
                    match scorer.similarity(left, right, params.metric, i, j) {
                        Ok((score, matches)) => {
                            done_for_iter.fetch_add(1, Ordering::Relaxed);
                            Some(PairScore {
                                left: i,
                                right: j,
                                score,
                                matches,
                            })
                        }
                        Err(err) => {
                            if let Ok(mut slot) = error_worker.lock()
                                && slot.is_none()
                            {
                                *slot = Some(err);
                            }
                            cancel_for_iter.store(true, Ordering::Relaxed);
                            None
                        }
                    }
                })
                .collect();

            if let Ok(mut slot) = error.lock()
                && let Some(err) = slot.take()
            {
                let _ = tx.send(ComputeMessage::Failed(err));
                return;
            }

            if cancel_worker.load(Ordering::Relaxed) {
                let _ = tx.send(ComputeMessage::Cancelled);
                return;
            }

            let _ = tx.send(ComputeMessage::Finished(ComputeResult { pairs }));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (spectra, params);
            let _ = tx.send(ComputeMessage::Failed(
                "native threaded compute is unavailable on wasm".to_string(),
            ));
        }
    });

    NativeComputeHandle {
        total,
        done,
        cancel,
        rx,
    }
}

pub enum IncrementalStep {
    Progress,
    Finished(ComputeResult),
    Cancelled,
}

pub struct IncrementalComputeState {
    params: ComputeParams,
    scorer: MetricScorer,
    spectra: Vec<SpectrumRecord>,
    i: usize,
    j: usize,
    total: usize,
    done: usize,
    cancel: bool,
    pairs: Vec<PairScore>,
}

impl IncrementalComputeState {
    pub fn new(spectra: Vec<SpectrumRecord>, params: ComputeParams) -> Result<Self, String> {
        let total = total_pairs(spectra.len());
        let scorer = MetricScorer::new(params)?;
        Ok(Self {
            params,
            scorer,
            spectra,
            i: 0,
            j: 0,
            total,
            done: 0,
            cancel: false,
            pairs: Vec::with_capacity(total),
        })
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn done(&self) -> usize {
        self.done
    }

    pub fn cancel(&mut self) {
        self.cancel = true;
    }

    pub fn step(&mut self, budget: usize) -> Result<IncrementalStep, String> {
        if self.cancel {
            return Ok(IncrementalStep::Cancelled);
        }

        let n = self.spectra.len();
        if n == 0 || self.i >= n {
            let result = ComputeResult {
                pairs: std::mem::take(&mut self.pairs),
            };
            return Ok(IncrementalStep::Finished(result));
        }

        let mut remaining = budget.max(1);
        while remaining > 0 && self.i < n {
            if self.cancel {
                return Ok(IncrementalStep::Cancelled);
            }
            let left = self.spectra[self.i].spectrum.as_ref();
            let right = self.spectra[self.j].spectrum.as_ref();
            let (score, matches) =
                self.scorer
                    .similarity(left, right, self.params.metric, self.i, self.j)?;
            self.pairs.push(PairScore {
                left: self.i,
                right: self.j,
                score,
                matches,
            });
            self.done += 1;

            self.j += 1;
            if self.j >= n {
                self.i += 1;
                self.j = self.i;
            }
            remaining -= 1;
        }

        if self.i >= n {
            let result = ComputeResult {
                pairs: std::mem::take(&mut self.pairs),
            };
            Ok(IncrementalStep::Finished(result))
        } else {
            Ok(IncrementalStep::Progress)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

    use super::{
        ComputeParams, IncrementalComputeState, IncrementalStep, SimilarityMetric, total_pairs,
    };
    use crate::io::{SpectrumMeta, SpectrumRecord};

    fn spectrum(id: usize, precursor: f64, peaks: &[(f64, f64)]) -> SpectrumRecord {
        let mut spec = GenericSpectrum::<f64, f64>::with_capacity(precursor, peaks.len())
            .expect("failed to allocate spectrum");
        for (mz, intensity) in peaks {
            spec.add_peak(*mz, *intensity)
                .expect("failed to add test peak");
        }
        SpectrumRecord {
            meta: SpectrumMeta {
                id,
                label: format!("s{id}"),
                raw_name: format!("s{id}"),
                feature_id: None,
                scans: None,
                filename: None,
                source_scan_usi: None,
                featurelist_feature_id: None,
                precursor_mz: precursor,
                num_peaks: peaks.len(),
            },
            spectrum: Arc::new(spec),
        }
    }

    #[test]
    fn computes_expected_pair_count_incrementally() {
        let spectra = vec![
            spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.5)]),
            spectrum(1, 101.0, &[(10.0, 1.0), (20.1, 0.5)]),
            spectrum(2, 102.0, &[(50.0, 1.0), (60.0, 0.5)]),
        ];

        let mut state = IncrementalComputeState::new(
            spectra,
            ComputeParams {
                metric: SimilarityMetric::CosineGreedy,
                tolerance: 0.2,
                mz_power: 0.0,
                intensity_power: 1.0,
            },
        )
        .expect("failed to create compute state");

        loop {
            match state.step(2).expect("incremental step failed") {
                IncrementalStep::Progress => {}
                IncrementalStep::Finished(result) => {
                    assert_eq!(result.pairs.len(), total_pairs(3));
                    break;
                }
                IncrementalStep::Cancelled => panic!("unexpected cancel"),
            }
        }
    }

    #[test]
    fn can_cancel_incremental_compute() {
        let spectra = vec![
            spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.5)]),
            spectrum(1, 101.0, &[(10.0, 1.0), (20.1, 0.5)]),
            spectrum(2, 102.0, &[(50.0, 1.0), (60.0, 0.5)]),
            spectrum(3, 103.0, &[(51.0, 1.0), (61.0, 0.5)]),
        ];

        let mut state = IncrementalComputeState::new(
            spectra,
            ComputeParams {
                metric: SimilarityMetric::CosineGreedy,
                tolerance: 0.2,
                mz_power: 0.0,
                intensity_power: 1.0,
            },
        )
        .expect("failed to create compute state");

        let _ = state.step(1).expect("first step failed");
        state.cancel();
        match state.step(100).expect("cancel step failed") {
            IncrementalStep::Cancelled => {}
            _ => panic!("expected cancelled state"),
        }
    }
}
