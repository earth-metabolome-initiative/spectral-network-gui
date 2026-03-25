use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};

use mass_spectrometry::prelude::{
    GenericSpectrum, GreedyCosine, HungarianCosine, LinearEntropy, ModifiedGreedyCosine,
    ModifiedHungarianCosine, ModifiedLinearEntropy, MsEntropyCleanSpectrum, ScalarSimilarity,
    SiriusMergeClosePeaks, SpectralProcessor,
};

use crate::io::SpectrumRecord;
use crate::metadata::{LotusMetadataIndex, ResolvedLotusQuery, short_inchikey_from_record};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarityMetric {
    CosineHungarian,
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

#[derive(Clone, Debug)]
pub struct SearchParams {
    pub compute: ComputeParams,
    pub parent_mass_tolerance: f64,
    pub min_matched_peaks: usize,
    pub min_similarity_threshold: f64,
    pub top_n: usize,
    pub taxonomy: Option<SearchTaxonomyConfig>,
}

#[derive(Clone, Debug)]
pub struct SearchTaxonomyConfig {
    pub lotus: Arc<LotusMetadataIndex>,
    pub query: ResolvedLotusQuery,
}

enum MetricScorer {
    CosineHungarian(HungarianCosine<f64, f64>),
    CosineGreedy(GreedyCosine<f64, f64>),
    ModifiedCosine(ModifiedHungarianCosine<f64, f64>),
    ModifiedGreedyCosine(ModifiedGreedyCosine<f64, f64>),
    LinearEntropyWeighted(LinearEntropy<f64, f64>),
    LinearEntropyUnweighted(LinearEntropy<f64, f64>),
    ModifiedLinearEntropyWeighted(ModifiedLinearEntropy<f64, f64>),
    ModifiedLinearEntropyUnweighted(ModifiedLinearEntropy<f64, f64>),
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
            SimilarityMetric::LinearEntropyWeighted => LinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                true,
            )
            .map(Self::LinearEntropyWeighted)
            .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label())),
            SimilarityMetric::LinearEntropyUnweighted => LinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                false,
            )
            .map(Self::LinearEntropyUnweighted)
            .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label())),
            SimilarityMetric::ModifiedLinearEntropyWeighted => ModifiedLinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                true,
            )
            .map(Self::ModifiedLinearEntropyWeighted)
            .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label())),
            SimilarityMetric::ModifiedLinearEntropyUnweighted => ModifiedLinearEntropy::new(
                params.mz_power,
                params.intensity_power,
                params.tolerance,
                false,
            )
            .map(Self::ModifiedLinearEntropyUnweighted)
            .map_err(|err| format!("failed to configure {}: {err:?}", params.metric.label())),
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
            Self::LinearEntropyWeighted(sim) => sim.similarity(left, right),
            Self::LinearEntropyUnweighted(sim) => sim.similarity(left, right),
            Self::ModifiedLinearEntropyWeighted(sim) => sim.similarity(left, right),
            Self::ModifiedLinearEntropyUnweighted(sim) => sim.similarity(left, right),
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

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub query_index: usize,
    pub library_index: usize,
    pub rank: usize,
    pub spectral_score: f64,
    pub taxonomic_score: f64,
    pub combined_score: f64,
    pub matches: usize,
    pub matched_organism_name: Option<String>,
    pub matched_organism_wikidata: Option<String>,
    pub matched_shared_rank: Option<String>,
    pub matched_short_inchikey: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub taxonomic_reranking_applied: bool,
    pub taxonomic_query: Option<String>,
}

#[derive(Debug)]
pub enum SearchMessage {
    Finished(SearchResult),
    Cancelled,
    Failed(String),
}

pub struct NativeSearchHandle {
    total: usize,
    done: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<SearchMessage>,
}

impl NativeSearchHandle {
    pub fn total(&self) -> usize {
        self.total
    }

    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn try_recv(&self) -> Option<SearchMessage> {
        self.rx.try_recv().ok()
    }
}

#[derive(Clone, Debug)]
struct SearchCandidate {
    query_index: usize,
    library_index: usize,
    spectral_score: f64,
    taxonomic_score: f64,
    combined_score: f64,
    matches: usize,
    matched_organism_name: Option<String>,
    matched_organism_wikidata: Option<String>,
    matched_shared_rank: Option<String>,
    matched_short_inchikey: Option<String>,
}

pub fn total_pairs(n: usize) -> usize {
    n.saturating_mul(n.saturating_add(1)) / 2
}

pub fn total_search_pairs(query_count: usize, library_count: usize) -> usize {
    query_count.saturating_mul(library_count)
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
            let spectra = match preprocess_spectra_for_metric(spectra, params) {
                Ok(spectra) => spectra,
                Err(err) => {
                    let _ = tx.send(ComputeMessage::Failed(err));
                    return;
                }
            };

            let scorer = match MetricScorer::new(params) {
                Ok(sim) => sim,
                Err(err) => {
                    let _ = tx.send(ComputeMessage::Failed(err));
                    return;
                }
            };

            let done_worker = Arc::clone(&done_for_thread);
            let cancel_worker = Arc::clone(&cancel_for_thread);

            let pairs = match score_all_pairs(&spectra, params, &scorer, done_worker, cancel_worker)
            {
                Ok(Some(pairs)) => pairs,
                Ok(None) => {
                    let _ = tx.send(ComputeMessage::Cancelled);
                    return;
                }
                Err(err) => {
                    let _ = tx.send(ComputeMessage::Failed(err));
                    return;
                }
            };

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

pub fn start_native_search(
    queries: Vec<SpectrumRecord>,
    library: Vec<SpectrumRecord>,
    params: SearchParams,
) -> NativeSearchHandle {
    let total = total_search_pairs(queries.len(), library.len());
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
            let queries = match preprocess_spectra_for_metric(queries, params.compute) {
                Ok(spectra) => spectra,
                Err(err) => {
                    let _ = tx.send(SearchMessage::Failed(err));
                    return;
                }
            };
            let library = match preprocess_spectra_for_metric(library, params.compute) {
                Ok(spectra) => spectra,
                Err(err) => {
                    let _ = tx.send(SearchMessage::Failed(err));
                    return;
                }
            };

            let scorer = match MetricScorer::new(params.compute) {
                Ok(sim) => sim,
                Err(err) => {
                    let _ = tx.send(SearchMessage::Failed(err));
                    return;
                }
            };

            let done_worker = Arc::clone(&done_for_thread);
            let cancel_worker = Arc::clone(&cancel_for_thread);

            let result = match score_query_library_pairs(
                &queries,
                &library,
                &params,
                &scorer,
                done_worker,
                cancel_worker,
            ) {
                Ok(Some(result)) => result,
                Ok(None) => {
                    let _ = tx.send(SearchMessage::Cancelled);
                    return;
                }
                Err(err) => {
                    let _ = tx.send(SearchMessage::Failed(err));
                    return;
                }
            };

            let _ = tx.send(SearchMessage::Finished(result));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (queries, library, params);
            let _ = tx.send(SearchMessage::Failed(
                "native threaded search is unavailable on wasm".to_string(),
            ));
        }
    });

    NativeSearchHandle {
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
        let spectra = preprocess_spectra_for_metric(spectra, params)?;
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

pub enum IncrementalSearchStep {
    Progress,
    Finished(SearchResult),
    Cancelled,
}

pub struct IncrementalSearchState {
    params: SearchParams,
    scorer: MetricScorer,
    queries: Vec<SpectrumRecord>,
    library: Vec<SpectrumRecord>,
    query_index: usize,
    library_index: usize,
    total: usize,
    done: usize,
    cancel: bool,
    candidates: Vec<SearchCandidate>,
}

impl IncrementalSearchState {
    pub fn new(
        queries: Vec<SpectrumRecord>,
        library: Vec<SpectrumRecord>,
        params: SearchParams,
    ) -> Result<Self, String> {
        let queries = preprocess_spectra_for_metric(queries, params.compute)?;
        let library = preprocess_spectra_for_metric(library, params.compute)?;
        let total = total_search_pairs(queries.len(), library.len());
        let scorer = MetricScorer::new(params.compute)?;
        Ok(Self {
            params,
            scorer,
            queries,
            library,
            query_index: 0,
            library_index: 0,
            total,
            done: 0,
            cancel: false,
            candidates: Vec::new(),
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

    pub fn step(&mut self, budget: usize) -> Result<IncrementalSearchStep, String> {
        if self.cancel {
            return Ok(IncrementalSearchStep::Cancelled);
        }

        let query_count = self.queries.len();
        let library_count = self.library.len();
        if query_count == 0 || library_count == 0 || self.query_index >= query_count {
            return Ok(IncrementalSearchStep::Finished(SearchResult {
                hits: finalize_search_candidates(
                    std::mem::take(&mut self.candidates),
                    self.params.top_n,
                ),
                taxonomic_reranking_applied: self.params.taxonomy.is_some(),
                taxonomic_query: self
                    .params
                    .taxonomy
                    .as_ref()
                    .map(|config| config.query.query_label.clone()),
            }));
        }

        let mut remaining = budget.max(1);
        while remaining > 0 && self.query_index < query_count {
            if self.cancel {
                return Ok(IncrementalSearchStep::Cancelled);
            }

            if search_parent_mass_passes(
                &self.queries[self.query_index],
                &self.library[self.library_index],
                &self.params,
            ) {
                let left = self.queries[self.query_index].spectrum.as_ref();
                let right = self.library[self.library_index].spectrum.as_ref();
                let (score, matches) = self.scorer.similarity(
                    left,
                    right,
                    self.params.compute.metric,
                    self.query_index,
                    self.library_index,
                )?;
                if search_match_passes(score, matches, &self.params) {
                    self.candidates.push(build_search_candidate(
                        self.query_index,
                        self.library_index,
                        score,
                        matches,
                        &self.library[self.library_index],
                        self.params.taxonomy.as_ref(),
                    ));
                }
            }
            self.done += 1;

            self.library_index += 1;
            if self.library_index >= library_count {
                self.query_index += 1;
                self.library_index = 0;
            }
            remaining -= 1;
        }

        if self.query_index >= query_count {
            Ok(IncrementalSearchStep::Finished(SearchResult {
                hits: finalize_search_candidates(
                    std::mem::take(&mut self.candidates),
                    self.params.top_n,
                ),
                taxonomic_reranking_applied: self.params.taxonomy.is_some(),
                taxonomic_query: self
                    .params
                    .taxonomy
                    .as_ref()
                    .map(|config| config.query.query_label.clone()),
            }))
        } else {
            Ok(IncrementalSearchStep::Progress)
        }
    }
}

fn preprocess_spectra_for_metric(
    spectra: Vec<SpectrumRecord>,
    params: ComputeParams,
) -> Result<Vec<SpectrumRecord>, String> {
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

#[cfg(not(target_arch = "wasm32"))]
fn score_all_pairs(
    spectra: &[SpectrumRecord],
    params: ComputeParams,
    scorer: &MetricScorer,
    done_worker: Arc<AtomicUsize>,
    cancel_worker: Arc<AtomicBool>,
) -> Result<Option<Vec<PairScore>>, String> {
    use rayon::prelude::*;

    let total = total_pairs(spectra.len());
    let pair_indices: Vec<(usize, usize)> = (0..spectra.len())
        .flat_map(|i| (i..spectra.len()).map(move |j| (i, j)))
        .collect();
    debug_assert_eq!(pair_indices.len(), total);

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
        return Err(err);
    }
    if cancel_worker.load(Ordering::Relaxed) {
        return Ok(None);
    }
    Ok(Some(pairs))
}

#[cfg(not(target_arch = "wasm32"))]
fn score_query_library_pairs(
    queries: &[SpectrumRecord],
    library: &[SpectrumRecord],
    params: &SearchParams,
    scorer: &MetricScorer,
    done_worker: Arc<AtomicUsize>,
    cancel_worker: Arc<AtomicBool>,
) -> Result<Option<SearchResult>, String> {
    use rayon::prelude::*;

    let total = total_search_pairs(queries.len(), library.len());
    let error = Arc::new(Mutex::new(None::<String>));
    let error_worker = Arc::clone(&error);
    let done_for_iter = Arc::clone(&done_worker);
    let cancel_for_iter = Arc::clone(&cancel_worker);

    let candidates: Vec<SearchCandidate> = (0..total)
        .into_par_iter()
        .filter_map(|flat_idx| {
            if cancel_for_iter.load(Ordering::Relaxed) {
                return None;
            }

            let query_idx = flat_idx / library.len();
            let library_idx = flat_idx % library.len();
            done_for_iter.fetch_add(1, Ordering::Relaxed);
            if !search_parent_mass_passes(&queries[query_idx], &library[library_idx], params) {
                return None;
            }
            let left = queries[query_idx].spectrum.as_ref();
            let right = library[library_idx].spectrum.as_ref();
            match scorer.similarity(left, right, params.compute.metric, query_idx, library_idx) {
                Ok((score, matches)) => {
                    if search_match_passes(score, matches, params) {
                        Some(build_search_candidate(
                            query_idx,
                            library_idx,
                            score,
                            matches,
                            &library[library_idx],
                            params.taxonomy.as_ref(),
                        ))
                    } else {
                        None
                    }
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
        return Err(err);
    }
    if cancel_worker.load(Ordering::Relaxed) {
        return Ok(None);
    }
    Ok(Some(SearchResult {
        hits: finalize_search_candidates(candidates, params.top_n),
        taxonomic_reranking_applied: params.taxonomy.is_some(),
        taxonomic_query: params
            .taxonomy
            .as_ref()
            .map(|config| config.query.query_label.clone()),
    }))
}

fn search_match_passes(score: f64, matches: usize, params: &SearchParams) -> bool {
    matches >= params.min_matched_peaks && score >= params.min_similarity_threshold
}

fn search_parent_mass_passes(
    query: &SpectrumRecord,
    library: &SpectrumRecord,
    params: &SearchParams,
) -> bool {
    (query.meta.precursor_mz - library.meta.precursor_mz).abs() <= params.parent_mass_tolerance
}

fn build_search_candidate(
    query_index: usize,
    library_index: usize,
    spectral_score: f64,
    matches: usize,
    library_record: &SpectrumRecord,
    taxonomy: Option<&SearchTaxonomyConfig>,
) -> SearchCandidate {
    let (
        taxonomic_score,
        matched_organism_name,
        matched_organism_wikidata,
        matched_shared_rank,
        matched_short_inchikey,
    ) = if let Some(config) = taxonomy {
        let short_inchikey = short_inchikey_from_record(library_record);
        match short_inchikey
            .as_deref()
            .and_then(|short| config.lotus.match_candidate(short, &config.query.lineage))
        {
            Some(matched) => (
                f64::from(matched.score),
                matched.matched_organism_name,
                matched.matched_organism_wikidata,
                matched.shared_rank.map(|rank| rank.label().to_string()),
                matched.matched_short_inchikey,
            ),
            None => (0.0, None, None, None, short_inchikey),
        }
    } else {
        (0.0, None, None, None, None)
    };

    SearchCandidate {
        query_index,
        library_index,
        spectral_score,
        taxonomic_score,
        combined_score: spectral_score + taxonomic_score,
        matches,
        matched_organism_name,
        matched_organism_wikidata,
        matched_shared_rank,
        matched_short_inchikey,
    }
}

fn finalize_search_candidates(candidates: Vec<SearchCandidate>, top_n: usize) -> Vec<SearchHit> {
    let mut by_query: Vec<Vec<SearchCandidate>> = Vec::new();
    for candidate in candidates {
        if by_query.len() <= candidate.query_index {
            by_query.resize_with(candidate.query_index + 1, Vec::new);
        }
        by_query[candidate.query_index].push(candidate);
    }

    let keep = top_n.max(1);
    let mut hits = Vec::new();
    for (query_index, mut query_hits) in by_query.into_iter().enumerate() {
        query_hits.sort_by(|left, right| {
            right
                .combined_score
                .total_cmp(&left.combined_score)
                .then(right.spectral_score.total_cmp(&left.spectral_score))
                .then(right.matches.cmp(&left.matches))
                .then(left.library_index.cmp(&right.library_index))
        });
        for (rank, hit) in query_hits.into_iter().take(keep).enumerate() {
            hits.push(SearchHit {
                query_index,
                library_index: hit.library_index,
                rank: rank + 1,
                spectral_score: hit.spectral_score,
                taxonomic_score: hit.taxonomic_score,
                combined_score: hit.combined_score,
                matches: hit.matches,
                matched_organism_name: hit.matched_organism_name,
                matched_organism_wikidata: hit.matched_organism_wikidata,
                matched_shared_rank: hit.matched_shared_rank,
                matched_short_inchikey: hit.matched_short_inchikey,
            });
        }
    }
    hits.sort_by(|left, right| {
        left.query_index
            .cmp(&right.query_index)
            .then(left.rank.cmp(&right.rank))
    });
    hits
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

    use super::{
        ComputeParams, IncrementalComputeState, IncrementalSearchState, IncrementalSearchStep,
        IncrementalStep, SearchParams, SearchResult, SearchTaxonomyConfig, SimilarityMetric,
        finalize_search_candidates, start_native_search, total_pairs, total_search_pairs,
    };
    use crate::io::{SpectrumMeta, SpectrumRecord};
    use crate::metadata::load_lotus_bytes;

    fn spectrum(id: usize, precursor: f64, peaks: &[(f64, f64)]) -> SpectrumRecord {
        spectrum_with_headers(id, precursor, peaks, &[])
    }

    fn spectrum_with_headers(
        id: usize,
        precursor: f64,
        peaks: &[(f64, f64)],
        headers: &[(&str, &str)],
    ) -> SpectrumRecord {
        let mut spec = GenericSpectrum::<f64, f64>::with_capacity(precursor, peaks.len())
            .expect("failed to allocate spectrum");
        for (mz, intensity) in peaks {
            spec.add_peak(*mz, *intensity)
                .expect("failed to add test peak");
        }
        let mut header_map = BTreeMap::new();
        for (key, value) in headers {
            header_map.insert((*key).to_string(), (*value).to_string());
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
                headers: header_map,
                precursor_mz: precursor,
                num_peaks: peaks.len(),
            },
            peaks: Arc::new(peaks.to_vec()),
            spectrum: Arc::new(spec),
        }
    }

    fn base_compute_params(metric: SimilarityMetric) -> ComputeParams {
        ComputeParams {
            metric,
            tolerance: 0.2,
            mz_power: 0.0,
            intensity_power: 1.0,
        }
    }

    fn sample_taxonomy_config() -> SearchTaxonomyConfig {
        let csv = concat!(
            "structure_inchikey,organism_wikidata,organism_name,organism_taxonomy_01domain,organism_taxonomy_02kingdom,organism_taxonomy_03phylum,organism_taxonomy_04class,organism_taxonomy_05order,organism_taxonomy_06family,organism_taxonomy_07tribe,organism_taxonomy_08genus,organism_taxonomy_09species,organism_taxonomy_10varietas\n",
            "\"AAAAAAAAAAAAAA-111\",http://www.wikidata.org/entity/Q1,\"Withania somnifera\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania somnifera,NA\n",
            "\"BBBBBBBBBBBBBB-222\",http://www.wikidata.org/entity/Q2,\"Panax ginseng\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Apiales,Araliaceae,NA,Panax,Panax ginseng,NA\n"
        );
        let loaded = load_lotus_bytes("lotus.csv", csv.as_bytes()).expect("lotus");
        SearchTaxonomyConfig {
            lotus: loaded.index.clone(),
            query: loaded
                .index
                .resolve_query_lineage("Withania somnifera")
                .expect("query lineage"),
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
            base_compute_params(SimilarityMetric::CosineGreedy),
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
            base_compute_params(SimilarityMetric::CosineGreedy),
        )
        .expect("failed to create compute state");

        let _ = state.step(1).expect("first step failed");
        state.cancel();
        match state.step(100).expect("cancel step failed") {
            IncrementalStep::Cancelled => {}
            _ => panic!("expected cancelled state"),
        }
    }

    #[test]
    fn linear_entropy_preprocessing_handles_close_peaks() {
        let spectra = vec![
            spectrum(
                0,
                100.0,
                &[(10.0, 1.0), (10.01, 0.5), (20.0, 0.4), (30.0, 0.3)],
            ),
            spectrum(
                1,
                101.0,
                &[(10.0, 1.0), (10.01, 0.5), (20.0, 0.6), (31.0, 0.2)],
            ),
        ];

        let mut state = IncrementalComputeState::new(
            spectra,
            base_compute_params(SimilarityMetric::LinearEntropyWeighted),
        )
        .expect("failed to create compute state");

        loop {
            match state.step(8).expect("incremental step failed") {
                IncrementalStep::Progress => {}
                IncrementalStep::Finished(result) => {
                    assert_eq!(result.pairs.len(), total_pairs(2));
                    break;
                }
                IncrementalStep::Cancelled => panic!("unexpected cancel"),
            }
        }
    }

    #[test]
    fn incremental_search_applies_filters_and_top_n() {
        let queries = vec![
            spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
            spectrum(1, 200.0, &[(50.0, 1.0), (60.0, 0.9), (70.0, 0.6)]),
        ];
        let library = vec![
            spectrum(10, 101.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
            spectrum(11, 102.0, &[(10.0, 1.0), (20.1, 0.7), (31.0, 0.4)]),
            spectrum(12, 201.0, &[(50.0, 1.0), (60.0, 0.9), (71.0, 0.4)]),
        ];
        let mut state = IncrementalSearchState::new(
            queries,
            library,
            SearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 5.0,
                min_matched_peaks: 2,
                min_similarity_threshold: 0.1,
                top_n: 1,
                taxonomy: None,
            },
        )
        .expect("failed to create search state");

        let result = loop {
            match state.step(2).expect("search step failed") {
                IncrementalSearchStep::Progress => {}
                IncrementalSearchStep::Finished(result) => break result,
                IncrementalSearchStep::Cancelled => panic!("unexpected cancel"),
            }
        };

        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.hits[0].query_index, 0);
        assert_eq!(result.hits[0].library_index, 0);
        assert_eq!(result.hits[0].rank, 1);
        assert_eq!(result.hits[1].query_index, 1);
        assert_eq!(result.hits[1].library_index, 2);
        assert_eq!(result.hits[1].rank, 1);
        assert_eq!(state.total(), total_search_pairs(2, 3));
    }

    #[test]
    fn search_candidate_sorting_uses_score_matches_then_library_index() {
        let result = SearchResult {
            hits: finalize_search_candidates(
                vec![
                    super::SearchCandidate {
                        query_index: 0,
                        library_index: 2,
                        spectral_score: 0.8,
                        taxonomic_score: 0.0,
                        combined_score: 0.8,
                        matches: 4,
                        matched_organism_name: None,
                        matched_organism_wikidata: None,
                        matched_shared_rank: None,
                        matched_short_inchikey: None,
                    },
                    super::SearchCandidate {
                        query_index: 0,
                        library_index: 1,
                        spectral_score: 0.8,
                        taxonomic_score: 0.0,
                        combined_score: 0.8,
                        matches: 5,
                        matched_organism_name: None,
                        matched_organism_wikidata: None,
                        matched_shared_rank: None,
                        matched_short_inchikey: None,
                    },
                    super::SearchCandidate {
                        query_index: 0,
                        library_index: 0,
                        spectral_score: 0.8,
                        taxonomic_score: 0.0,
                        combined_score: 0.8,
                        matches: 5,
                        matched_organism_name: None,
                        matched_organism_wikidata: None,
                        matched_shared_rank: None,
                        matched_short_inchikey: None,
                    },
                ],
                3,
            ),
            taxonomic_reranking_applied: false,
            taxonomic_query: None,
        };

        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.library_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            result.hits.iter().map(|hit| hit.rank).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn incremental_search_respects_parent_mass_tolerance() {
        let queries = vec![spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)])];
        let library = vec![
            spectrum(10, 100.03, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
            spectrum(11, 100.20, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
        ];
        let mut state = IncrementalSearchState::new(
            queries,
            library,
            SearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 0.05,
                min_matched_peaks: 2,
                min_similarity_threshold: 0.1,
                top_n: 5,
                taxonomy: None,
            },
        )
        .expect("failed to create search state");

        let result = loop {
            match state.step(8).expect("search step failed") {
                IncrementalSearchStep::Progress => {}
                IncrementalSearchStep::Finished(result) => break result,
                IncrementalSearchStep::Cancelled => panic!("unexpected cancel"),
            }
        };

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].library_index, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_search_matches_incremental_result_shape() {
        let queries = vec![spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)])];
        let library = vec![
            spectrum(10, 101.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
            spectrum(11, 102.0, &[(10.0, 1.0), (20.1, 0.7), (31.0, 0.4)]),
        ];
        let params = SearchParams {
            compute: base_compute_params(SimilarityMetric::CosineGreedy),
            parent_mass_tolerance: 5.0,
            min_matched_peaks: 2,
            min_similarity_threshold: 0.1,
            top_n: 2,
            taxonomy: None,
        };

        let handle = start_native_search(queries.clone(), library.clone(), params.clone());
        let native = loop {
            if let Some(message) = handle.try_recv() {
                match message {
                    super::SearchMessage::Finished(result) => break result,
                    super::SearchMessage::Cancelled => panic!("unexpected cancel"),
                    super::SearchMessage::Failed(err) => panic!("native search failed: {err}"),
                }
            }
            std::thread::yield_now();
        };

        let mut incremental =
            IncrementalSearchState::new(queries, library, params).expect("search state");
        let incremental = loop {
            match incremental.step(32).expect("incremental step") {
                IncrementalSearchStep::Progress => {}
                IncrementalSearchStep::Finished(result) => break result,
                IncrementalSearchStep::Cancelled => panic!("unexpected cancel"),
            }
        };

        assert_eq!(native, incremental);
    }

    #[test]
    fn taxonomic_reranking_can_promote_lower_spectral_hit() {
        let queries = vec![spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)])];
        let library = vec![
            spectrum_with_headers(
                10,
                100.0,
                &[(10.0, 1.0), (21.0, 0.8), (35.0, 0.3)],
                &[("INCHIKEY", "AAAAAAAAAAAAAA-111")],
            ),
            spectrum_with_headers(
                11,
                100.0,
                &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)],
                &[("INCHIKEY", "BBBBBBBBBBBBBB-222")],
            ),
        ];
        let mut state = IncrementalSearchState::new(
            queries,
            library,
            SearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 0.1,
                min_matched_peaks: 1,
                min_similarity_threshold: 0.0,
                top_n: 2,
                taxonomy: Some(sample_taxonomy_config()),
            },
        )
        .expect("failed to create search state");

        let result = loop {
            match state.step(8).expect("search step failed") {
                IncrementalSearchStep::Progress => {}
                IncrementalSearchStep::Finished(result) => break result,
                IncrementalSearchStep::Cancelled => panic!("unexpected cancel"),
            }
        };

        assert!(result.taxonomic_reranking_applied);
        assert_eq!(result.hits[0].library_index, 0);
        assert_eq!(result.hits[0].taxonomic_score, 9.0);
        assert_eq!(
            result.hits[0].matched_organism_name.as_deref(),
            Some("Withania somnifera")
        );
        assert!(result.hits[0].combined_score > result.hits[1].combined_score);
    }
}
