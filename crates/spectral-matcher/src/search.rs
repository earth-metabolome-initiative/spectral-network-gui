use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Reverse;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::{BinaryHeap, HashMap};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::api::{
    JobProgressStage, NetworkArtifact, NetworkRequest, NetworkSpectrum, SearchArtifact,
    SearchArtifactHit, SearchArtifactResult, SearchRequest,
};
use crate::export::{SearchQueryKey, export_search_tsv};
use crate::mgf::load_mgf_bytes;
#[cfg(not(target_arch = "wasm32"))]
use crate::mgf::load_mgf_path;
use crate::model::{CandidateHit, LoadedSpectra, SearchResult, SpectrumRecord};
use crate::network::{PairScore, SelectedNeighbor, build_network_from_selected_neighbors};
use crate::similarity::{ComputeParams, MetricScorer, preprocess_spectra_for_metric};
use crate::taxonomy::{LotusMetadataIndex, ResolvedLotusQuery, short_inchikey_from_record};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LibrarySearchParams {
    pub compute: ComputeParams,
    pub parent_mass_tolerance: f64,
    pub min_matched_peaks: usize,
    pub min_similarity_threshold: f64,
    pub top_n: usize,
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

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug)]
struct SearchTaxonomyConfig {
    lotus: LotusMetadataIndex,
    query: ResolvedLotusQuery,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct FileFingerprint {
    len: u64,
    modified: Option<(u64, u32)>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct CachedMgf {
    fingerprint: FileFingerprint,
    min_peaks: usize,
    max_peaks: usize,
    loaded: LoadedSpectra,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct RankedNeighbor {
    neighbor: usize,
    score: f64,
    matches: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl PartialEq for RankedNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.neighbor == other.neighbor
            && self.matches == other.matches
            && self.score.to_bits() == other.score.to_bits()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Eq for RankedNeighbor {}

#[cfg(not(target_arch = "wasm32"))]
impl PartialOrd for RankedNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Ord for RankedNeighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.neighbor.cmp(&self.neighbor))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mgf_cache() -> &'static Mutex<HashMap<PathBuf, CachedMgf>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedMgf>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn total_search_pairs(query_count: usize, library_count: usize) -> usize {
    query_count.saturating_mul(library_count)
}

pub fn total_pairs(count: usize) -> usize {
    count.saturating_mul(count.saturating_add(1)) / 2
}

#[cfg(not(target_arch = "wasm32"))]
fn total_network_pairs(count: usize) -> usize {
    count.saturating_mul(count.saturating_sub(1)) / 2
}

fn load_search_taxonomy_config(
    request: &crate::api::SearchTaxonomyRequest,
) -> Result<SearchTaxonomyConfig, String> {
    let lotus = if let Some(text) = request.lotus_csv_text.as_deref() {
        crate::taxonomy::load_lotus_bytes(text.as_bytes())?
    } else if let Some(path) = request.lotus_csv_path.as_deref() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::taxonomy::load_lotus_path(std::path::Path::new(path))?
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Err("LOTUS CSV path loading is unavailable on wasm".to_string());
        }
    } else {
        return Err(format!(
            "LOTUS source '{}' is missing CSV text and path",
            request.lotus_source_label
        ));
    };

    let query = lotus
        .resolve_query_lineage(&request.query_text)
        .ok_or_else(|| format!("Biosource not found in LOTUS: {}", request.query_text.trim()))?;

    Ok(SearchTaxonomyConfig { lotus, query })
}

fn effective_search_params(
    params: &LibrarySearchParams,
    library_count: usize,
    taxonomy_enabled: bool,
) -> LibrarySearchParams {
    let mut effective = params.clone();
    if taxonomy_enabled {
        effective.top_n = library_count.max(1);
    }
    effective
}

pub fn search_library(
    queries: Vec<SpectrumRecord>,
    library: Vec<SpectrumRecord>,
    params: LibrarySearchParams,
) -> Result<SearchResult, String> {
    let query_count = queries.len();
    let library_count = library.len();
    let queries = preprocess_spectra_for_metric(queries, params.compute)?;
    let library = preprocess_spectra_for_metric(library, params.compute)?;
    let scorer = MetricScorer::new(params.compute)?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let total = total_search_pairs(query_count, library_count);
        let done = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let hits = score_query_library_pairs(
            &queries,
            &library,
            &params,
            &scorer,
            total,
            done,
            cancel,
            &|_, _, _| {},
            &|| false,
        )?
        .ok_or_else(|| "search was unexpectedly cancelled".to_string())?;
        Ok(SearchResult {
            hits,
            query_count,
            library_count,
            metric: params.compute.metric,
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let candidates = search_candidates(&queries, &library, &params, &scorer)?;
        Ok(SearchResult {
            hits: finalize_search_candidates(candidates, params.top_n),
            query_count,
            library_count,
            metric: params.compute.metric,
        })
    }
}

pub fn build_network_artifact(request: NetworkRequest) -> Result<NetworkArtifact, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        return build_network_artifact_with_progress(request, |_, _, _| {}, || false);
    }

    #[cfg(target_arch = "wasm32")]
    {
    let loaded = load_request_spectra(
        &request.source_label,
        request.mgf_text.as_deref(),
        request.mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    let spectra = preprocess_spectra_for_metric(loaded.spectra.clone(), request.build.compute)?;
    let scorer = MetricScorer::new(request.build.compute)?;
    #[cfg(not(target_arch = "wasm32"))]
    let pair_scores = score_all_pairs_parallel(&spectra, &scorer)?;
    #[cfg(target_arch = "wasm32")]
    let pair_scores = score_all_pairs(&spectra, &scorer)?;
    let metas = loaded
        .spectra
        .iter()
        .map(|record| record.meta.clone())
        .collect::<Vec<_>>();
    let network = build_network(
        &metas,
        &pair_scores,
        request.build.threshold.clamp(0.0, 1.0),
        request.build.top_k.max(1),
    );
    let spectra = loaded
        .spectra
        .iter()
        .map(|record| NetworkSpectrum {
            meta: record.meta.clone(),
            peaks: record.peaks.as_ref().clone(),
        })
        .collect();
    Ok(NetworkArtifact {
        source_label: loaded.source_label,
        parse_stats: loaded.stats,
        build: request.build,
        spectra,
        network,
    })
    }
}

pub fn run_search_request(request: SearchRequest) -> Result<SearchArtifact, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        return run_search_request_with_progress(request, |_, _, _| {}, || false);
    }

    #[cfg(target_arch = "wasm32")]
    {
    let queries = load_request_spectra(
        &request.query_source_label,
        request.query_mgf_text.as_deref(),
        request.query_mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    let library = load_request_spectra(
        &request.library_source_label,
        request.library_mgf_text.as_deref(),
        request.library_mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    let taxonomy = request
        .taxonomy
        .as_ref()
        .map(load_search_taxonomy_config)
        .transpose()?;
    let search_params =
        effective_search_params(&request.search, library.spectra.len(), taxonomy.is_some());
    let query_key = request.query_key.unwrap_or(SearchQueryKey::FeatureId);
    let result = search_library(
        queries.spectra.clone(),
        library.spectra.clone(),
        search_params,
    )?;
    let enriched = build_search_artifact_result(
        result,
        &library.spectra,
        taxonomy.as_ref(),
        request.search.top_n,
    );
    let tsv = export_search_tsv(&enriched, &queries.spectra, &library.spectra, query_key);
    Ok(SearchArtifact {
        query_source_label: queries.source_label,
        library_source_label: library.source_label,
        query_stats: queries.stats,
        library_stats: library.stats,
        search: request.search,
        taxonomy: request.taxonomy,
        query_key,
        query_spectra: queries
            .spectra
            .iter()
            .map(|record| record.meta.clone())
            .collect(),
        library_spectra: library
            .spectra
            .iter()
            .map(|record| record.meta.clone())
            .collect(),
        result: enriched,
        tsv,
    })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn build_network_artifact_with_progress<F>(
    request: NetworkRequest,
    on_progress: F,
    is_cancelled: impl Fn() -> bool + Send + Sync,
) -> Result<NetworkArtifact, String>
where
    F: Fn(JobProgressStage, u64, u64) + Send + Sync,
{
    on_progress(JobProgressStage::LoadingSpectra, 0, 1);
    let loaded = load_request_spectra(
        &request.source_label,
        request.mgf_text.as_deref(),
        request.mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    on_progress(JobProgressStage::LoadingSpectra, 1, 1);

    let spectra = preprocess_spectra_for_metric(loaded.spectra.clone(), request.build.compute)?;
    let scorer = MetricScorer::new(request.build.compute)?;
    let total = total_network_pairs(spectra.len()) as u64;
    let done = Arc::new(AtomicUsize::new(0));
    on_progress(JobProgressStage::Scoring, 0, total);
    let selected_neighbors = score_network_neighbors_parallel(
        &spectra,
        &scorer,
        request.build.threshold.clamp(0.0, 1.0),
        request.build.top_k.max(1),
        total as usize,
        Arc::clone(&done),
        &on_progress,
        &is_cancelled,
    )?;
    on_progress(
        JobProgressStage::Scoring,
        done.load(Ordering::Relaxed) as u64,
        total,
    );
    on_progress(JobProgressStage::BuildingNetwork, 0, 1);

    let metas = loaded
        .spectra
        .iter()
        .map(|record| record.meta.clone())
        .collect::<Vec<_>>();
    let network = build_network_from_selected_neighbors(&metas, &selected_neighbors);
    let spectra = loaded
        .spectra
        .iter()
        .map(|record| NetworkSpectrum {
            meta: record.meta.clone(),
            peaks: record.peaks.as_ref().clone(),
        })
        .collect();
    on_progress(JobProgressStage::Finalizing, 1, 1);
    Ok(NetworkArtifact {
        source_label: loaded.source_label,
        parse_stats: loaded.stats,
        build: request.build,
        spectra,
        network,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_search_request_with_progress<F>(
    request: SearchRequest,
    on_progress: F,
    is_cancelled: impl Fn() -> bool + Send + Sync,
) -> Result<SearchArtifact, String>
where
    F: Fn(JobProgressStage, u64, u64) + Send + Sync,
{
    on_progress(JobProgressStage::LoadingQuery, 0, 1);
    let queries = load_request_spectra(
        &request.query_source_label,
        request.query_mgf_text.as_deref(),
        request.query_mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    on_progress(JobProgressStage::LoadingQuery, 1, 1);

    on_progress(JobProgressStage::LoadingLibrary, 0, 1);
    let library = load_request_spectra(
        &request.library_source_label,
        request.library_mgf_text.as_deref(),
        request.library_mgf_path.as_deref(),
        request.parse.min_peaks,
        request.parse.max_peaks,
    )?;
    on_progress(JobProgressStage::LoadingLibrary, 1, 1);

    let queries_processed =
        preprocess_spectra_for_metric(queries.spectra.clone(), request.search.compute)?;
    let library_processed =
        preprocess_spectra_for_metric(library.spectra.clone(), request.search.compute)?;
    let taxonomy = request
        .taxonomy
        .as_ref()
        .map(load_search_taxonomy_config)
        .transpose()?;
    let search_params =
        effective_search_params(&request.search, library_processed.len(), taxonomy.is_some());
    let scorer = MetricScorer::new(search_params.compute)?;
    let total = total_search_pairs(queries_processed.len(), library_processed.len());
    let done = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    on_progress(JobProgressStage::Scoring, 0, total as u64);
    let hits = score_query_library_pairs(
        &queries_processed,
        &library_processed,
        &search_params,
        &scorer,
        total,
        Arc::clone(&done),
        cancel,
        &on_progress,
        &is_cancelled,
    )?
    .ok_or_else(|| "search was unexpectedly cancelled".to_string())?;
    on_progress(
        JobProgressStage::Scoring,
        done.load(Ordering::Relaxed) as u64,
        total as u64,
    );

    on_progress(JobProgressStage::Finalizing, 0, 1);
    let query_key = request.query_key.unwrap_or(SearchQueryKey::FeatureId);
    let result = SearchResult {
        hits,
        query_count: queries.spectra.len(),
        library_count: library.spectra.len(),
        metric: request.search.compute.metric,
    };
    let enriched = build_search_artifact_result(
        result,
        &library.spectra,
        taxonomy.as_ref(),
        request.search.top_n,
    );
    let tsv = export_search_tsv(&enriched, &queries.spectra, &library.spectra, query_key);
    on_progress(JobProgressStage::Finalizing, 1, 1);
    Ok(SearchArtifact {
        query_source_label: queries.source_label,
        library_source_label: library.source_label,
        query_stats: queries.stats,
        library_stats: library.stats,
        search: request.search,
        taxonomy: request.taxonomy,
        query_key,
        query_spectra: queries
            .spectra
            .iter()
            .map(|record| record.meta.clone())
            .collect(),
        library_spectra: library
            .spectra
            .iter()
            .map(|record| record.meta.clone())
            .collect(),
        result: enriched,
        tsv,
    })
}

fn load_request_spectra(
    source_label: &str,
    mgf_text: Option<&str>,
    mgf_path: Option<&str>,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<LoadedSpectra, String> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = mgf_path {
        return load_mgf_path_cached(Path::new(path), min_peaks, max_peaks);
    }

    if let Some(text) = mgf_text {
        return load_mgf_bytes(source_label, text.as_bytes(), min_peaks, max_peaks);
    }

    #[cfg(target_arch = "wasm32")]
    if mgf_path.is_some() {
        return Err("MGF path requests are unavailable on wasm; provide inline MGF text".to_string());
    }

    Err("request did not provide an MGF source".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_mgf_path_cached(path: &Path, min_peaks: usize, max_peaks: usize) -> Result<LoadedSpectra, String> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|err| format!("cannot resolve {}: {err}", path.display()))?;
    let fingerprint = file_fingerprint(&canonical)?;

    if let Ok(cache) = mgf_cache().lock()
        && let Some(cached) = cache.get(&canonical)
        && cached.fingerprint == fingerprint
        && cached.min_peaks == min_peaks
        && cached.max_peaks == max_peaks
    {
        return Ok(cached.loaded.clone());
    }

    let loaded = load_mgf_path(&canonical, min_peaks, max_peaks)?;
    if let Ok(mut cache) = mgf_cache().lock() {
        cache.insert(
            canonical,
            CachedMgf {
                fingerprint,
                min_peaks,
                max_peaks,
                loaded: loaded.clone(),
            },
        );
    }
    Ok(loaded)
}

#[cfg(not(target_arch = "wasm32"))]
fn file_fingerprint(path: &Path) -> Result<FileFingerprint, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| format!("cannot stat {}: {err}", path.display()))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs(), duration.subsec_nanos()));
    Ok(FileFingerprint {
        len: metadata.len(),
        modified,
    })
}

pub fn start_native_search(
    queries: Vec<SpectrumRecord>,
    library: Vec<SpectrumRecord>,
    params: LibrarySearchParams,
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
            let query_count = queries.len();
            let library_count = library.len();

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

            let total = total_search_pairs(query_count, library_count);
            let result = match score_query_library_pairs(
                &queries,
                &library,
                &params,
                &scorer,
                total,
                done_for_thread,
                cancel_for_thread,
                &|_, _, _| {},
                &|| false,
            ) {
                Ok(Some(hits)) => SearchResult {
                    hits,
                    query_count,
                    library_count,
                    metric: params.compute.metric,
                },
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

pub enum IncrementalSearchStep {
    Progress,
    Finished(SearchResult),
    Cancelled,
}

pub struct IncrementalSearchState {
    params: LibrarySearchParams,
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
        params: LibrarySearchParams,
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
                query_count,
                library_count,
                metric: self.params.compute.metric,
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
                let (score, matches) =
                    self.scorer
                        .similarity(left, right, self.query_index, self.library_index)?;
                if search_match_passes(score, matches, &self.params) {
                    self.candidates.push(SearchCandidate {
                        query_index: self.query_index,
                        library_index: self.library_index,
                        spectral_score: score,
                        taxonomic_score: 0.0,
                        combined_score: score,
                        matches,
                        matched_organism_name: None,
                        matched_organism_wikidata: None,
                        matched_shared_rank: None,
                        matched_short_inchikey: None,
                    });
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
                query_count,
                library_count,
                metric: self.params.compute.metric,
            }))
        } else {
            Ok(IncrementalSearchStep::Progress)
        }
    }
}

fn search_candidates(
    queries: &[SpectrumRecord],
    library: &[SpectrumRecord],
    params: &LibrarySearchParams,
    scorer: &MetricScorer,
) -> Result<Vec<SearchCandidate>, String> {
    let mut candidates = Vec::new();
    for query_idx in 0..queries.len() {
        for library_idx in 0..library.len() {
            if !search_parent_mass_passes(&queries[query_idx], &library[library_idx], params) {
                continue;
            }
            let left = queries[query_idx].spectrum.as_ref();
            let right = library[library_idx].spectrum.as_ref();
            let (score, matches) = scorer.similarity(left, right, query_idx, library_idx)?;
            if search_match_passes(score, matches, params) {
                candidates.push(SearchCandidate {
                    query_index: query_idx,
                    library_index: library_idx,
                    spectral_score: score,
                    taxonomic_score: 0.0,
                    combined_score: score,
                    matches,
                    matched_organism_name: None,
                    matched_organism_wikidata: None,
                    matched_shared_rank: None,
                    matched_short_inchikey: None,
                });
            }
        }
    }
    Ok(candidates)
}

#[cfg(not(target_arch = "wasm32"))]
fn score_query_library_pairs(
    queries: &[SpectrumRecord],
    library: &[SpectrumRecord],
    params: &LibrarySearchParams,
    scorer: &MetricScorer,
    total: usize,
    done_worker: Arc<AtomicUsize>,
    cancel_worker: Arc<AtomicBool>,
    on_progress: &(impl Fn(JobProgressStage, u64, u64) + Sync),
    is_cancelled: &(impl Fn() -> bool + Sync),
) -> Result<Option<Vec<CandidateHit>>, String> {
    let error = Arc::new(Mutex::new(None::<String>));
    let error_worker = Arc::clone(&error);
    let done_for_iter = Arc::clone(&done_worker);
    let cancel_for_iter = Arc::clone(&cancel_worker);
    let report_every = progress_report_every(total);

    let candidates: Vec<SearchCandidate> = (0..total)
        .into_par_iter()
        .filter_map(|flat_idx| {
            if cancel_for_iter.load(Ordering::Relaxed) || is_cancelled() {
                cancel_for_iter.store(true, Ordering::Relaxed);
                return None;
            }

            let query_idx = flat_idx / library.len();
            let library_idx = flat_idx % library.len();
            let completed = done_for_iter.fetch_add(1, Ordering::Relaxed) + 1;
            if completed % report_every == 0 || completed == total {
                on_progress(JobProgressStage::Scoring, completed as u64, total as u64);
            }
            if !search_parent_mass_passes(&queries[query_idx], &library[library_idx], params) {
                return None;
            }
            let left = queries[query_idx].spectrum.as_ref();
            let right = library[library_idx].spectrum.as_ref();
            match scorer.similarity(left, right, query_idx, library_idx) {
                Ok((score, matches)) => search_match_passes(score, matches, params).then_some(
                    SearchCandidate {
                        query_index: query_idx,
                        library_index: library_idx,
                        spectral_score: score,
                        taxonomic_score: 0.0,
                        combined_score: score,
                        matches,
                        matched_organism_name: None,
                        matched_organism_wikidata: None,
                        matched_shared_rank: None,
                        matched_short_inchikey: None,
                    },
                ),
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
    Ok(Some(finalize_search_candidates(candidates, params.top_n)))
}

fn score_all_pairs(
    spectra: &[SpectrumRecord],
    scorer: &MetricScorer,
) -> Result<Vec<PairScore>, String> {
    let mut pairs = Vec::with_capacity(total_pairs(spectra.len()));
    for left_index in 0..spectra.len() {
        for right_index in left_index..spectra.len() {
            let left = spectra[left_index].spectrum.as_ref();
            let right = spectra[right_index].spectrum.as_ref();
            let (score, matches) = scorer.similarity(left, right, left_index, right_index)?;
            pairs.push(PairScore {
                left: left_index,
                right: right_index,
                score,
                matches,
            });
        }
    }
    Ok(pairs)
}

#[cfg(not(target_arch = "wasm32"))]
fn score_network_neighbors_parallel(
    spectra: &[SpectrumRecord],
    scorer: &MetricScorer,
    threshold: f64,
    top_k: usize,
    total: usize,
    done_worker: Arc<AtomicUsize>,
    on_progress: &(impl Fn(JobProgressStage, u64, u64) + Sync),
    is_cancelled: &(impl Fn() -> bool + Sync),
) -> Result<Vec<Vec<SelectedNeighbor>>, String> {
    let threshold = threshold.clamp(0.0, 1.0);
    let top_k = top_k.max(1);
    let report_every = progress_report_every(total);
    let neighbors: Vec<Mutex<BinaryHeap<Reverse<RankedNeighbor>>>> = (0..spectra.len())
        .map(|_| Mutex::new(BinaryHeap::new()))
        .collect();
    let error = Arc::new(Mutex::new(None::<String>));
    let error_worker = Arc::clone(&error);

    (0..spectra.len()).into_par_iter().for_each(|left_index| {
        if is_cancelled() {
            if let Ok(mut slot) = error_worker.lock()
                && slot.is_none()
            {
                *slot = Some("job cancelled".to_string());
            }
            return;
        }
        let mut local_heap: BinaryHeap<Reverse<RankedNeighbor>> = BinaryHeap::new();
        let left = spectra[left_index].spectrum.as_ref();

        for right_index in (left_index + 1)..spectra.len() {
            if is_cancelled() {
                if let Ok(mut slot) = error_worker.lock()
                    && slot.is_none()
                {
                    *slot = Some("job cancelled".to_string());
                }
                return;
            }
            let completed = done_worker.fetch_add(1, Ordering::Relaxed) + 1;
            if completed % report_every == 0 || completed == total {
                on_progress(JobProgressStage::Scoring, completed as u64, total as u64);
            }
            let right = spectra[right_index].spectrum.as_ref();
            match scorer.similarity(left, right, left_index, right_index) {
                Ok((score, matches)) => {
                    if score < threshold {
                        continue;
                    }
                    let left_candidate = RankedNeighbor {
                        neighbor: right_index,
                        score,
                        matches,
                    };
                    let right_candidate = RankedNeighbor {
                        neighbor: left_index,
                        score,
                        matches,
                    };
                    push_ranked_neighbor(&mut local_heap, left_candidate, top_k);
                    if let Ok(mut heap) = neighbors[right_index].lock() {
                        push_ranked_neighbor(&mut heap, right_candidate, top_k);
                    }
                }
                Err(err) => {
                    if let Ok(mut slot) = error_worker.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                }
            }
        }

        if let Ok(mut heap) = neighbors[left_index].lock() {
            for ranked in local_heap.into_iter().map(|entry| entry.0) {
                push_ranked_neighbor(&mut heap, ranked, top_k);
            }
        }
    });

    if let Ok(mut slot) = error.lock()
        && let Some(err) = slot.take()
    {
        return Err(err);
    }

    Ok(neighbors
        .into_iter()
        .map(|heap| ranked_heap_to_selected_neighbors(heap, top_k))
        .collect())
}

#[cfg(not(target_arch = "wasm32"))]
fn push_ranked_neighbor(
    heap: &mut BinaryHeap<Reverse<RankedNeighbor>>,
    candidate: RankedNeighbor,
    top_k: usize,
) {
    if heap.len() < top_k {
        heap.push(Reverse(candidate));
        return;
    }

    if heap.peek().is_some_and(|worst| candidate > worst.0) {
        let _ = heap.pop();
        heap.push(Reverse(candidate));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn progress_report_every(total: usize) -> usize {
    (total / 1000).max(10_000)
}

#[cfg(not(target_arch = "wasm32"))]
fn ranked_heap_to_selected_neighbors(
    heap: Mutex<BinaryHeap<Reverse<RankedNeighbor>>>,
    top_k: usize,
) -> Vec<SelectedNeighbor> {
    let Ok(heap) = heap.into_inner() else {
        return Vec::new();
    };
    let mut ranked: Vec<RankedNeighbor> = heap.into_iter().map(|entry| entry.0).collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.neighbor.cmp(&b.neighbor)));
    ranked.truncate(top_k);
    ranked
        .into_iter()
        .map(|neighbor| SelectedNeighbor {
            neighbor: neighbor.neighbor,
            score: neighbor.score,
            matches: neighbor.matches,
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn score_all_pairs_parallel(
    spectra: &[SpectrumRecord],
    scorer: &MetricScorer,
    done_worker: Option<Arc<AtomicUsize>>,
) -> Result<Vec<PairScore>, String> {
    let pair_indices: Vec<(usize, usize)> = (0..spectra.len())
        .flat_map(|i| (i..spectra.len()).map(move |j| (i, j)))
        .collect();

    let error = Arc::new(Mutex::new(None::<String>));
    let error_worker = Arc::clone(&error);

    let pairs: Vec<PairScore> = pair_indices
        .into_par_iter()
        .filter_map(|(left_index, right_index)| {
            if let Some(done) = &done_worker {
                done.fetch_add(1, Ordering::Relaxed);
            }
            let left = spectra[left_index].spectrum.as_ref();
            let right = spectra[right_index].spectrum.as_ref();
            match scorer.similarity(left, right, left_index, right_index) {
                Ok((score, matches)) => Some(PairScore {
                    left: left_index,
                    right: right_index,
                    score,
                    matches,
                }),
                Err(err) => {
                    if let Ok(mut slot) = error_worker.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
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

    Ok(pairs)
}

fn search_match_passes(score: f64, matches: usize, params: &LibrarySearchParams) -> bool {
    matches >= params.min_matched_peaks && score >= params.min_similarity_threshold
}

fn search_parent_mass_passes(
    query: &SpectrumRecord,
    library: &SpectrumRecord,
    params: &LibrarySearchParams,
) -> bool {
    (query.meta.precursor_mz - library.meta.precursor_mz).abs() <= params.parent_mass_tolerance
}

fn finalize_search_candidates(candidates: Vec<SearchCandidate>, top_n: usize) -> Vec<CandidateHit> {
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
        query_hits.sort_by(search_candidate_order);
        for (rank, hit) in query_hits.into_iter().take(keep).enumerate() {
            hits.push(CandidateHit {
                query_index,
                library_index: hit.library_index,
                rank: rank + 1,
                spectral_score: hit.spectral_score,
                matches: hit.matches,
                payload: (),
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

fn build_search_artifact_result(
    base: SearchResult,
    library: &[SpectrumRecord],
    taxonomy: Option<&SearchTaxonomyConfig>,
    top_n: usize,
) -> SearchArtifactResult {
    let hits = enrich_search_hits(base.hits, library, taxonomy, top_n);
    SearchArtifactResult {
        hits,
        query_count: base.query_count,
        library_count: base.library_count,
        metric: base.metric,
        taxonomic_reranking_applied: taxonomy.is_some(),
        taxonomic_query: taxonomy.map(|config| config.query.query_label.clone()),
    }
}

fn enrich_search_hits(
    base_hits: Vec<CandidateHit>,
    library: &[SpectrumRecord],
    taxonomy: Option<&SearchTaxonomyConfig>,
    top_n: usize,
) -> Vec<SearchArtifactHit> {
    if let Some(taxonomy) = taxonomy {
        let candidates = base_hits
            .iter()
            .map(|hit| build_search_candidate(hit, library, Some(taxonomy)))
            .collect::<Vec<_>>();
        finalize_search_artifact_candidates(candidates, top_n)
    } else {
        base_hits
            .into_iter()
            .map(|hit| SearchArtifactHit {
                query_index: hit.query_index,
                library_index: hit.library_index,
                rank: hit.rank,
                spectral_score: hit.spectral_score,
                taxonomic_score: 0.0,
                combined_score: hit.spectral_score,
                matches: hit.matches,
                matched_organism_name: None,
                matched_organism_wikidata: None,
                matched_shared_rank: None,
                matched_short_inchikey: None,
            })
            .collect()
    }
}

fn build_search_candidate(
    hit: &CandidateHit,
    library: &[SpectrumRecord],
    taxonomy: Option<&SearchTaxonomyConfig>,
) -> SearchCandidate {
    let library_record = &library[hit.library_index];
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
        query_index: hit.query_index,
        library_index: hit.library_index,
        spectral_score: hit.spectral_score,
        taxonomic_score,
        combined_score: hit.spectral_score + taxonomic_score,
        matches: hit.matches,
        matched_organism_name,
        matched_organism_wikidata,
        matched_shared_rank,
        matched_short_inchikey,
    }
}

fn finalize_search_artifact_candidates(
    candidates: Vec<SearchCandidate>,
    top_n: usize,
) -> Vec<SearchArtifactHit> {
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
        query_hits.sort_by(search_candidate_order);
        for (rank, hit) in query_hits.into_iter().take(keep).enumerate() {
            hits.push(SearchArtifactHit {
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

fn search_candidate_order(left: &SearchCandidate, right: &SearchCandidate) -> std::cmp::Ordering {
    right
        .combined_score
        .total_cmp(&left.combined_score)
        .then(right.spectral_score.total_cmp(&left.spectral_score))
        .then(right.matches.cmp(&left.matches))
        .then(left.library_index.cmp(&right.library_index))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

    use super::{
        ComputeParams, IncrementalSearchState, IncrementalSearchStep, LibrarySearchParams,
        SearchResult, finalize_search_candidates, run_search_request, search_library,
        start_native_search, total_search_pairs,
    };
    use crate::export::SearchQueryKey;
    use crate::model::{SpectrumMetadata, SpectrumRecord};
    use crate::similarity::SimilarityMetric;

    fn spectrum(id: usize, precursor: f64, peaks: &[(f64, f64)]) -> SpectrumRecord {
        let mut spec = GenericSpectrum::<f64, f64>::with_capacity(precursor, peaks.len())
            .expect("failed to allocate spectrum");
        for (mz, intensity) in peaks {
            spec.add_peak(*mz, *intensity)
                .expect("failed to add test peak");
        }
        SpectrumRecord {
            meta: SpectrumMetadata {
                id,
                label: format!("s{id}"),
                raw_name: format!("s{id}"),
                feature_id: None,
                scans: None,
                filename: None,
                source_scan_usi: None,
                featurelist_feature_id: None,
                headers: BTreeMap::new(),
                precursor_mz: precursor,
                num_peaks: peaks.len(),
            },
            peaks: Arc::new(peaks.to_vec()),
            spectrum: Arc::new(spec),
            payload: (),
        }
    }

    fn base_compute_params(metric: SimilarityMetric) -> ComputeParams {
        ComputeParams {
            metric,
            tolerance: 0.2,
            mz_power: 0.0,
            intensity_power: 1.0,
            top_n_peaks: None,
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
            LibrarySearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 5.0,
                min_matched_peaks: 2,
                min_similarity_threshold: 0.1,
                top_n: 1,
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
        assert_eq!(result.hits[1].query_index, 1);
        assert_eq!(result.hits[1].library_index, 2);
        assert_eq!(state.total(), total_search_pairs(2, 3));
    }

    #[test]
    fn run_search_request_applies_taxonomic_reranking_before_final_top_n() {
        let request = crate::api::SearchRequest {
            query_source_label: "query.mgf".to_string(),
            query_mgf_text: Some(
                "BEGIN IONS\nNAME=q\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n"
                    .to_string(),
            ),
            query_mgf_path: None,
            library_source_label: "library.mgf".to_string(),
            library_mgf_text: Some(
                concat!(
                    "BEGIN IONS\nNAME=withania\nPEPMASS=100.0\n",
                    "INCHIKEY=AAAAAAAAAAAAAA-111\n",
                    "10 100\n21 80\n35 30\nEND IONS\n",
                    "BEGIN IONS\nNAME=panax\nPEPMASS=100.0\n",
                    "INCHIKEY=BBBBBBBBBBBBBB-222\n",
                    "10 100\n20 80\n30 50\nEND IONS\n",
                )
                .to_string(),
            ),
            library_mgf_path: None,
            parse: crate::api::ParseConfig {
                min_peaks: 1,
                max_peaks: 1000,
            },
            search: LibrarySearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 0.1,
                min_matched_peaks: 1,
                min_similarity_threshold: 0.0,
                top_n: 1,
            },
            taxonomy: Some(crate::api::SearchTaxonomyRequest {
                query_text: "Withania somnifera".to_string(),
                lotus_source_label: "lotus.csv".to_string(),
                lotus_csv_text: Some(
                    concat!(
                        "structure_inchikey,organism_wikidata,organism_name,organism_taxonomy_01domain,organism_taxonomy_02kingdom,organism_taxonomy_03phylum,organism_taxonomy_04class,organism_taxonomy_05order,organism_taxonomy_06family,organism_taxonomy_07tribe,organism_taxonomy_08genus,organism_taxonomy_09species,organism_taxonomy_10varietas\n",
                        "\"AAAAAAAAAAAAAA-111\",http://www.wikidata.org/entity/Q1,\"Withania somnifera\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania somnifera,NA\n",
                        "\"BBBBBBBBBBBBBB-222\",http://www.wikidata.org/entity/Q2,\"Panax ginseng\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Apiales,Araliaceae,NA,Panax,Panax ginseng,NA\n",
                    )
                    .to_string(),
                ),
                lotus_csv_path: None,
            }),
            query_key: Some(SearchQueryKey::FeatureId),
        };

        let artifact = run_search_request(request).expect("search request");

        assert!(artifact.result.taxonomic_reranking_applied);
        assert_eq!(artifact.result.taxonomic_query.as_deref(), Some("Withania somnifera"));
        assert_eq!(artifact.result.hits.len(), 1);
        assert_eq!(artifact.result.hits[0].library_index, 0);
        assert_eq!(artifact.result.hits[0].taxonomic_score, 9.0);
        assert_eq!(
            artifact.result.hits[0].matched_organism_name.as_deref(),
            Some("Withania somnifera")
        );
    }

    #[test]
    fn run_search_request_accepts_genus_level_taxonomic_queries() {
        let request = crate::api::SearchRequest {
            query_source_label: "query.mgf".to_string(),
            query_mgf_text: Some(
                "BEGIN IONS\nNAME=q\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n"
                    .to_string(),
            ),
            query_mgf_path: None,
            library_source_label: "library.mgf".to_string(),
            library_mgf_text: Some(
                concat!(
                    "BEGIN IONS\nNAME=withania\nPEPMASS=100.0\n",
                    "INCHIKEY=AAAAAAAAAAAAAA-111\n",
                    "10 100\n21 80\n35 30\nEND IONS\n",
                    "BEGIN IONS\nNAME=panax\nPEPMASS=100.0\n",
                    "INCHIKEY=BBBBBBBBBBBBBB-222\n",
                    "10 100\n20 80\n30 50\nEND IONS\n",
                )
                .to_string(),
            ),
            library_mgf_path: None,
            parse: crate::api::ParseConfig {
                min_peaks: 1,
                max_peaks: 1000,
            },
            search: LibrarySearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 0.1,
                min_matched_peaks: 1,
                min_similarity_threshold: 0.0,
                top_n: 1,
            },
            taxonomy: Some(crate::api::SearchTaxonomyRequest {
                query_text: "Withania".to_string(),
                lotus_source_label: "lotus.csv".to_string(),
                lotus_csv_text: Some(
                    concat!(
                        "structure_inchikey,organism_wikidata,organism_name,organism_taxonomy_01domain,organism_taxonomy_02kingdom,organism_taxonomy_03phylum,organism_taxonomy_04class,organism_taxonomy_05order,organism_taxonomy_06family,organism_taxonomy_07tribe,organism_taxonomy_08genus,organism_taxonomy_09species,organism_taxonomy_10varietas\n",
                        "\"AAAAAAAAAAAAAA-111\",http://www.wikidata.org/entity/Q1,\"Withania somnifera\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania somnifera,NA\n",
                        "\"BBBBBBBBBBBBBB-222\",http://www.wikidata.org/entity/Q2,\"Panax ginseng\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Apiales,Araliaceae,NA,Panax,Panax ginseng,NA\n",
                    )
                    .to_string(),
                ),
                lotus_csv_path: None,
            }),
            query_key: Some(SearchQueryKey::FeatureId),
        };

        let artifact = run_search_request(request).expect("search request");

        assert!(artifact.result.taxonomic_reranking_applied);
        assert_eq!(artifact.result.taxonomic_query.as_deref(), Some("Withania"));
        assert_eq!(artifact.result.hits.len(), 1);
        assert_eq!(artifact.result.hits[0].library_index, 0);
        assert_eq!(artifact.result.hits[0].taxonomic_score, 8.0);
        assert_eq!(artifact.result.hits[0].matched_shared_rank.as_deref(), Some("genus"));
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
            query_count: 1,
            library_count: 3,
            metric: SimilarityMetric::CosineGreedy,
        };

        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.library_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn search_respects_parent_mass_tolerance() {
        let queries = vec![spectrum(0, 100.0, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)])];
        let library = vec![
            spectrum(10, 100.03, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
            spectrum(11, 100.20, &[(10.0, 1.0), (20.0, 0.8), (30.0, 0.5)]),
        ];
        let result = search_library(
            queries,
            library,
            LibrarySearchParams {
                compute: base_compute_params(SimilarityMetric::CosineGreedy),
                parent_mass_tolerance: 0.05,
                min_matched_peaks: 2,
                min_similarity_threshold: 0.1,
                top_n: 5,
            },
        )
        .expect("search should succeed");

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
        let params = LibrarySearchParams {
            compute: base_compute_params(SimilarityMetric::CosineGreedy),
            parent_mass_tolerance: 5.0,
            min_matched_peaks: 2,
            min_similarity_threshold: 0.1,
            top_n: 2,
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
}
