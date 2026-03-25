use serde::{Deserialize, Serialize};

use crate::export::SearchQueryKey;
use crate::model::{ParseStats, SpectrumMetadata};
use crate::network::SpectralNetwork;
use crate::search::LibrarySearchParams;
use crate::similarity::ComputeParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParseConfig {
    #[serde(default = "default_min_peaks")]
    pub min_peaks: usize,
    #[serde(default = "default_max_peaks")]
    pub max_peaks: usize,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            min_peaks: default_min_peaks(),
            max_peaks: default_max_peaks(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkBuildParams {
    pub compute: ComputeParams,
    pub threshold: f64,
    pub top_k: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkRequest {
    pub source_label: String,
    #[serde(default)]
    pub mgf_text: Option<String>,
    #[serde(default)]
    pub mgf_path: Option<String>,
    #[serde(default)]
    pub parse: ParseConfig,
    pub build: NetworkBuildParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query_source_label: String,
    #[serde(default)]
    pub query_mgf_text: Option<String>,
    #[serde(default)]
    pub query_mgf_path: Option<String>,
    pub library_source_label: String,
    #[serde(default)]
    pub library_mgf_text: Option<String>,
    #[serde(default)]
    pub library_mgf_path: Option<String>,
    #[serde(default)]
    pub parse: ParseConfig,
    pub search: LibrarySearchParams,
    #[serde(default)]
    pub taxonomy: Option<SearchTaxonomyRequest>,
    #[serde(default)]
    pub query_key: Option<SearchQueryKey>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchTaxonomyRequest {
    pub query_text: String,
    pub lotus_source_label: String,
    #[serde(default)]
    pub lotus_csv_text: Option<String>,
    #[serde(default)]
    pub lotus_csv_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkSpectrum {
    pub meta: SpectrumMetadata,
    pub peaks: Vec<(f64, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkArtifact {
    pub source_label: String,
    pub parse_stats: ParseStats,
    pub build: NetworkBuildParams,
    pub spectra: Vec<NetworkSpectrum>,
    pub network: SpectralNetwork,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchArtifact {
    pub query_source_label: String,
    pub library_source_label: String,
    pub query_stats: ParseStats,
    pub library_stats: ParseStats,
    pub search: LibrarySearchParams,
    pub taxonomy: Option<SearchTaxonomyRequest>,
    pub query_key: SearchQueryKey,
    pub query_spectra: Vec<SpectrumMetadata>,
    pub library_spectra: Vec<SpectrumMetadata>,
    pub result: SearchArtifactResult,
    pub tsv: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchArtifactResult {
    pub hits: Vec<SearchArtifactHit>,
    pub query_count: usize,
    pub library_count: usize,
    pub metric: crate::similarity::SimilarityMetric,
    pub taxonomic_reranking_applied: bool,
    pub taxonomic_query: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchArtifactHit {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobCreatedResponse {
    pub job_id: u64,
    pub status: JobStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobProgressStage {
    Queued,
    LoadingSpectra,
    LoadingQuery,
    LoadingLibrary,
    Scoring,
    BuildingNetwork,
    Finalizing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobProgress {
    pub stage: JobProgressStage,
    pub completed: u64,
    pub total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Finished,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatusResponse {
    pub job_id: u64,
    pub status: JobStatus,
    pub error: Option<String>,
    #[serde(default)]
    pub progress: Option<JobProgress>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum MatcherJobResult {
    Network(NetworkArtifact),
    LibrarySearch(SearchArtifact),
}

const fn default_min_peaks() -> usize {
    5
}

const fn default_max_peaks() -> usize {
    1000
}
