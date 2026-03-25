pub mod api;
pub mod export;
pub mod mgf;
pub mod model;
pub mod network;
pub mod search;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
pub mod similarity;
pub mod taxonomy;

pub use api::{
    HealthResponse, JobCreatedResponse, JobProgress, JobProgressStage, JobStatus,
    JobStatusResponse, MatcherJobResult, NetworkArtifact, NetworkBuildParams, NetworkRequest,
    NetworkSpectrum, ParseConfig, SearchArtifact, SearchArtifactHit, SearchArtifactResult,
    SearchRequest, SearchTaxonomyRequest,
};
pub use export::{SearchQueryKey, export_search_json, export_search_tsv};
#[cfg(not(target_arch = "wasm32"))]
pub use export::{save_json_to_path, save_tsv_to_path};
#[cfg(target_arch = "wasm32")]
pub use export::download_tsv_file;
pub use mgf::load_mgf_bytes;
#[cfg(not(target_arch = "wasm32"))]
pub use mgf::{NativeLoadHandle, NativeLoadMessage, load_mgf_path, start_native_mgf_load};
#[cfg(target_arch = "wasm32")]
pub use mgf::load_mgf_file_for_wasm;
pub use model::{
    CandidateHit, HitLike, LoadedSpectra, ParseStats, SearchResult, SpectrumCollection,
    SpectrumMetadata, SpectrumRecord,
};
pub use network::{
    ComponentSelection, NetworkEdge, NetworkNode, SpectralNetwork, build_network,
};
pub use search::{
    IncrementalSearchState, IncrementalSearchStep, LibrarySearchParams, NativeSearchHandle,
    SearchMessage, build_network_artifact, build_network_artifact_with_progress,
    run_search_request, run_search_request_with_progress, search_library, start_native_search,
    total_search_pairs,
};
#[cfg(not(target_arch = "wasm32"))]
pub use server::serve;
pub use similarity::{
    ComputeParams, MetricScorer, SimilarityMetric, preprocess_spectra_for_metric,
};
