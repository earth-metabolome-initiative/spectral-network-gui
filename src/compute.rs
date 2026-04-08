pub use spectral_matcher::{ComputeParams, SimilarityMetric};

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
