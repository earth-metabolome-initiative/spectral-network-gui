use std::collections::BTreeMap;
use std::sync::Arc;

use mass_spectrometry::prelude::GenericSpectrum;
use serde::{Deserialize, Serialize};

use crate::similarity::SimilarityMetric;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParseStats {
    pub ions_blocks: usize,
    pub accepted: usize,
    pub dropped_missing_name: usize,
    pub dropped_missing_precursor_mz: usize,
    pub dropped_too_few_peaks: usize,
    pub dropped_too_many_peaks: usize,
    pub dropped_nonpositive_intensity_peaks: usize,
    pub dropped_duplicate_mz: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectrumMetadata {
    pub id: usize,
    pub label: String,
    pub raw_name: String,
    pub feature_id: Option<String>,
    pub scans: Option<String>,
    pub filename: Option<String>,
    pub source_scan_usi: Option<String>,
    pub featurelist_feature_id: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub precursor_mz: f64,
    pub num_peaks: usize,
}

#[derive(Clone)]
pub struct SpectrumRecord<T = ()> {
    pub meta: SpectrumMetadata,
    pub peaks: Arc<Vec<(f64, f64)>>,
    pub spectrum: Arc<GenericSpectrum<f64, f64>>,
    pub payload: T,
}

#[derive(Clone)]
pub struct SpectrumCollection<T = ()> {
    pub source_label: String,
    pub spectra: Vec<SpectrumRecord<T>>,
    pub stats: ParseStats,
}

pub type LoadedSpectra = SpectrumCollection<()>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateHit<H = ()> {
    pub query_index: usize,
    pub library_index: usize,
    pub rank: usize,
    pub spectral_score: f64,
    pub matches: usize,
    pub payload: H,
}

pub trait HitLike {
    type Payload;

    fn query_index(&self) -> usize;
    fn library_index(&self) -> usize;
    fn rank(&self) -> usize;
    fn spectral_score(&self) -> f64;
    fn matches(&self) -> usize;
    fn payload(&self) -> &Self::Payload;
}

impl<H> HitLike for CandidateHit<H> {
    type Payload = H;

    fn query_index(&self) -> usize {
        self.query_index
    }

    fn library_index(&self) -> usize {
        self.library_index
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn spectral_score(&self) -> f64 {
        self.spectral_score
    }

    fn matches(&self) -> usize {
        self.matches
    }

    fn payload(&self) -> &Self::Payload {
        &self.payload
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchResult<H = ()> {
    pub hits: Vec<CandidateHit<H>>,
    pub query_count: usize,
    pub library_count: usize,
    pub metric: SimilarityMetric,
}
