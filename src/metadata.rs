use std::collections::HashMap;
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use crate::io::SpectrumRecord;

const TAXONOMY_COLUMN_NAMES: [&str; 10] = [
    "organism_taxonomy_01domain",
    "organism_taxonomy_02kingdom",
    "organism_taxonomy_03phylum",
    "organism_taxonomy_04class",
    "organism_taxonomy_05order",
    "organism_taxonomy_06family",
    "organism_taxonomy_07tribe",
    "organism_taxonomy_08genus",
    "organism_taxonomy_09species",
    "organism_taxonomy_10varietas",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaxonomicRank {
    Domain,
    Kingdom,
    Phylum,
    Class,
    Order,
    Family,
    Tribe,
    Genus,
    Species,
    Varietas,
}

impl TaxonomicRank {
    pub const ALL: [Self; 10] = [
        Self::Domain,
        Self::Kingdom,
        Self::Phylum,
        Self::Class,
        Self::Order,
        Self::Family,
        Self::Tribe,
        Self::Genus,
        Self::Species,
        Self::Varietas,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Domain => "domain",
            Self::Kingdom => "kingdom",
            Self::Phylum => "phylum",
            Self::Class => "class",
            Self::Order => "order",
            Self::Family => "family",
            Self::Tribe => "tribe",
            Self::Genus => "genus",
            Self::Species => "species",
            Self::Varietas => "varietas",
        }
    }

    pub fn score(self) -> u8 {
        match self {
            Self::Domain => 1,
            Self::Kingdom => 2,
            Self::Phylum => 3,
            Self::Class => 4,
            Self::Order => 5,
            Self::Family => 6,
            Self::Tribe => 7,
            Self::Genus => 8,
            Self::Species => 9,
            Self::Varietas => 10,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Domain => 0,
            Self::Kingdom => 1,
            Self::Phylum => 2,
            Self::Class => 3,
            Self::Order => 4,
            Self::Family => 5,
            Self::Tribe => 6,
            Self::Genus => 7,
            Self::Species => 8,
            Self::Varietas => 9,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaxonomyLineage {
    ranks: [Option<String>; 10],
}

impl TaxonomyLineage {
    pub fn from_rank_values(values: [Option<String>; 10]) -> Self {
        Self { ranks: values }
    }

    pub fn value_for(&self, rank: TaxonomicRank) -> Option<&str> {
        self.ranks[rank.index()].as_deref()
    }

    pub fn specificity_score(&self) -> usize {
        TaxonomicRank::ALL
            .iter()
            .rev()
            .find_map(|rank| self.value_for(*rank).map(|_| rank.score() as usize))
            .unwrap_or(0)
    }

    pub fn deepest_shared_rank(&self, other: &Self) -> Option<TaxonomicRank> {
        TaxonomicRank::ALL.iter().rev().copied().find(|rank| {
            match (self.value_for(*rank), other.value_for(*rank)) {
                (Some(left), Some(right)) => left == right,
                _ => false,
            }
        })
    }

    pub fn truncated_to(&self, rank: TaxonomicRank) -> Self {
        let mut ranks: [Option<String>; 10] = Default::default();
        for candidate in TaxonomicRank::ALL {
            if candidate.index() > rank.index() {
                break;
            }
            ranks[candidate.index()] = self.value_for(candidate).map(ToOwned::to_owned);
        }
        Self { ranks }
    }

    fn merge_prefer_more_specific(&mut self, other: &Self) {
        if other.specificity_score() > self.specificity_score() {
            *self = other.clone();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LotusBiosource {
    pub compound_name: Option<String>,
    pub organism_name: String,
    pub organism_wikidata: Option<String>,
    pub compound_wikidata: Option<String>,
    pub reference_doi: Option<String>,
    pub lineage: TaxonomyLineage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedLotusQuery {
    pub query_label: String,
    pub lineage: TaxonomyLineage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaxonomicMatch {
    pub score: u8,
    pub shared_rank: Option<TaxonomicRank>,
    pub matched_organism_name: Option<String>,
    pub matched_organism_wikidata: Option<String>,
    pub matched_short_inchikey: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct LotusMetadataIndex {
    by_short_inchikey: HashMap<String, Vec<LotusBiosource>>,
    by_organism_name: HashMap<String, TaxonomyLineage>,
    by_organism_wikidata: HashMap<String, TaxonomyLineage>,
    by_taxon_name: HashMap<String, TaxonomyLineage>,
}

impl LotusMetadataIndex {
    pub fn resolve_query_lineage(&self, input: &str) -> Option<ResolvedLotusQuery> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(qid) = normalized_qid(trimmed) {
            let lineage = self.by_organism_wikidata.get(&qid)?.clone();
            return Some(ResolvedLotusQuery {
                query_label: qid,
                lineage,
            });
        }

        if let Some(lineage) = self.by_organism_name.get(trimmed) {
            return Some(ResolvedLotusQuery {
                query_label: trimmed.to_string(),
                lineage: lineage.clone(),
            });
        }

        let lineage = self.by_taxon_name.get(trimmed)?.clone();
        Some(ResolvedLotusQuery {
            query_label: trimmed.to_string(),
            lineage,
        })
    }

    pub fn match_candidate(
        &self,
        short_inchikey: &str,
        query_lineage: &TaxonomyLineage,
    ) -> Option<TaxonomicMatch> {
        let biosources = self.by_short_inchikey.get(short_inchikey)?;
        let mut best: Option<TaxonomicMatch> = None;

        for biosource in biosources {
            let shared_rank = biosource.lineage.deepest_shared_rank(query_lineage);
            let score = shared_rank.map(TaxonomicRank::score).unwrap_or(0);
            let candidate = TaxonomicMatch {
                score,
                shared_rank,
                matched_organism_name: Some(biosource.organism_name.clone()),
                matched_organism_wikidata: biosource.organism_wikidata.clone(),
                matched_short_inchikey: Some(short_inchikey.to_string()),
            };

            let replace = match &best {
                None => true,
                Some(current) => {
                    candidate.score > current.score
                        || (candidate.score == current.score
                            && biosource.lineage.specificity_score()
                                > current
                                    .shared_rank
                                    .map(|rank| rank.score() as usize)
                                    .unwrap_or(0))
                        || (candidate.score == current.score
                            && candidate.matched_organism_name.as_deref()
                                < current.matched_organism_name.as_deref())
                }
            };
            if replace {
                best = Some(candidate);
            }
        }

        best
    }

    pub fn occurrences_for_short_inchikey(
        &self,
        short_inchikey: &str,
    ) -> Option<&[LotusBiosource]> {
        self.by_short_inchikey
            .get(short_inchikey)
            .map(Vec::as_slice)
    }

    pub fn row_count(&self) -> usize {
        self.by_short_inchikey
            .values()
            .map(std::vec::Vec::len)
            .sum::<usize>()
    }

    pub fn structure_count(&self) -> usize {
        self.by_short_inchikey.len()
    }

    pub fn queryable_organism_count(&self) -> usize {
        self.by_organism_name
            .len()
            .max(self.by_organism_wikidata.len())
            .max(self.by_taxon_name.len())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LotusMetadataStats {
    pub rows: usize,
    pub indexed_structures: usize,
    pub indexed_biosources: usize,
    pub queryable_organisms: usize,
}

#[derive(Clone, Debug)]
pub struct LoadedLotusMetadata {
    pub source_label: String,
    pub index: Arc<LotusMetadataIndex>,
    pub stats: LotusMetadataStats,
}

#[cfg(not(target_arch = "wasm32"))]
pub enum NativeLotusLoadMessage {
    Finished(LoadedLotusMetadata),
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
pub struct NativeLotusLoadHandle {
    rx: mpsc::Receiver<NativeLotusLoadMessage>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeLotusLoadHandle {
    pub fn try_recv(&self) -> Option<NativeLotusLoadMessage> {
        self.rx.try_recv().ok()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_lotus_path(path: &Path) -> Result<LoadedLotusMetadata, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    load_lotus_bytes(&path.display().to_string(), &bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_native_lotus_load(path: &Path) -> Result<NativeLotusLoadHandle, String> {
    let source_label = path.display().to_string();
    let bytes =
        std::fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || match load_lotus_bytes(&source_label, &bytes) {
        Ok(loaded) => {
            let _ = tx.send(NativeLotusLoadMessage::Finished(loaded));
        }
        Err(err) => {
            let _ = tx.send(NativeLotusLoadMessage::Failed(err));
        }
    });
    Ok(NativeLotusLoadHandle { rx })
}

pub fn load_lotus_bytes(source_label: &str, bytes: &[u8]) -> Result<LoadedLotusMetadata, String> {
    parse_lotus_reader(source_label, bytes)
}

pub fn short_inchikey_from_record(record: &SpectrumRecord) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for (key, value) in &record.meta.headers {
        let normalized = normalize_key(key);
        let priority = if normalized == "ik2d" {
            0
        } else if normalized == "structureinchikey" || normalized == "gnpsinchikey" {
            1
        } else if normalized.contains("inchikey") {
            2
        } else {
            continue;
        };
        let Some(short) = short_inchikey(value) else {
            continue;
        };
        match &best {
            Some((best_priority, _)) if *best_priority <= priority => {}
            _ => best = Some((priority, short)),
        }
    }
    best.map(|(_, value)| value)
}

fn parse_lotus_reader<R: Read>(
    source_label: &str,
    reader: R,
) -> Result<LoadedLotusMetadata, String> {
    let mut csv = csv::ReaderBuilder::new().flexible(true).from_reader(reader);
    let headers = csv
        .headers()
        .map_err(|err| format!("failed to read LOTUS header: {err}"))?
        .clone();

    let short_inchikey_idx = header_index(&headers, "structure_inchikey")?;
    let organism_name_idx = header_index(&headers, "organism_name")?;
    let organism_wikidata_idx = header_index(&headers, "organism_wikidata")?;
    let compound_name_idx = optional_header_index_any(
        &headers,
        &[
            "structure_nameTraditional",
            "structure_name",
            "compound_name",
            "traditional_name",
        ],
    );
    let compound_wikidata_idx =
        optional_header_index_any(&headers, &["structure_wikidata", "compound_wikidata"]);
    let reference_doi_idx = optional_header_index_any(&headers, &["reference_doi", "doi"]);
    let taxonomy_indices = TAXONOMY_COLUMN_NAMES
        .iter()
        .map(|name| header_index(&headers, name))
        .collect::<Result<Vec<_>, _>>()?;

    let mut by_short_inchikey: HashMap<String, Vec<LotusBiosource>> = HashMap::new();
    let mut by_organism_name: HashMap<String, TaxonomyLineage> = HashMap::new();
    let mut by_organism_wikidata: HashMap<String, TaxonomyLineage> = HashMap::new();
    let mut by_taxon_name: HashMap<String, TaxonomyLineage> = HashMap::new();
    let mut rows = 0usize;

    for record in csv.records() {
        let record = record.map_err(|err| format!("failed to parse LOTUS row: {err}"))?;
        rows += 1;

        let Some(short_key) = record.get(short_inchikey_idx).and_then(short_inchikey) else {
            continue;
        };
        let organism_name = record
            .get(organism_name_idx)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("")
            .to_string();
        if organism_name.is_empty() {
            continue;
        }

        let organism_wikidata = record
            .get(organism_wikidata_idx)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let compound_name = compound_name_idx
            .and_then(|idx| record.get(idx))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let compound_wikidata = compound_wikidata_idx
            .and_then(|idx| record.get(idx))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .and_then(normalized_qid);
        let reference_doi = reference_doi_idx
            .and_then(|idx| record.get(idx))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);

        let mut rank_values: [Option<String>; 10] = Default::default();
        for (slot, idx) in rank_values.iter_mut().zip(taxonomy_indices.iter().copied()) {
            *slot = record
                .get(idx)
                .map(str::trim)
                .filter(|value| !value.is_empty() && *value != "NA")
                .map(ToOwned::to_owned);
        }
        let lineage = TaxonomyLineage::from_rank_values(rank_values);
        let biosource = LotusBiosource {
            compound_name,
            organism_name: organism_name.clone(),
            organism_wikidata: organism_wikidata.clone(),
            compound_wikidata,
            reference_doi,
            lineage: lineage.clone(),
        };
        by_short_inchikey
            .entry(short_key)
            .or_default()
            .push(biosource);

        by_organism_name
            .entry(organism_name)
            .and_modify(|existing| existing.merge_prefer_more_specific(&lineage))
            .or_insert(lineage.clone());

        if let Some(qid) = organism_wikidata.and_then(|value| normalized_qid(&value)) {
            by_organism_wikidata
                .entry(qid)
                .and_modify(|existing| existing.merge_prefer_more_specific(&lineage))
                .or_insert(lineage.clone());
        }

        for rank in TaxonomicRank::ALL {
            let Some(name) = lineage.value_for(rank) else {
                continue;
            };
            let truncated = lineage.truncated_to(rank);
            by_taxon_name
                .entry(name.to_string())
                .and_modify(|existing| existing.merge_prefer_more_specific(&truncated))
                .or_insert(truncated);
        }
    }

    let index = LotusMetadataIndex {
        by_short_inchikey,
        by_organism_name,
        by_organism_wikidata,
        by_taxon_name,
    };
    let stats = LotusMetadataStats {
        rows,
        indexed_structures: index.structure_count(),
        indexed_biosources: index.row_count(),
        queryable_organisms: index.queryable_organism_count(),
    };

    Ok(LoadedLotusMetadata {
        source_label: source_label.to_string(),
        index: Arc::new(index),
        stats,
    })
}

fn optional_header_index_any(headers: &csv::StringRecord, targets: &[&str]) -> Option<usize> {
    headers.iter().position(|header| {
        let normalized = normalize_key(header);
        targets
            .iter()
            .any(|target| normalized == normalize_key(target))
    })
}

fn header_index(headers: &csv::StringRecord, target: &str) -> Result<usize, String> {
    headers
        .iter()
        .position(|header| header.trim() == target)
        .ok_or_else(|| format!("LOTUS file is missing required column '{target}'"))
}

fn normalize_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn normalized_qid(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = trimmed
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(trimmed);
    let upper = candidate.to_ascii_uppercase();
    let suffix = upper.strip_prefix('Q')?;
    if suffix.chars().all(|ch| ch.is_ascii_digit()) {
        Some(format!("Q{suffix}"))
    } else {
        None
    }
}

pub fn short_inchikey(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_matches('"');
    if trimmed.is_empty() {
        return None;
    }
    let upper = trimmed.to_ascii_uppercase();
    let compact: String = upper.chars().filter(|ch| *ch != '-').collect();
    if compact.len() < 14 {
        return None;
    }
    Some(compact.chars().take(14).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc};

    use super::{
        LoadedLotusMetadata, TaxonomicRank, TaxonomyLineage, load_lotus_bytes,
        short_inchikey_from_record,
    };
    use crate::io::{SpectrumMeta, SpectrumRecord};

    fn sample_record_with_headers(headers: &[(&str, &str)]) -> SpectrumRecord {
        let mut header_map = BTreeMap::new();
        for (key, value) in headers {
            header_map.insert((*key).to_string(), (*value).to_string());
        }
        SpectrumRecord {
            meta: SpectrumMeta {
                id: 1,
                label: "record".to_string(),
                raw_name: "record".to_string(),
                feature_id: None,
                scans: None,
                filename: None,
                source_scan_usi: None,
                featurelist_feature_id: None,
                headers: header_map,
                precursor_mz: 100.0,
                num_peaks: 0,
            },
            peaks: Arc::new(Vec::new()),
            spectrum: Arc::new(
                GenericSpectrum::<f64, f64>::with_capacity(100.0, 0).expect("spectrum"),
            ),
            payload: (),
        }
    }

    fn sample_lotus() -> LoadedLotusMetadata {
        let csv = concat!(
            "structure_inchikey,structure_nameTraditional,structure_wikidata,reference_doi,organism_wikidata,organism_name,organism_taxonomy_01domain,organism_taxonomy_02kingdom,organism_taxonomy_03phylum,organism_taxonomy_04class,organism_taxonomy_05order,organism_taxonomy_06family,organism_taxonomy_07tribe,organism_taxonomy_08genus,organism_taxonomy_09species,organism_taxonomy_10varietas\n",
            "\"ABCDEFGHIJKLMN-AAAA\",Withanolide A,http://www.wikidata.org/entity/Q100,10.1000/alpha,http://www.wikidata.org/entity/Q1,\"Withania somnifera\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania somnifera,NA\n",
            "\"ABCDEFGHIJKLMN-BBBB\",Withanolide A,http://www.wikidata.org/entity/Q100,10.1000/beta,http://www.wikidata.org/entity/Q2,\"Withania coagulans\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania coagulans,NA\n",
            "\"ZZZZZZZZZZZZZZ-CCCC\",Ginsenoside Rg1,http://www.wikidata.org/entity/Q200,10.1000/gamma,http://www.wikidata.org/entity/Q3,\"Panax ginseng\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Apiales,Araliaceae,NA,Panax,Panax ginseng,NA\n"
        );
        load_lotus_bytes("lotus.csv", csv.as_bytes()).expect("lotus should parse")
    }

    #[test]
    fn parses_quoted_lotus_csv_and_indexes_short_inchikeys() {
        let loaded = sample_lotus();
        assert_eq!(loaded.stats.rows, 3);
        assert_eq!(loaded.stats.indexed_structures, 2);
        assert_eq!(loaded.stats.indexed_biosources, 3);
    }

    #[test]
    fn resolves_query_lineage_by_name_qid_and_genus() {
        let loaded = sample_lotus();
        let by_name = loaded
            .index
            .resolve_query_lineage("Withania somnifera")
            .expect("name lineage");
        assert_eq!(
            by_name.lineage.value_for(TaxonomicRank::Species),
            Some("Withania somnifera")
        );

        let by_qid = loaded
            .index
            .resolve_query_lineage("Q1")
            .expect("qid lineage");
        assert_eq!(by_qid.query_label, "Q1");
        assert_eq!(
            by_qid.lineage.value_for(TaxonomicRank::Genus),
            Some("Withania")
        );

        let by_genus = loaded
            .index
            .resolve_query_lineage("Withania")
            .expect("genus lineage");
        assert_eq!(by_genus.query_label, "Withania");
        assert_eq!(
            by_genus.lineage.value_for(TaxonomicRank::Family),
            Some("Solanaceae")
        );
        assert_eq!(
            by_genus.lineage.value_for(TaxonomicRank::Genus),
            Some("Withania")
        );
        assert_eq!(by_genus.lineage.value_for(TaxonomicRank::Species), None);
    }

    #[test]
    fn taxonomy_scoring_uses_deepest_shared_rank() {
        let query = TaxonomyLineage::from_rank_values([
            Some("Eukaryota".to_string()),
            Some("Archaeplastida".to_string()),
            Some("Streptophyta".to_string()),
            Some("Magnoliopsida".to_string()),
            Some("Solanales".to_string()),
            Some("Solanaceae".to_string()),
            None,
            Some("Withania".to_string()),
            Some("Withania somnifera".to_string()),
            None,
        ]);
        let candidate = TaxonomyLineage::from_rank_values([
            Some("Eukaryota".to_string()),
            Some("Archaeplastida".to_string()),
            Some("Streptophyta".to_string()),
            Some("Magnoliopsida".to_string()),
            Some("Solanales".to_string()),
            Some("Solanaceae".to_string()),
            None,
            Some("Withania".to_string()),
            Some("Withania coagulans".to_string()),
            None,
        ]);
        assert_eq!(
            query.deepest_shared_rank(&candidate),
            Some(TaxonomicRank::Genus)
        );
    }

    #[test]
    fn candidate_match_prefers_highest_scoring_biosource() {
        let loaded = sample_lotus();
        let query = loaded
            .index
            .resolve_query_lineage("Withania somnifera")
            .expect("query");
        let matched = loaded
            .index
            .match_candidate("ABCDEFGHIJKLMN", &query.lineage)
            .expect("match");
        assert_eq!(matched.score, 9);
        assert_eq!(matched.shared_rank, Some(TaxonomicRank::Species));
        assert_eq!(
            matched.matched_organism_name.as_deref(),
            Some("Withania somnifera")
        );
    }

    #[test]
    fn occurrences_keep_compound_qid_and_reference_doi() {
        let loaded = sample_lotus();
        let occurrences = loaded
            .index
            .occurrences_for_short_inchikey("ABCDEFGHIJKLMN")
            .expect("occurrences");
        assert_eq!(occurrences.len(), 2);
        assert_eq!(
            occurrences[0].compound_name.as_deref(),
            Some("Withanolide A")
        );
        assert_eq!(occurrences[0].compound_wikidata.as_deref(), Some("Q100"));
        assert_eq!(
            occurrences[0].reference_doi.as_deref(),
            Some("10.1000/alpha")
        );
    }

    #[test]
    fn derives_short_inchikey_from_library_headers() {
        let record = sample_record_with_headers(&[("InChIKey", "abcdefghijklmN-XYZ")]);
        assert_eq!(
            short_inchikey_from_record(&record).as_deref(),
            Some("ABCDEFGHIJKLMN")
        );
    }
}
