use std::collections::HashMap;
use std::io::Read;

use crate::model::SpectrumRecord;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    pub organism_name: String,
    pub organism_wikidata: Option<String>,
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
                                > current.shared_rank.map(|rank| rank.score() as usize).unwrap_or(0))
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
}

pub fn load_lotus_bytes(bytes: &[u8]) -> Result<LotusMetadataIndex, String> {
    parse_lotus_reader(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_lotus_path(path: &std::path::Path) -> Result<LotusMetadataIndex, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    load_lotus_bytes(&bytes)
}

pub fn short_inchikey_from_record<T>(record: &SpectrumRecord<T>) -> Option<String> {
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

fn parse_lotus_reader<R: Read>(reader: R) -> Result<LotusMetadataIndex, String> {
    let mut csv = csv::ReaderBuilder::new().flexible(true).from_reader(reader);
    let headers = csv
        .headers()
        .map_err(|err| format!("failed to read LOTUS header: {err}"))?
        .clone();

    let short_inchikey_idx = header_index(&headers, "structure_inchikey")?;
    let organism_name_idx = header_index(&headers, "organism_name")?;
    let organism_wikidata_idx = header_index(&headers, "organism_wikidata")?;
    let taxonomy_indices = TAXONOMY_COLUMN_NAMES
        .iter()
        .map(|name| header_index(&headers, name))
        .collect::<Result<Vec<_>, _>>()?;

    let mut by_short_inchikey: HashMap<String, Vec<LotusBiosource>> = HashMap::new();
    let mut by_organism_name: HashMap<String, TaxonomyLineage> = HashMap::new();
    let mut by_organism_wikidata: HashMap<String, TaxonomyLineage> = HashMap::new();
    let mut by_taxon_name: HashMap<String, TaxonomyLineage> = HashMap::new();

    for record in csv.records() {
        let record = record.map_err(|err| format!("failed to parse LOTUS row: {err}"))?;

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
            organism_name: organism_name.clone(),
            organism_wikidata: organism_wikidata.clone(),
            lineage: lineage.clone(),
        };
        by_short_inchikey.entry(short_key).or_default().push(biosource);

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

    Ok(LotusMetadataIndex {
        by_short_inchikey,
        by_organism_name,
        by_organism_wikidata,
        by_taxon_name,
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

#[cfg(test)]
mod tests {
    use super::{TaxonomicRank, load_lotus_bytes};

    fn sample_lotus() -> super::LotusMetadataIndex {
        let csv = concat!(
            "structure_inchikey,organism_wikidata,organism_name,organism_taxonomy_01domain,organism_taxonomy_02kingdom,organism_taxonomy_03phylum,organism_taxonomy_04class,organism_taxonomy_05order,organism_taxonomy_06family,organism_taxonomy_07tribe,organism_taxonomy_08genus,organism_taxonomy_09species,organism_taxonomy_10varietas\n",
            "\"ABCDEFGHIJKLMN-AAAA\",http://www.wikidata.org/entity/Q1,\"Withania somnifera\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania somnifera,NA\n",
            "\"ABCDEFGHIJKLMN-BBBB\",http://www.wikidata.org/entity/Q2,\"Withania coagulans\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Solanales,Solanaceae,NA,Withania,Withania coagulans,NA\n",
            "\"ZZZZZZZZZZZZZZ-CCCC\",http://www.wikidata.org/entity/Q3,\"Panax ginseng\",Eukaryota,Archaeplastida,Streptophyta,Magnoliopsida,Apiales,Araliaceae,NA,Panax,Panax ginseng,NA\n",
        );
        load_lotus_bytes(csv.as_bytes()).expect("lotus")
    }

    #[test]
    fn resolves_query_lineage_by_name_qid_and_genus() {
        let lotus = sample_lotus();

        let by_name = lotus
            .resolve_query_lineage("Withania somnifera")
            .expect("name lineage");
        assert_eq!(
            by_name.lineage.value_for(TaxonomicRank::Species),
            Some("Withania somnifera")
        );

        let by_qid = lotus.resolve_query_lineage("Q1").expect("qid lineage");
        assert_eq!(by_qid.query_label, "Q1");
        assert_eq!(by_qid.lineage.value_for(TaxonomicRank::Genus), Some("Withania"));

        let by_genus = lotus
            .resolve_query_lineage("Withania")
            .expect("genus lineage");
        assert_eq!(by_genus.query_label, "Withania");
        assert_eq!(
            by_genus.lineage.value_for(TaxonomicRank::Family),
            Some("Solanaceae")
        );
        assert_eq!(by_genus.lineage.value_for(TaxonomicRank::Genus), Some("Withania"));
        assert_eq!(by_genus.lineage.value_for(TaxonomicRank::Species), None);
    }
}
