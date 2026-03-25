use std::collections::{BTreeSet, HashSet};

use crate::compute::SearchResult;
use crate::io::SpectrumRecord;
use crate::network::{ComponentSelection, SpectralNetwork};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchQueryKey {
    FeatureId,
    FeaturelistFeatureId,
    Scans,
    RawName,
    Label,
    NodeId,
}

impl SearchQueryKey {
    pub const ALL: [Self; 6] = [
        Self::FeatureId,
        Self::FeaturelistFeatureId,
        Self::Scans,
        Self::RawName,
        Self::Label,
        Self::NodeId,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::FeatureId => "FEATURE_ID",
            Self::FeaturelistFeatureId => "FEATURELIST_FEATURE_ID",
            Self::Scans => "SCANS",
            Self::RawName => "raw_name",
            Self::Label => "label",
            Self::NodeId => "node_id",
        }
    }

    pub fn value_for(self, record: &SpectrumRecord) -> String {
        match self {
            Self::FeatureId => record.meta.feature_id.clone().unwrap_or_default(),
            Self::FeaturelistFeatureId => record
                .meta
                .featurelist_feature_id
                .clone()
                .unwrap_or_default(),
            Self::Scans => record.meta.scans.clone().unwrap_or_default(),
            Self::RawName => record.meta.raw_name.clone(),
            Self::Label => record.meta.label.clone(),
            Self::NodeId => record.meta.id.to_string(),
        }
    }
}

pub fn export_csv_strings(
    network: &SpectralNetwork,
    selection: ComponentSelection,
) -> (String, String) {
    let visible = network.visible_node_set(selection);
    let nodes = nodes_csv(network, &visible);
    let edges = edges_csv(network, &visible);
    (nodes, edges)
}

pub fn export_search_tsv(
    result: &SearchResult,
    queries: &[SpectrumRecord],
    library: &[SpectrumRecord],
    query_key: SearchQueryKey,
) -> String {
    let dynamic_headers: BTreeSet<String> = result
        .hits
        .iter()
        .filter_map(|hit| library.get(hit.library_index))
        .flat_map(|record| record.meta.headers.keys().cloned())
        .collect();

    let mut header = vec![
        "query_export_key".to_string(),
        "query_node_id".to_string(),
        "query_feature_id".to_string(),
        "query_featurelist_feature_id".to_string(),
        "query_scans".to_string(),
        "query_label".to_string(),
        "query_raw_name".to_string(),
        "hit_rank".to_string(),
        "hit_spectral_score".to_string(),
        "hit_taxonomic_score".to_string(),
        "hit_combined_score".to_string(),
        "hit_matches".to_string(),
        "hit_taxonomic_shared_rank".to_string(),
        "hit_taxonomic_organism_name".to_string(),
        "hit_taxonomic_organism_wikidata".to_string(),
        "hit_taxonomic_short_inchikey".to_string(),
        "hit_precursor_mz".to_string(),
        "hit_raw_name".to_string(),
    ];
    header.extend(dynamic_headers.iter().map(|key| format!("hit_{key}")));

    let mut out = String::new();
    push_tsv_row(&mut out, &header);

    for hit in &result.hits {
        let Some(query) = queries.get(hit.query_index) else {
            continue;
        };
        let Some(hit_record) = library.get(hit.library_index) else {
            continue;
        };

        let mut row = vec![
            query_key.value_for(query),
            query.meta.id.to_string(),
            query.meta.feature_id.clone().unwrap_or_default(),
            query
                .meta
                .featurelist_feature_id
                .clone()
                .unwrap_or_default(),
            query.meta.scans.clone().unwrap_or_default(),
            query.meta.label.clone(),
            query.meta.raw_name.clone(),
            hit.rank.to_string(),
            format!("{:.8}", hit.spectral_score),
            format!("{:.8}", hit.taxonomic_score),
            format!("{:.8}", hit.combined_score),
            hit.matches.to_string(),
            hit.matched_shared_rank.clone().unwrap_or_default(),
            hit.matched_organism_name.clone().unwrap_or_default(),
            hit.matched_organism_wikidata.clone().unwrap_or_default(),
            hit.matched_short_inchikey.clone().unwrap_or_default(),
            format!("{:.6}", hit_record.meta.precursor_mz),
            hit_record.meta.raw_name.clone(),
        ];
        row.extend(dynamic_headers.iter().map(|key| {
            hit_record
                .meta
                .headers
                .get(key)
                .cloned()
                .unwrap_or_default()
        }));
        push_tsv_row(&mut out, &row);
    }

    out
}

fn nodes_csv(network: &SpectralNetwork, visible: &HashSet<usize>) -> String {
    let mut nodes: Vec<_> = network
        .nodes
        .iter()
        .filter(|n| visible.contains(&n.id))
        .collect();
    nodes.sort_by_key(|n| n.id);

    let mut csv =
        String::from("node_id,label,raw_name,precursor_mz,num_peaks,component_id,degree\n");
    for node in nodes {
        csv.push_str(&format!(
            "{},{},{},{:.6},{},{},{}\n",
            node.id,
            escape_csv(&node.label),
            escape_csv(&node.raw_name),
            node.precursor_mz,
            node.num_peaks,
            node.component_id,
            node.degree
        ));
    }
    csv
}

fn edges_csv(network: &SpectralNetwork, visible: &HashSet<usize>) -> String {
    let mut edges: Vec<_> = network
        .edges
        .iter()
        .filter(|e| visible.contains(&e.source) && visible.contains(&e.target))
        .collect();
    edges.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

    let mut csv = String::from("source,target,score,matches\n");
    for edge in edges {
        csv.push_str(&format!(
            "{},{},{:.8},{}\n",
            edge.source, edge.target, edge.score, edge.matches
        ));
    }
    csv
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        value.to_string()
    }
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\n', " ")
        .replace('\r', " ")
}

fn push_tsv_row(out: &mut String, values: &[String]) {
    let mut first = true;
    for value in values {
        if !first {
            out.push('\t');
        }
        first = false;
        out.push_str(&escape_tsv(value));
    }
    out.push('\n');
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_csvs_to_directory(
    dir: &std::path::Path,
    nodes_csv: &str,
    edges_csv: &str,
) -> Result<(), String> {
    let nodes_path = dir.join("nodes.csv");
    let edges_path = dir.join("edges.csv");
    std::fs::write(&nodes_path, nodes_csv)
        .map_err(|err| format!("failed to write {}: {err}", nodes_path.display()))?;
    std::fs::write(&edges_path, edges_csv)
        .map_err(|err| format!("failed to write {}: {err}", edges_path.display()))?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_tsv_to_path(path: &std::path::Path, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

#[cfg(target_arch = "wasm32")]
pub fn download_csv_files(nodes_csv: &str, edges_csv: &str) -> Result<(), String> {
    download_one("nodes.csv", "text/csv;charset=utf-8", nodes_csv.as_bytes())?;
    download_one("edges.csv", "text/csv;charset=utf-8", edges_csv.as_bytes())?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn download_tsv_file(filename: &str, contents: &str) -> Result<(), String> {
    download_one(
        filename,
        "text/tab-separated-values;charset=utf-8",
        contents.as_bytes(),
    )
}

#[cfg(target_arch = "wasm32")]
fn download_one(filename: &str, mime_type: &str, bytes: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("window unavailable")?;
    let document = window.document().ok_or("document unavailable")?;

    let array = js_sys::Uint8Array::from(bytes);
    let array_parts = js_sys::Array::new();
    array_parts.push(&array.buffer());

    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime_type);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&array_parts, &bag)
        .map_err(|err| format!("failed to build Blob: {err:?}"))?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|err| format!("failed to create object URL: {err:?}"))?;

    let anchor = document
        .create_element("a")
        .map_err(|err| format!("failed to create anchor: {err:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "failed to cast anchor element".to_string())?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();

    web_sys::Url::revoke_object_url(&url)
        .map_err(|err| format!("failed to revoke object URL: {err:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc};

    use crate::compute::{SearchHit, SearchResult};
    use crate::io::{SpectrumMeta, SpectrumRecord};
    use crate::network::{ComponentSelection, NetworkEdge, NetworkNode, SpectralNetwork};

    use super::{SearchQueryKey, export_csv_strings, export_search_tsv};

    fn sample_network() -> SpectralNetwork {
        SpectralNetwork {
            nodes: vec![
                NetworkNode {
                    id: 0,
                    label: "a".to_string(),
                    raw_name: "A raw".to_string(),
                    feature_id: None,
                    scans: None,
                    filename: None,
                    source_scan_usi: None,
                    featurelist_feature_id: None,
                    precursor_mz: 100.0,
                    num_peaks: 10,
                    component_id: 0,
                    degree: 1,
                },
                NetworkNode {
                    id: 1,
                    label: "b".to_string(),
                    raw_name: "B,raw".to_string(),
                    feature_id: None,
                    scans: None,
                    filename: None,
                    source_scan_usi: None,
                    featurelist_feature_id: None,
                    precursor_mz: 101.0,
                    num_peaks: 12,
                    component_id: 0,
                    degree: 1,
                },
            ],
            edges: vec![NetworkEdge {
                source: 0,
                target: 1,
                score: 0.75,
                matches: 8,
            }],
            components: vec![vec![0, 1]],
            largest_component_id: Some(0),
        }
    }

    fn spectrum_record(id: usize, raw_name: &str, headers: &[(&str, &str)]) -> SpectrumRecord {
        let spectrum =
            GenericSpectrum::<f64, f64>::with_capacity(100.0 + id as f64, 0).expect("spectrum");
        let mut header_map = BTreeMap::new();
        for (key, value) in headers {
            header_map.insert((*key).to_string(), (*value).to_string());
        }
        SpectrumRecord {
            meta: SpectrumMeta {
                id,
                label: format!("label_{id}"),
                raw_name: raw_name.to_string(),
                feature_id: Some(format!("feature_{id}")),
                scans: Some(format!("scan_{id}")),
                filename: None,
                source_scan_usi: None,
                featurelist_feature_id: Some(format!("flf_{id}")),
                headers: header_map,
                precursor_mz: 100.0 + id as f64,
                num_peaks: 0,
            },
            peaks: Arc::new(Vec::new()),
            spectrum: Arc::new(spectrum),
            payload: (),
        }
    }

    #[test]
    fn csv_export_has_headers_and_rows() {
        let network = sample_network();
        let (nodes_csv, edges_csv) = export_csv_strings(&network, ComponentSelection::All);

        assert!(
            nodes_csv
                .starts_with("node_id,label,raw_name,precursor_mz,num_peaks,component_id,degree")
        );
        assert!(nodes_csv.contains("0,a,A raw,100.000000,10,0,1"));
        assert!(nodes_csv.contains("1,b,\"B,raw\",101.000000,12,0,1"));

        assert!(edges_csv.starts_with("source,target,score,matches"));
        assert!(edges_csv.contains("0,1,0.75000000,8"));
    }

    #[test]
    fn search_tsv_exports_one_row_per_hit_with_dynamic_headers() {
        let queries = vec![spectrum_record(0, "query 0", &[("QUERY_ONLY", "x")])];
        let library = vec![
            spectrum_record(
                10,
                "hit one",
                &[("COMPOUND_NAME", "cmpd\t1"), ("INCHIKEY", "AAAA")],
            ),
            spectrum_record(11, "hit two", &[("SMILES", "C\nC"), ("ADDUCT", "[M+H]+")]),
        ];
        let result = SearchResult {
            hits: vec![
                SearchHit {
                    query_index: 0,
                    library_index: 0,
                    rank: 1,
                    spectral_score: 0.95,
                    taxonomic_score: 9.0,
                    combined_score: 9.95,
                    matches: 6,
                    matched_organism_name: Some("Withania somnifera".to_string()),
                    matched_organism_wikidata: Some("Q1".to_string()),
                    matched_shared_rank: Some("species".to_string()),
                    matched_short_inchikey: Some("AAAA".to_string()),
                },
                SearchHit {
                    query_index: 0,
                    library_index: 1,
                    rank: 2,
                    spectral_score: 0.75,
                    taxonomic_score: 0.0,
                    combined_score: 0.75,
                    matches: 4,
                    matched_organism_name: None,
                    matched_organism_wikidata: None,
                    matched_shared_rank: None,
                    matched_short_inchikey: Some("BBBB".to_string()),
                },
            ],
            taxonomic_reranking_applied: true,
            taxonomic_query: Some("Withania somnifera".to_string()),
        };

        let tsv = export_search_tsv(&result, &queries, &library, SearchQueryKey::FeatureId);
        let mut lines = tsv.lines();
        let header = lines.next().expect("header");
        assert!(header.contains("query_export_key"));
        assert!(header.contains("hit_COMPOUND_NAME"));
        assert!(header.contains("hit_INCHIKEY"));
        assert!(header.contains("hit_SMILES"));
        assert!(header.contains("hit_ADDUCT"));
        assert!(header.contains("hit_spectral_score"));
        assert!(header.contains("hit_taxonomic_score"));
        assert!(header.contains("hit_combined_score"));

        let first = lines.next().expect("first row");
        let second = lines.next().expect("second row");
        assert!(first.starts_with("feature_0\t0\tfeature_0"));
        assert!(first.contains("\t1\t0.95000000\t9.00000000\t9.95000000\t6\tspecies\tWithania somnifera\tQ1\tAAAA\t110.000000\thit one\t"));
        assert!(first.contains("cmpd 1"));
        assert!(second.contains(
            "\t2\t0.75000000\t0.00000000\t0.75000000\t4\t\t\t\tBBBB\t111.000000\thit two\t"
        ));
        assert!(second.contains("C C"));
    }
}
