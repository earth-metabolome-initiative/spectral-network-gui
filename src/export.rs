use std::collections::HashSet;

use crate::network::{ComponentSelection, SpectralNetwork};

pub fn export_csv_strings(
    network: &SpectralNetwork,
    selection: ComponentSelection,
) -> (String, String) {
    let visible = network.visible_node_set(selection);
    let nodes = nodes_csv(network, &visible);
    let edges = edges_csv(network, &visible);
    (nodes, edges)
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

#[cfg(target_arch = "wasm32")]
pub fn download_csv_files(nodes_csv: &str, edges_csv: &str) -> Result<(), String> {
    download_one("nodes.csv", nodes_csv.as_bytes())?;
    download_one("edges.csv", edges_csv.as_bytes())?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn download_one(filename: &str, bytes: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("window unavailable")?;
    let document = window.document().ok_or("document unavailable")?;

    let array = js_sys::Uint8Array::from(bytes);
    let array_parts = js_sys::Array::new();
    array_parts.push(&array.buffer());

    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type("text/csv;charset=utf-8");
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
    use crate::network::{ComponentSelection, NetworkEdge, NetworkNode, SpectralNetwork};

    use super::export_csv_strings;

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
}
