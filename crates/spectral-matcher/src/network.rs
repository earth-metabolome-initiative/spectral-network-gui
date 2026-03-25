use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::model::SpectrumMetadata;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: usize,
    pub label: String,
    pub raw_name: String,
    pub feature_id: Option<String>,
    pub scans: Option<String>,
    pub filename: Option<String>,
    pub source_scan_usi: Option<String>,
    pub featurelist_feature_id: Option<String>,
    pub precursor_mz: f64,
    pub num_peaks: usize,
    pub component_id: usize,
    pub degree: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub source: usize,
    pub target: usize,
    pub score: f64,
    pub matches: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralNetwork {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub components: Vec<Vec<usize>>,
    pub largest_component_id: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentSelection {
    All,
    Largest,
    Component(usize),
}

impl SpectralNetwork {
    pub fn visible_node_ids(&self, selection: ComponentSelection) -> Vec<usize> {
        match selection {
            ComponentSelection::All => self.nodes.iter().map(|n| n.id).collect(),
            ComponentSelection::Largest => {
                let Some(component_id) = self.largest_component_id else {
                    return self.nodes.iter().map(|n| n.id).collect();
                };
                self.nodes
                    .iter()
                    .filter(|n| n.component_id == component_id)
                    .map(|n| n.id)
                    .collect()
            }
            ComponentSelection::Component(component_id) => self
                .nodes
                .iter()
                .filter(|n| n.component_id == component_id)
                .map(|n| n.id)
                .collect(),
        }
    }

    pub fn visible_node_set(&self, selection: ComponentSelection) -> HashSet<usize> {
        self.visible_node_ids(selection).into_iter().collect()
    }

    pub fn visible_edges(&self, selection: ComponentSelection) -> Vec<&NetworkEdge> {
        let visible = self.visible_node_set(selection);
        self.edges
            .iter()
            .filter(|e| visible.contains(&e.source) && visible.contains(&e.target))
            .collect()
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
pub struct SelectedNeighbor {
    pub neighbor: usize,
    pub score: f64,
    pub matches: usize,
}

pub fn build_network(
    metas: &[SpectrumMetadata],
    scores: &[PairScore],
    threshold: f64,
    top_k: usize,
) -> SpectralNetwork {
    let n = metas.len();

    let mut neighbors: Vec<Vec<(usize, f64, usize)>> = vec![Vec::new(); n];
    for pair in scores {
        if pair.left == pair.right || pair.score < threshold {
            continue;
        }
        neighbors[pair.left].push((pair.right, pair.score, pair.matches));
        neighbors[pair.right].push((pair.left, pair.score, pair.matches));
    }

    let mut selected_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (node_id, node_neighbors) in neighbors.iter_mut().enumerate() {
        node_neighbors.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (neighbor_id, _, _) in node_neighbors.iter().take(top_k) {
            selected_neighbors[node_id].insert(*neighbor_id);
        }
    }

    let mut edge_map: HashMap<(usize, usize), NetworkEdge> = HashMap::new();
    for pair in scores {
        if pair.left == pair.right || pair.score < threshold {
            continue;
        }
        let (a, b) = if pair.left < pair.right {
            (pair.left, pair.right)
        } else {
            (pair.right, pair.left)
        };

        if selected_neighbors[a].contains(&b) || selected_neighbors[b].contains(&a) {
            edge_map.entry((a, b)).or_insert(NetworkEdge {
                source: a,
                target: b,
                score: pair.score,
                matches: pair.matches,
            });
        }
    }

    let mut edges: Vec<NetworkEdge> = edge_map.into_values().collect();
    edges.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

    assemble_network(metas, edges)
}

pub fn build_network_from_selected_neighbors(
    metas: &[SpectrumMetadata],
    selected_neighbors: &[Vec<SelectedNeighbor>],
) -> SpectralNetwork {
    let mut edge_map: HashMap<(usize, usize), NetworkEdge> = HashMap::new();
    for (source, neighbors) in selected_neighbors.iter().enumerate() {
        for neighbor in neighbors {
            if source == neighbor.neighbor {
                continue;
            }
            let (a, b) = if source < neighbor.neighbor {
                (source, neighbor.neighbor)
            } else {
                (neighbor.neighbor, source)
            };
            edge_map.entry((a, b)).or_insert(NetworkEdge {
                source: a,
                target: b,
                score: neighbor.score,
                matches: neighbor.matches,
            });
        }
    }

    let mut edges: Vec<NetworkEdge> = edge_map.into_values().collect();
    edges.sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.target)));

    assemble_network(metas, edges)
}

fn assemble_network(metas: &[SpectrumMetadata], edges: Vec<NetworkEdge>) -> SpectralNetwork {
    let n = metas.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in &edges {
        adjacency[edge.source].push(edge.target);
        adjacency[edge.target].push(edge.source);
    }

    let mut component_id_by_node = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if component_id_by_node[start] != usize::MAX {
            continue;
        }
        let cid = components.len();
        let mut stack = vec![start];
        component_id_by_node[start] = cid;
        let mut members = Vec::new();
        while let Some(node) = stack.pop() {
            members.push(node);
            for &next in &adjacency[node] {
                if component_id_by_node[next] == usize::MAX {
                    component_id_by_node[next] = cid;
                    stack.push(next);
                }
            }
        }
        components.push(members);
    }

    let mut degree = vec![0usize; n];
    for edge in &edges {
        degree[edge.source] += 1;
        degree[edge.target] += 1;
    }

    let nodes = metas
        .iter()
        .map(|meta| NetworkNode {
            id: meta.id,
            label: meta.label.clone(),
            raw_name: meta.raw_name.clone(),
            feature_id: meta.feature_id.clone(),
            scans: meta.scans.clone(),
            filename: meta.filename.clone(),
            source_scan_usi: meta.source_scan_usi.clone(),
            featurelist_feature_id: meta.featurelist_feature_id.clone(),
            precursor_mz: meta.precursor_mz,
            num_peaks: meta.num_peaks,
            component_id: component_id_by_node[meta.id],
            degree: degree[meta.id],
        })
        .collect::<Vec<_>>();

    let largest_component_id = components
        .iter()
        .enumerate()
        .max_by_key(|(_, members)| members.len())
        .map(|(idx, _)| idx);

    SpectralNetwork {
        nodes,
        edges,
        components,
        largest_component_id,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        ComponentSelection, PairScore, SelectedNeighbor, build_network,
        build_network_from_selected_neighbors,
    };
    use crate::model::SpectrumMetadata;

    fn meta(id: usize) -> SpectrumMetadata {
        SpectrumMetadata {
            id,
            label: format!("s{id}"),
            raw_name: format!("raw{id}"),
            feature_id: None,
            scans: None,
            filename: None,
            source_scan_usi: None,
            featurelist_feature_id: None,
            headers: BTreeMap::new(),
            precursor_mz: 100.0 + id as f64,
            num_peaks: 10,
        }
    }

    #[test]
    fn threshold_and_topk_with_or_rule_keeps_expected_edges() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.9,
                matches: 5,
            },
            PairScore {
                left: 0,
                right: 2,
                score: 0.8,
                matches: 5,
            },
            PairScore {
                left: 1,
                right: 2,
                score: 0.7,
                matches: 5,
            },
            PairScore {
                left: 1,
                right: 3,
                score: 0.6,
                matches: 5,
            },
        ];

        let network = build_network(&metas, &scores, 0.5, 1);
        let edge_pairs = network
            .edges
            .iter()
            .map(|e| (e.source, e.target))
            .collect::<Vec<_>>();

        assert!(edge_pairs.contains(&(0, 1)));
        assert!(edge_pairs.contains(&(0, 2)));
        assert!(edge_pairs.contains(&(1, 3)));
        assert!(!edge_pairs.contains(&(1, 2)));
    }

    #[test]
    fn component_selection_reports_largest_component() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3), meta(4)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.9,
                matches: 1,
            },
            PairScore {
                left: 1,
                right: 2,
                score: 0.9,
                matches: 1,
            },
            PairScore {
                left: 3,
                right: 4,
                score: 0.9,
                matches: 1,
            },
        ];

        let network = build_network(&metas, &scores, 0.5, 5);
        let largest_visible = network.visible_node_ids(ComponentSelection::Largest);
        assert_eq!(largest_visible.len(), 3);
    }

    #[test]
    fn selected_neighbor_build_matches_pair_score_build() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.9,
                matches: 5,
            },
            PairScore {
                left: 0,
                right: 2,
                score: 0.8,
                matches: 5,
            },
            PairScore {
                left: 1,
                right: 2,
                score: 0.7,
                matches: 5,
            },
            PairScore {
                left: 1,
                right: 3,
                score: 0.6,
                matches: 5,
            },
        ];
        let selected = vec![
            vec![SelectedNeighbor {
                neighbor: 1,
                score: 0.9,
                matches: 5,
            }],
            vec![
                SelectedNeighbor {
                    neighbor: 0,
                    score: 0.9,
                    matches: 5,
                },
                SelectedNeighbor {
                    neighbor: 3,
                    score: 0.6,
                    matches: 5,
                },
            ],
            vec![SelectedNeighbor {
                neighbor: 0,
                score: 0.8,
                matches: 5,
            }],
            vec![SelectedNeighbor {
                neighbor: 1,
                score: 0.6,
                matches: 5,
            }],
        ];

        let expected = build_network(&metas, &scores, 0.5, 1);
        let actual = build_network_from_selected_neighbors(&metas, &selected);

        assert_eq!(expected.edges.len(), actual.edges.len());
        assert_eq!(
            expected
                .edges
                .iter()
                .map(|edge| (edge.source, edge.target, edge.score, edge.matches))
                .collect::<Vec<_>>(),
            actual
                .edges
                .iter()
                .map(|edge| (edge.source, edge.target, edge.score, edge.matches))
                .collect::<Vec<_>>()
        );
    }
}
