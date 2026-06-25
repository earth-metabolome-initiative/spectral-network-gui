use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct NetworkNode {
    pub id: usize,
    pub spectrum_id: String,
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

#[derive(Clone, Debug)]
pub struct NetworkEdge {
    pub source: usize,
    pub target: usize,
    pub score: f64,
    pub matches: usize,
}

#[derive(Clone, Debug)]
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
