use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use eframe::egui;
use egui_extras::{Column, TableBuilder};

#[cfg(not(target_arch = "wasm32"))]
use crate::attributes::AttributeTable;
use crate::attributes::LoadedAttributeTable;
#[cfg(target_arch = "wasm32")]
use crate::compute::NativeComputeHandle;
use crate::compute::{
    ComputeMessage, ComputeParams, IncrementalComputeState, IncrementalStep, PairScore,
    SimilarityMetric,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::compute::{NativeComputeHandle, start_native_compute};
use crate::export::export_csv_strings;
#[cfg(target_arch = "wasm32")]
use crate::io::load_mgf_bytes;
#[cfg(not(target_arch = "wasm32"))]
use crate::io::load_mgf_path;
use crate::io::{LoadedSpectra, ParseStats, SpectrumMeta, SpectrumRecord};
use crate::layout::force_directed_layout;
use crate::network::{ComponentSelection, SpectralNetwork, build_network};
use crate::render::{GraphViewState, draw_network};

const MIN_PEAKS: usize = 5;
const MAX_PEAKS: usize = 1000;
const LAYOUT_STOP_EPSILON: f32 = 0.0015;
const LAYOUT_STOP_STREAK: usize = 15;
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_MGF_PATH: &str = "spectral-cosine-similarity/fixtures/mapp_batch_00231.mgf";

#[cfg(target_arch = "wasm32")]
struct UploadedFile {
    name: String,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeAttrMatchField {
    NodeId,
    FeatureId,
    RawName,
    Label,
}

impl NodeAttrMatchField {
    fn label(self) -> &'static str {
        match self {
            Self::NodeId => "node_id",
            Self::FeatureId => "feature_id",
            Self::RawName => "raw_name",
            Self::Label => "label",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeAttrPanelDock {
    Bottom,
    Right,
    Detached,
}

impl NodeAttrPanelDock {
    fn label(self) -> &'static str {
        match self {
            Self::Bottom => "Bottom split",
            Self::Right => "Right split",
            Self::Detached => "Detached window",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructurePanelMode {
    SelectedNode,
    SelectedSet,
}

impl StructurePanelMode {
    fn label(self) -> &'static str {
        match self {
            Self::SelectedNode => "Selected node",
            Self::SelectedSet => "Selected set",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeAttrSortColumn {
    NodeId,
    TableColumn(usize),
}

struct NodeColoring {
    fills: HashMap<usize, egui::Color32>,
    mode: NodeColorMode,
}

enum NodeColorMode {
    Categorical {
        legend: Vec<(String, egui::Color32)>,
    },
    Continuous,
}

struct DepictFetchResult {
    request_key: String,
    content_type: Option<String>,
    bytes: Vec<u8>,
    error: Option<String>,
}

struct DepictImageState {
    image_uri: Option<String>,
    image_bytes: Option<egui::load::Bytes>,
    loading: bool,
    error: Option<String>,
}

struct SelectedStructureEntry {
    node_id: usize,
    display_label: String,
    smiles: String,
    annotations: Vec<(String, String)>,
}

pub struct SpectralApp {
    #[cfg(not(target_arch = "wasm32"))]
    mgf_path: String,
    source_label: Option<String>,
    parse_stats: Option<ParseStats>,
    spectra: Vec<SpectrumRecord>,

    tolerance_input: String,
    mz_power_input: String,
    intensity_power_input: String,
    selected_metric: SimilarityMetric,

    threshold: f64,
    top_k: usize,
    hide_singletons: bool,
    node_force: f32,
    edge_force: f32,
    layout_running: bool,
    stabilize_iterations: usize,
    layout_mean_displacement: f32,
    layout_low_motion_streak: usize,
    request_fit_view: bool,

    pair_scores: Option<Vec<PairScore>>,
    network: Option<SpectralNetwork>,
    component_selection: ComponentSelection,
    positions: HashMap<usize, [f32; 2]>,
    view_state: GraphViewState,
    hovered_node_id: Option<usize>,
    selected_node_id: Option<usize>,

    native_compute: Option<NativeComputeHandle>,
    incremental_compute: Option<IncrementalComputeState>,

    status_message: Option<String>,
    error_message: Option<String>,

    node_attributes: Option<LoadedAttributeTable>,
    edge_attributes: Option<LoadedAttributeTable>,
    node_attr_match_field: NodeAttrMatchField,
    depict_smiles_column: Option<usize>,
    color_nodes_by_attribute: bool,
    node_color_attr_column: usize,
    show_categorical_legend: bool,
    show_node_attributes_panel: bool,
    node_attr_search_query: String,
    node_attr_panel_dock: NodeAttrPanelDock,
    node_attr_sort_column: NodeAttrSortColumn,
    node_attr_sort_ascending: bool,
    node_attr_bottom_height: f32,
    node_attr_right_width: f32,
    pending_center_node_id: Option<usize>,
    table_node_filter: Option<HashSet<usize>>,
    depict_cache: HashMap<String, DepictImageState>,
    depict_tx: std::sync::mpsc::Sender<DepictFetchResult>,
    depict_rx: Receiver<DepictFetchResult>,
    structure_panel_mode: StructurePanelMode,
    show_selection_structures_detached: bool,
    selection_structures_limit: usize,
    selection_structures_image_height: f32,
    structure_caption_columns: Vec<usize>,

    #[cfg(target_arch = "wasm32")]
    upload_promise: Option<poll_promise::Promise<Result<UploadedFile, String>>>,
}

impl SpectralApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let (depict_tx, depict_rx) = mpsc::channel();
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            mgf_path: DEFAULT_MGF_PATH.to_string(),
            source_label: None,
            parse_stats: None,
            spectra: Vec::new(),
            tolerance_input: "0.02".to_string(),
            mz_power_input: "0".to_string(),
            intensity_power_input: "1".to_string(),
            selected_metric: SimilarityMetric::default(),
            threshold: 0.7,
            top_k: 10,
            hide_singletons: true,
            node_force: 1.0,
            edge_force: 1.0,
            layout_running: false,
            stabilize_iterations: 100,
            layout_mean_displacement: 0.0,
            layout_low_motion_streak: 0,
            request_fit_view: false,
            pair_scores: None,
            network: None,
            component_selection: ComponentSelection::All,
            positions: HashMap::new(),
            view_state: GraphViewState {
                pan: egui::Vec2::ZERO,
                zoom: 1.0,
                dragging_node_id: None,
                box_select_start: None,
                box_select_current: None,
            },
            hovered_node_id: None,
            selected_node_id: None,
            native_compute: None,
            incremental_compute: None,
            status_message: None,
            error_message: None,
            node_attributes: None,
            edge_attributes: None,
            node_attr_match_field: NodeAttrMatchField::NodeId,
            depict_smiles_column: None,
            color_nodes_by_attribute: false,
            node_color_attr_column: 0,
            show_categorical_legend: true,
            show_node_attributes_panel: true,
            node_attr_search_query: String::new(),
            node_attr_panel_dock: NodeAttrPanelDock::Bottom,
            node_attr_sort_column: NodeAttrSortColumn::NodeId,
            node_attr_sort_ascending: true,
            node_attr_bottom_height: 260.0,
            node_attr_right_width: 560.0,
            pending_center_node_id: None,
            table_node_filter: None,
            depict_cache: HashMap::new(),
            depict_tx,
            depict_rx,
            structure_panel_mode: StructurePanelMode::SelectedNode,
            show_selection_structures_detached: false,
            selection_structures_limit: 48,
            selection_structures_image_height: 260.0,
            structure_caption_columns: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            upload_promise: None,
        }
    }

    fn clear_compute_outputs(&mut self) {
        self.pair_scores = None;
        self.network = None;
        self.positions.clear();
        self.hovered_node_id = None;
        self.selected_node_id = None;
        self.component_selection = ComponentSelection::All;
        self.layout_running = false;
        self.layout_mean_displacement = 0.0;
        self.layout_low_motion_streak = 0;
        self.request_fit_view = false;
        self.view_state.dragging_node_id = None;
        self.view_state.box_select_start = None;
        self.view_state.box_select_current = None;
        self.pending_center_node_id = None;
        self.table_node_filter = None;
    }

    fn set_loaded_spectra(&mut self, loaded: LoadedSpectra) {
        self.source_label = Some(loaded.source_label.clone());
        self.parse_stats = Some(loaded.stats);
        self.spectra = loaded.spectra;
        self.clear_compute_outputs();
        self.status_message = Some(format!(
            "Loaded {} spectra from {}",
            self.spectra.len(),
            loaded.source_label
        ));
        self.error_message = None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn reset_node_attribute_table_state(&mut self, columns: &[String]) {
        let table_columns = columns.len();
        self.node_attr_search_query.clear();
        self.node_attr_sort_column = if table_columns > 0 {
            NodeAttrSortColumn::TableColumn(0)
        } else {
            NodeAttrSortColumn::NodeId
        };
        self.node_attr_sort_ascending = true;
        self.depict_smiles_column = columns.iter().enumerate().find_map(|(idx, col)| {
            let normalized = normalized_column_name(col);
            if normalized == "smiles" || normalized.ends_with("smiles") {
                Some(idx)
            } else {
                None
            }
        });
        self.structure_caption_columns =
            default_structure_caption_columns(columns, self.depict_smiles_column);
        if table_columns == 0 {
            self.node_color_attr_column = 0;
            self.color_nodes_by_attribute = false;
        } else if self.node_color_attr_column >= table_columns {
            self.node_color_attr_column = 0;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_from_path(&mut self) {
        let path = self.mgf_path.trim();
        if path.is_empty() {
            self.error_message = Some("MGF path is empty".to_string());
            return;
        }

        match load_mgf_path(Path::new(path), MIN_PEAKS, MAX_PEAKS) {
            Ok(loaded) => self.set_loaded_spectra(loaded),
            Err(err) => {
                self.error_message = Some(err);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_attributes_from_tsv(&mut self, source_label: String, text: &str, is_node: bool) {
        match AttributeTable::parse_tsv(text) {
            Ok(table) => {
                let loaded = LoadedAttributeTable::new(source_label.clone(), table);
                let rows = loaded.table.rows.len();
                let cols = loaded.table.columns.len();
                if is_node {
                    let column_names = loaded.table.columns.clone();
                    self.node_attributes = Some(loaded);
                    self.reset_node_attribute_table_state(&column_names);
                } else {
                    self.edge_attributes = Some(loaded);
                }
                self.status_message = Some(format!(
                    "Loaded {} attribute table: {} rows, {} columns from {}",
                    if is_node { "node" } else { "edge" },
                    rows,
                    cols,
                    source_label
                ));
                self.error_message = None;
            }
            Err(err) => {
                self.error_message = Some(format!(
                    "Failed to parse {} attributes: {err}",
                    if is_node { "node" } else { "edge" }
                ));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn pick_and_load_attributes(&mut self, is_node: bool) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("TSV", &["tsv", "txt"])
            .pick_file()
        else {
            return;
        };

        match std::fs::read_to_string(&path) {
            Ok(content) => {
                self.load_attributes_from_tsv(path.display().to_string(), &content, is_node);
            }
            Err(err) => {
                self.error_message = Some(format!(
                    "Failed to read attribute TSV {}: {err}",
                    path.display()
                ));
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn start_upload_dialog(&mut self) {
        self.upload_promise = Some(poll_promise::Promise::spawn_local(async move {
            let Some(file_handle) = rfd::AsyncFileDialog::new()
                .add_filter("MGF", &["mgf"])
                .pick_file()
                .await
            else {
                return Err("No file selected".to_string());
            };

            let name = file_handle.file_name();
            let bytes = file_handle.read().await;
            Ok(UploadedFile { name, bytes })
        }));
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_upload_dialog(&mut self) {
        let Some(promise) = &self.upload_promise else {
            return;
        };

        let Some(result) = promise.ready() else {
            return;
        };

        match result {
            Ok(file) => match load_mgf_bytes(&file.name, &file.bytes, MIN_PEAKS, MAX_PEAKS) {
                Ok(loaded) => self.set_loaded_spectra(loaded),
                Err(err) => self.error_message = Some(err),
            },
            Err(err) => {
                if err != "No file selected" {
                    self.error_message = Some(err.clone());
                }
            }
        }

        self.upload_promise = None;
    }

    fn parse_compute_params(&self) -> Result<ComputeParams, String> {
        let tolerance = self
            .tolerance_input
            .trim()
            .parse::<f64>()
            .map_err(|_| "Invalid tolerance".to_string())?;
        let mz_power = self
            .mz_power_input
            .trim()
            .parse::<f64>()
            .map_err(|_| "Invalid mz_power".to_string())?;
        let intensity_power = self
            .intensity_power_input
            .trim()
            .parse::<f64>()
            .map_err(|_| "Invalid intensity_power".to_string())?;

        Ok(ComputeParams {
            metric: self.selected_metric,
            tolerance,
            mz_power,
            intensity_power,
        })
    }

    fn start_compute(&mut self) {
        if self.spectra.is_empty() {
            self.error_message = Some("Load spectra before starting compute".to_string());
            return;
        }

        let params = match self.parse_compute_params() {
            Ok(params) => params,
            Err(err) => {
                self.error_message = Some(err);
                return;
            }
        };

        self.clear_compute_outputs();
        self.error_message = None;
        self.status_message = Some(format!(
            "Computing {} scores...",
            self.selected_metric.label()
        ));

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native_compute = Some(start_native_compute(self.spectra.clone(), params));
        }

        #[cfg(target_arch = "wasm32")]
        {
            match IncrementalComputeState::new(self.spectra.clone(), params) {
                Ok(state) => {
                    self.incremental_compute = Some(state);
                }
                Err(err) => {
                    self.error_message = Some(err);
                    self.status_message = None;
                }
            }
        }
    }

    fn cancel_compute(&mut self) {
        if let Some(handle) = &self.native_compute {
            handle.cancel();
        }
        if let Some(state) = &mut self.incremental_compute {
            state.cancel();
        }
        self.status_message = Some("Cancelling compute...".to_string());
    }

    fn rebuild_network(&mut self) {
        let Some(scores) = self.pair_scores.as_ref() else {
            return;
        };

        let metas: Vec<SpectrumMeta> = self.spectra.iter().map(|s| s.meta.clone()).collect();
        let network = build_network(&metas, scores, self.threshold, self.top_k);

        if let ComponentSelection::Component(cid) = self.component_selection
            && cid >= network.components.len()
        {
            self.component_selection = ComponentSelection::All;
        }
        let visible =
            visible_node_ids_for_view(&network, self.component_selection, self.hide_singletons);
        self.selected_node_id = keep_selected_if_visible(
            self.selected_node_id,
            &network,
            self.component_selection,
            self.hide_singletons,
        );
        let layout = force_directed_layout(
            &network,
            &visible,
            &self.positions,
            120,
            self.node_force,
            self.edge_force,
        );
        for (node_id, pos) in layout.positions {
            self.positions.insert(node_id, pos);
        }
        self.layout_mean_displacement = layout.mean_displacement;
        self.layout_low_motion_streak = 0;
        self.layout_running = false;
        self.request_fit_view = true;

        self.status_message = Some(format!(
            "Network rebuilt: threshold={:.3}, top-k={}, nodes={}, edges={}, components={}",
            self.threshold,
            self.top_k,
            network.nodes.len(),
            network.edges.len(),
            network.components.len()
        ));
        self.network = Some(network);
    }

    fn poll_compute(&mut self, ctx: &egui::Context) {
        let mut finished: Option<ComputeMessage> = None;

        if let Some(handle) = &self.native_compute {
            if let Some(msg) = handle.try_recv() {
                finished = Some(msg);
            } else {
                ctx.request_repaint();
            }
        }

        if let Some(state) = &mut self.incremental_compute {
            match state.step(2_000) {
                Ok(IncrementalStep::Progress) => {
                    ctx.request_repaint();
                }
                Ok(IncrementalStep::Finished(result)) => {
                    finished = Some(ComputeMessage::Finished(result));
                }
                Ok(IncrementalStep::Cancelled) => {
                    finished = Some(ComputeMessage::Cancelled);
                }
                Err(err) => {
                    finished = Some(ComputeMessage::Failed(err));
                }
            }
        }

        if let Some(message) = finished {
            self.native_compute = None;
            self.incremental_compute = None;

            match message {
                ComputeMessage::Finished(result) => {
                    self.pair_scores = Some(result.pairs);
                    self.status_message = Some("Compute finished".to_string());
                    self.rebuild_network();
                }
                ComputeMessage::Cancelled => {
                    self.status_message = Some("Compute cancelled".to_string());
                }
                ComputeMessage::Failed(err) => {
                    self.error_message = Some(err);
                }
            }
        }
    }

    fn compute_progress(&self) -> Option<(usize, usize)> {
        if let Some(handle) = &self.native_compute {
            return Some((handle.done(), handle.total()));
        }
        if let Some(state) = &self.incremental_compute {
            return Some((state.done(), state.total()));
        }
        None
    }

    fn is_computing(&self) -> bool {
        self.native_compute.is_some() || self.incremental_compute.is_some()
    }

    fn relayout_visible(&mut self, iterations: usize) -> Option<f32> {
        let Some(network) = &self.network else {
            return None;
        };
        let visible =
            visible_node_ids_for_view(network, self.component_selection, self.hide_singletons);
        if visible.is_empty() {
            self.layout_mean_displacement = 0.0;
            self.layout_low_motion_streak = 0;
            return None;
        }
        let layout_result = force_directed_layout(
            network,
            &visible,
            &self.positions,
            iterations,
            self.node_force,
            self.edge_force,
        );
        for (node_id, pos) in layout_result.positions {
            self.positions.insert(node_id, pos);
        }
        self.layout_mean_displacement = layout_result.mean_displacement;
        self.request_fit_view = true;
        Some(layout_result.mean_displacement)
    }

    fn fit_view_to_visible_nodes(&mut self, canvas_size: egui::Vec2, visible_node_ids: &[usize]) {
        if visible_node_ids.is_empty() || canvas_size.x <= 1.0 || canvas_size.y <= 1.0 {
            self.view_state.pan = egui::Vec2::ZERO;
            self.view_state.zoom = 1.0;
            return;
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut found = false;

        for node_id in visible_node_ids {
            let Some(pos) = self.positions.get(node_id) else {
                continue;
            };
            found = true;
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
            min_y = min_y.min(pos[1]);
            max_y = max_y.max(pos[1]);
        }

        if !found {
            self.view_state.pan = egui::Vec2::ZERO;
            self.view_state.zoom = 1.0;
            return;
        }

        let padding = (visible_node_ids.len() as f32).sqrt() * 0.03 + 0.5;
        min_x -= padding;
        max_x += padding;
        min_y -= padding;
        max_y += padding;

        let width = (max_x - min_x).max(0.2);
        let height = (max_y - min_y).max(0.2);
        let base_scale = (canvas_size.x.min(canvas_size.y) * 0.4).max(1e-6);
        let half_w = (canvas_size.x * 0.5 - 64.0).max(32.0);
        let half_h = (canvas_size.y * 0.5 - 64.0).max(32.0);
        let half_span_x = (width * 0.5).max(0.1);
        let half_span_y = (height * 0.5).max(0.1);
        let zoom_x = half_w / (half_span_x * base_scale);
        let zoom_y = half_h / (half_span_y * base_scale);
        self.view_state.zoom = zoom_x.min(zoom_y).clamp(0.01, 8.0);

        let center_x = (min_x + max_x) * 0.5;
        let center_y = (min_y + max_y) * 0.5;
        let scaled = base_scale * self.view_state.zoom;
        self.view_state.pan = egui::Vec2::new(-center_x * scaled, -center_y * scaled);
    }

    fn center_view_on_node(&mut self, canvas_size: egui::Vec2, node_id: usize) -> bool {
        if canvas_size.x <= 1.0 || canvas_size.y <= 1.0 {
            return false;
        }
        let Some(pos) = self.positions.get(&node_id) else {
            return false;
        };
        let base_scale = (canvas_size.x.min(canvas_size.y) * 0.4).max(1e-6);
        let scaled = base_scale * self.view_state.zoom;
        self.view_state.pan = egui::Vec2::new(-pos[0] * scaled, -pos[1] * scaled);
        true
    }

    fn focus_view_on_node_set(
        &mut self,
        canvas_size: egui::Vec2,
        component_nodes: &[usize],
        node_id: usize,
    ) -> bool {
        if canvas_size.x <= 1.0 || canvas_size.y <= 1.0 {
            return false;
        }
        if component_nodes.is_empty() {
            return self.center_view_on_node(canvas_size, node_id);
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut found = 0usize;

        for member_id in component_nodes {
            let Some(pos) = self.positions.get(member_id) else {
                continue;
            };
            found += 1;
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
            min_y = min_y.min(pos[1]);
            max_y = max_y.max(pos[1]);
        }

        if found == 0 {
            return self.center_view_on_node(canvas_size, node_id);
        }

        // Fit full component with generous margins so all nodes remain visible.
        let padding = (found as f32).sqrt() * 0.02 + 0.25;
        min_x -= padding;
        max_x += padding;
        min_y -= padding;
        max_y += padding;

        let width = (max_x - min_x).max(0.05);
        let height = (max_y - min_y).max(0.05);
        let base_scale = (canvas_size.x.min(canvas_size.y) * 0.4).max(1e-6);
        let half_w = (canvas_size.x * 0.5 - 64.0).max(32.0);
        let half_h = (canvas_size.y * 0.5 - 64.0).max(32.0);
        let half_span_x = (width * 0.5).max(0.05);
        let half_span_y = (height * 0.5).max(0.05);
        let zoom_x = half_w / (half_span_x * base_scale);
        let zoom_y = half_h / (half_span_y * base_scale);
        let fit_zoom = zoom_x.min(zoom_y);
        self.view_state.zoom = (fit_zoom * 0.94).clamp(0.02, 16.0);

        let center_x = (min_x + max_x) * 0.5;
        let center_y = (min_y + max_y) * 0.5;
        let scaled = base_scale * self.view_state.zoom;
        self.view_state.pan = egui::Vec2::new(-center_x * scaled, -center_y * scaled);
        true
    }

    fn node_attribute_key_for_node(&self, node: &crate::network::NetworkNode) -> Option<String> {
        match self.node_attr_match_field {
            NodeAttrMatchField::NodeId => Some(node.id.to_string()),
            NodeAttrMatchField::FeatureId => node.feature_id.clone(),
            NodeAttrMatchField::RawName => Some(node.raw_name.clone()),
            NodeAttrMatchField::Label => Some(node.label.clone()),
        }
    }

    fn selected_node_smiles(&self, node: &crate::network::NetworkNode) -> Option<String> {
        let table = self.node_attributes.as_ref()?;
        let smiles_col_idx = self.depict_smiles_column?;
        let key = self.node_attribute_key_for_node(node)?;
        let row = table.find_row(&key)?;
        let smiles = row.get(smiles_col_idx)?.trim();
        if smiles.is_empty() {
            None
        } else {
            Some(smiles.to_string())
        }
    }

    fn selected_visible_node(
        &self,
        network: &SpectralNetwork,
    ) -> Option<crate::network::NetworkNode> {
        let node_id = self.selected_node_id?;
        let visible_set =
            visible_node_set_for_view(network, self.component_selection, self.hide_singletons);
        if !visible_set.contains(&node_id) {
            return None;
        }
        network.nodes.iter().find(|n| n.id == node_id).cloned()
    }

    fn ensure_depiction_request(&mut self, request_key: String, uri: String) {
        if self
            .depict_cache
            .get(&request_key)
            .is_some_and(|state| state.loading || state.image_bytes.is_some())
        {
            return;
        }

        self.depict_cache.insert(
            request_key.clone(),
            DepictImageState {
                image_uri: None,
                image_bytes: None,
                loading: true,
                error: None,
            },
        );

        let mut request = ehttp::Request::get(uri);
        request.headers = ehttp::Headers::new(&[
            ("Accept", "image/svg+xml, image/*;q=0.9, */*;q=0.8"),
            ("Cache-Control", "no-cache"),
        ]);

        let tx = self.depict_tx.clone();
        ehttp::fetch(request, move |result| {
            let fetch_result = match result {
                Ok(response) => {
                    if !(200..300).contains(&response.status) {
                        DepictFetchResult {
                            request_key,
                            content_type: response.content_type().map(ToOwned::to_owned),
                            bytes: response.bytes,
                            error: Some(format!(
                                "depict API returned HTTP {} {}",
                                response.status, response.status_text
                            )),
                        }
                    } else {
                        DepictFetchResult {
                            request_key,
                            content_type: response.content_type().map(ToOwned::to_owned),
                            bytes: response.bytes,
                            error: None,
                        }
                    }
                }
                Err(err) => DepictFetchResult {
                    request_key,
                    content_type: None,
                    bytes: Vec::new(),
                    error: Some(format!("depict API request failed: {err}")),
                },
            };
            let _ = tx.send(fetch_result);
        });
    }

    fn poll_depiction(&mut self, ctx: &egui::Context) {
        let mut received = false;
        loop {
            match self.depict_rx.try_recv() {
                Ok(result) => {
                    received = true;
                    let entry = self
                        .depict_cache
                        .entry(result.request_key.clone())
                        .or_insert(DepictImageState {
                            image_uri: None,
                            image_bytes: None,
                            loading: false,
                            error: None,
                        });
                    entry.loading = false;

                    if let Some(err) = result.error {
                        entry.error = Some(err);
                        entry.image_uri = None;
                        entry.image_bytes = None;
                        continue;
                    }

                    let image_ext = detect_depict_image_extension(
                        result.content_type.as_deref(),
                        &result.bytes,
                    );
                    let Some(ext) = image_ext else {
                        let preview = String::from_utf8(result.bytes)
                            .ok()
                            .map(|text| text.chars().take(200).collect::<String>())
                            .unwrap_or_else(|| "<binary response>".to_string());
                        entry.error = Some(format!(
                            "Unsupported depiction response (content-type={:?}). Preview: {}",
                            result.content_type, preview
                        ));
                        entry.image_uri = None;
                        entry.image_bytes = None;
                        continue;
                    };

                    let request_id = stable_hash(&result.request_key);
                    entry.image_uri = Some(format!("bytes://np_depict_{request_id}.{ext}"));
                    entry.image_bytes = Some(egui::load::Bytes::from(result.bytes));
                    entry.error = None;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        if received || self.depict_cache.values().any(|entry| entry.loading) {
            ctx.request_repaint();
        }
    }

    fn draw_depiction_widget(&self, ui: &mut egui::Ui, request_key: &str, size: egui::Vec2) {
        let Some(state) = self.depict_cache.get(request_key) else {
            ui.add(egui::Spinner::new());
            return;
        };

        if state.loading {
            ui.add(egui::Spinner::new());
            return;
        }
        if let (Some(uri), Some(bytes)) = (&state.image_uri, &state.image_bytes) {
            ui.add(
                egui::Image::from_bytes(uri.clone(), bytes.clone())
                    .fit_to_exact_size(size)
                    .maintain_aspect_ratio(true),
            );
            return;
        }
        if let Some(err) = &state.error {
            ui.small(err);
            return;
        }
        ui.small("Depiction unavailable.");
    }

    fn selected_structures_from_node_selection(&self) -> Vec<SelectedStructureEntry> {
        let Some(network) = self.network.as_ref() else {
            return Vec::new();
        };
        let Some(table) = self.node_attributes.as_ref() else {
            return Vec::new();
        };
        let Some(smiles_col_idx) = self.depict_smiles_column else {
            return Vec::new();
        };
        let Some(filter) = self.table_node_filter.as_ref() else {
            return Vec::new();
        };

        let node_by_id: HashMap<usize, &crate::network::NetworkNode> =
            network.nodes.iter().map(|n| (n.id, n)).collect();
        let mut node_ids: Vec<usize> = filter.iter().copied().collect();
        node_ids.sort_unstable();
        node_ids.dedup();

        node_ids
            .into_iter()
            .filter_map(|node_id| {
                let node = node_by_id.get(&node_id)?;
                let key = self.node_attribute_key_for_node(node)?;
                let row = table.find_row(&key)?;
                let smiles = row.get(smiles_col_idx)?.trim();
                if smiles.is_empty() {
                    return None;
                }
                let display_label = node
                    .feature_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| node.label.clone());
                let annotations = self
                    .structure_caption_columns
                    .iter()
                    .filter_map(|idx| {
                        let col_name = table.table.columns.get(*idx)?;
                        let value = row.get(*idx)?.trim();
                        if value.is_empty() {
                            None
                        } else {
                            Some((col_name.clone(), value.to_string()))
                        }
                    })
                    .collect();
                Some(SelectedStructureEntry {
                    node_id,
                    display_label,
                    smiles: smiles.to_string(),
                    annotations,
                })
            })
            .collect()
    }

    fn draw_selected_structures_gallery(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Max structures");
            ui.add(
                egui::DragValue::new(&mut self.selection_structures_limit)
                    .range(1..=500)
                    .speed(1.0),
            );
            ui.label("Image height");
            ui.add(
                egui::DragValue::new(&mut self.selection_structures_image_height)
                    .range(140.0..=700.0)
                    .speed(1.0),
            );
        });
        if self.table_node_filter.is_none() {
            ui.small("Use rectangle selection on the graph to choose nodes.");
            return;
        }
        if self.depict_smiles_column.is_none() {
            ui.small("Select a SMILES column in the left panel.");
            return;
        }

        let selected_structures = self.selected_structures_from_node_selection();
        if selected_structures.is_empty() {
            ui.small("No SMILES available for the selected nodes.");
            return;
        }

        let max_items = self.selection_structures_limit.max(1);
        let structures: Vec<&SelectedStructureEntry> =
            selected_structures.iter().take(max_items).collect();
        if selected_structures.len() > structures.len() {
            ui.small(format!(
                "Showing {} / {} structures (increase Max structures to show more).",
                structures.len(),
                selected_structures.len()
            ));
        } else {
            ui.small(format!("Showing {} structures.", structures.len()));
        }

        let gallery_height = ui.available_height().max(260.0);
        egui::ScrollArea::vertical()
            .max_height(gallery_height)
            .show(ui, |ui| {
                for entry in structures {
                    ui.group(|ui| {
                        let card_width = ui.available_width().max(260.0);
                        let card_height =
                            self.selection_structures_image_height.clamp(140.0, 700.0);
                        let depict_uri = naturalproducts_depict_uri(
                            &entry.smiles,
                            card_width as u32,
                            card_height as u32,
                        );
                        self.ensure_depiction_request(depict_uri.clone(), depict_uri.clone());
                        self.draw_depiction_widget(
                            ui,
                            &depict_uri,
                            egui::vec2(card_width, card_height),
                        );
                        if ui.link(&entry.display_label).clicked() {
                            self.selected_node_id = Some(entry.node_id);
                            self.pending_center_node_id = Some(entry.node_id);
                        }
                        if entry.annotations.is_empty() {
                            ui.small("No additional structure metadata selected.");
                        } else {
                            for (column, value) in &entry.annotations {
                                ui.add(egui::Label::new(format!("{column}: {value}")).wrap());
                            }
                        }
                    });
                    ui.add_space(6.0);
                }
            });
    }

    fn node_attribute_coloring(
        &self,
        network: &SpectralNetwork,
        visible: &HashSet<usize>,
    ) -> Option<NodeColoring> {
        if !self.color_nodes_by_attribute {
            return None;
        }
        let table = self.node_attributes.as_ref()?;
        let column_count = table.table.columns.len();
        if column_count == 0 {
            return None;
        }
        let column_idx = self
            .node_color_attr_column
            .min(column_count.saturating_sub(1));

        let mut colors = HashMap::with_capacity(visible.len());
        for node in &network.nodes {
            if visible.contains(&node.id) {
                colors.insert(node.id, egui::Color32::from_gray(120));
            }
        }

        let mut values: Vec<(usize, String)> = Vec::new();
        values.reserve(visible.len());
        for node in &network.nodes {
            if !visible.contains(&node.id) {
                continue;
            }
            let Some(key) = self.node_attribute_key_for_node(node) else {
                continue;
            };
            let Some(row) = table.find_row(&key) else {
                continue;
            };
            let Some(raw_value) = row.get(column_idx) else {
                continue;
            };
            let value = raw_value.trim();
            if value.is_empty() {
                continue;
            }
            values.push((node.id, value.to_string()));
        }

        if values.is_empty() {
            return Some(NodeColoring {
                fills: colors,
                mode: NodeColorMode::Categorical { legend: Vec::new() },
            });
        }

        let mut numeric_values: Vec<(usize, f64)> = Vec::with_capacity(values.len());
        let mut all_numeric = true;
        for (node_id, value) in &values {
            match value.parse::<f64>() {
                Ok(num) => numeric_values.push((*node_id, num)),
                Err(_) => {
                    all_numeric = false;
                    break;
                }
            }
        }

        if all_numeric {
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for (_, value) in &numeric_values {
                min_value = min_value.min(*value);
                max_value = max_value.max(*value);
            }
            let span = (max_value - min_value).abs();
            if span <= f64::EPSILON {
                for (node_id, _) in numeric_values {
                    colors.insert(node_id, diverging_color(0.5));
                }
            } else {
                for (node_id, value) in numeric_values {
                    let t = ((value - min_value) / (max_value - min_value)) as f32;
                    colors.insert(node_id, diverging_color(t));
                }
            }
            return Some(NodeColoring {
                fills: colors,
                mode: NodeColorMode::Continuous,
            });
        }

        let mut categories: Vec<String> = values.iter().map(|(_, value)| value.clone()).collect();
        categories.sort();
        categories.dedup();
        let legend = categories
            .iter()
            .enumerate()
            .map(|(idx, category)| (category.clone(), categorical_color(idx)))
            .collect::<Vec<_>>();
        let category_to_index: HashMap<String, usize> = categories
            .into_iter()
            .enumerate()
            .map(|(idx, category)| (category, idx))
            .collect();

        for (node_id, value) in values {
            if let Some(index) = category_to_index.get(&value) {
                colors.insert(node_id, categorical_color(*index));
            }
        }

        Some(NodeColoring {
            fills: colors,
            mode: NodeColorMode::Categorical { legend },
        })
    }

    fn draw_attribute_table_setup(
        ui: &mut egui::Ui,
        title: &str,
        table: &mut Option<LoadedAttributeTable>,
        show_match_field: bool,
        node_attr_match_field: &mut NodeAttrMatchField,
    ) {
        ui.group(|ui| {
            ui.label(title);
            if let Some(loaded) = table.as_mut() {
                ui.small(format!(
                    "{} | rows={} cols={}",
                    loaded.source_label,
                    loaded.table.rows.len(),
                    loaded.table.columns.len()
                ));
                let mut key_col = loaded.key_column();
                egui::ComboBox::from_label("Key column")
                    .selected_text(
                        loaded
                            .table
                            .columns
                            .get(key_col)
                            .cloned()
                            .unwrap_or_else(|| "<invalid>".to_string()),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, col) in loaded.table.columns.iter().enumerate() {
                            ui.selectable_value(&mut key_col, idx, col);
                        }
                    });
                if key_col != loaded.key_column() {
                    loaded.set_key_column(key_col);
                }

                if show_match_field {
                    egui::ComboBox::from_label("Node match field")
                        .selected_text(node_attr_match_field.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                node_attr_match_field,
                                NodeAttrMatchField::NodeId,
                                NodeAttrMatchField::NodeId.label(),
                            );
                            ui.selectable_value(
                                node_attr_match_field,
                                NodeAttrMatchField::FeatureId,
                                NodeAttrMatchField::FeatureId.label(),
                            );
                            ui.selectable_value(
                                node_attr_match_field,
                                NodeAttrMatchField::RawName,
                                NodeAttrMatchField::RawName.label(),
                            );
                            ui.selectable_value(
                                node_attr_match_field,
                                NodeAttrMatchField::Label,
                                NodeAttrMatchField::Label.label(),
                            );
                        });
                }
            } else {
                ui.small("No table loaded");
            }
        });
    }

    fn draw_node_attributes_panel(&mut self, ui: &mut egui::Ui, panel_id_suffix: &str) {
        ui.horizontal(|ui| {
            let label = if self.show_node_attributes_panel {
                "Hide node attributes"
            } else {
                "Show node attributes"
            };
            if ui.button(label).clicked() {
                self.show_node_attributes_panel = !self.show_node_attributes_panel;
            }
            if let Some(table) = &self.node_attributes {
                ui.small(format!(
                    "Node TSV: {} rows | key={} | match={}",
                    table.table.rows.len(),
                    table
                        .table
                        .columns
                        .get(table.key_column())
                        .map_or("<none>", String::as_str),
                    self.node_attr_match_field.label()
                ));
            } else {
                ui.small("Load a node attributes TSV from the left panel.");
            }
        });

        if !self.show_node_attributes_panel {
            return;
        }

        let Some(network) = &self.network else {
            ui.label("No network yet.");
            return;
        };
        let Some(table) = &self.node_attributes else {
            ui.label("No node attributes table loaded.");
            return;
        };
        let column_names = table.table.columns.clone();

        if let Some(filter_len) = self.table_node_filter.as_ref().map(|f| f.len()) {
            ui.horizontal(|ui| {
                ui.small(format!("Rectangle filter: {} node(s)", filter_len));
                if ui.button("Clear rectangle filter").clicked() {
                    self.table_node_filter = None;
                }
            });
        }

        let visible_ids =
            visible_node_ids_for_view(network, self.component_selection, self.hide_singletons);
        let node_by_id: HashMap<usize, &crate::network::NetworkNode> =
            network.nodes.iter().map(|n| (n.id, n)).collect();
        let feature_to_node_id: HashMap<String, usize> = network
            .nodes
            .iter()
            .filter_map(|n| {
                n.feature_id.as_deref().and_then(|fid| {
                    let trimmed = fid.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some((trimmed.to_string(), n.id))
                    }
                })
            })
            .collect();
        let mut matched_rows: Vec<(usize, usize)> = Vec::new();
        for node_id in &visible_ids {
            let Some(node) = node_by_id.get(node_id) else {
                continue;
            };
            if self.hide_singletons && node.degree == 0 {
                continue;
            }
            if let Some(filter) = &self.table_node_filter
                && !filter.contains(node_id)
            {
                continue;
            }
            let Some(key) = self.node_attribute_key_for_node(node) else {
                continue;
            };
            if let Some(row_idx) = table.find_row_index(&key) {
                matched_rows.push((*node_id, row_idx));
            }
        }

        ui.horizontal(|ui| {
            ui.label("Search");
            ui.add(
                egui::TextEdit::singleline(&mut self.node_attr_search_query)
                    .hint_text("Filter matched rows..."),
            );
            if ui.button("Clear search").clicked() {
                self.node_attr_search_query.clear();
            }
        });

        let query = self.node_attr_search_query.trim().to_ascii_lowercase();
        let matched_count = matched_rows.len();
        let mut filtered_rows: Vec<(usize, usize)> = matched_rows
            .iter()
            .copied()
            .filter(|(node_id, row_idx)| {
                let node_id_text = node_id.to_string();
                let node_id_lower = node_id_text.to_ascii_lowercase();
                let Some(row) = table.row(*row_idx) else {
                    return false;
                };

                if query.is_empty() {
                    return true;
                }
                if node_id_lower.contains(&query) {
                    return true;
                }
                row.iter()
                    .any(|value| value.to_ascii_lowercase().contains(&query))
            })
            .collect();

        filtered_rows.sort_by(
            |(left_node_id, left_row_idx), (right_node_id, right_row_idx)| {
                let ordering = match self.node_attr_sort_column {
                    NodeAttrSortColumn::NodeId => left_node_id.cmp(right_node_id),
                    NodeAttrSortColumn::TableColumn(col_idx) => {
                        let left_value = table
                            .row(*left_row_idx)
                            .and_then(|row| row.get(col_idx))
                            .map_or("", String::as_str);
                        let right_value = table
                            .row(*right_row_idx)
                            .and_then(|row| row.get(col_idx))
                            .map_or("", String::as_str);
                        compare_node_attr_values(left_value, right_value)
                    }
                };

                if self.node_attr_sort_ascending {
                    ordering
                } else {
                    ordering.reverse()
                }
            },
        );

        ui.small(format!(
            "Matched rows for visible nodes: {} / {} (shown: {})",
            matched_count,
            visible_ids.len(),
            filtered_rows.len()
        ));

        let selected_row_idx = self
            .selected_node_id
            .and_then(|node_id| node_by_id.get(&node_id).copied())
            .and_then(|node| self.node_attribute_key_for_node(node))
            .and_then(|key| table.find_row_index(&key));

        if let Some(row) = selected_row_idx.and_then(|row_idx| table.row(row_idx)) {
            ui.collapsing("Selected node attributes", |ui| {
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    egui::Grid::new("selected_node_attr_grid")
                        .striped(true)
                        .show(ui, |ui| {
                            for (col, value) in table.table.columns.iter().zip(row.iter()) {
                                ui.label(col);
                                ui.label(value);
                                ui.end_row();
                            }
                        });
                });
            });
        }

        ui.separator();
        if filtered_rows.is_empty() {
            ui.label("No visible node matched the current key/match-field selection.");
        }

        let table_height = (ui.available_height() * 0.58).clamp(120.0, 420.0);
        let text_height = ui.text_style_height(&egui::TextStyle::Body).max(18.0);
        let sort_column = self.node_attr_sort_column;
        let sort_ascending = self.node_attr_sort_ascending;
        let feature_id_column_idx = column_names
            .iter()
            .position(|name| normalized_column_name(name) == "featureid");
        let mut clicked_sort: Option<NodeAttrSortColumn> = None;
        let mut clicked_feature_id: Option<String> = None;
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), table_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                egui::ScrollArea::horizontal()
                    .id_salt(format!("node_attr_horizontal_scroll_{panel_id_suffix}"))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let mut table_builder = TableBuilder::new(ui)
                            .striped(true)
                            .resizable(true)
                            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                            .min_scrolled_height(0.0)
                            .max_scroll_height(table_height)
                            .vscroll(true);

                        for _ in &table.table.columns {
                            table_builder = table_builder.column(
                                Column::initial(170.0)
                                    .at_least(120.0)
                                    .clip(false)
                                    .resizable(true),
                            );
                        }

                        table_builder
                            .header(text_height + 4.0, |mut header| {
                                for (col_idx, col_name) in column_names.iter().enumerate() {
                                    header.col(|ui| {
                                        let sort_col = NodeAttrSortColumn::TableColumn(col_idx);
                                        let label = sort_header_label(
                                            sort_column,
                                            sort_ascending,
                                            sort_col,
                                            col_name,
                                        );
                                        if ui
                                            .add(
                                                egui::Label::new(label)
                                                    .truncate()
                                                    .sense(egui::Sense::click()),
                                            )
                                            .clicked()
                                        {
                                            clicked_sort = Some(sort_col);
                                        }
                                    });
                                }
                            })
                            .body(|body| {
                                body.rows(text_height + 2.0, filtered_rows.len(), |mut row| {
                                    let idx = row.index();
                                    let (node_id, row_idx) = filtered_rows[idx];
                                    let maybe_row = table.row(row_idx);
                                    let is_selected = Some(node_id) == self.selected_node_id;

                                    for col_idx in 0..column_names.len() {
                                        row.col(|ui| {
                                            let value = maybe_row
                                                .and_then(|r| r.get(col_idx))
                                                .map_or("", String::as_str);
                                            if Some(col_idx) == feature_id_column_idx {
                                                if !value.trim().is_empty()
                                                    && ui.link(value).clicked()
                                                {
                                                    clicked_feature_id = Some(value.to_string());
                                                }
                                            } else if is_selected {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(220, 180, 80),
                                                    value,
                                                );
                                            } else {
                                                ui.add(egui::Label::new(value).truncate());
                                            }
                                        });
                                    }
                                });
                            });
                    });
            },
        );

        if let Some(clicked) = clicked_sort {
            if self.node_attr_sort_column == clicked {
                self.node_attr_sort_ascending = !self.node_attr_sort_ascending;
            } else {
                self.node_attr_sort_column = clicked;
                self.node_attr_sort_ascending = true;
            }
            ui.ctx().request_repaint();
        }

        if let Some(feature_id) = clicked_feature_id {
            let feature_id = feature_id.trim().to_string();
            if !feature_id.is_empty() {
                let maybe_node_id = feature_to_node_id.get(&feature_id).copied();
                if let Some(node_id) = maybe_node_id {
                    self.selected_node_id = Some(node_id);
                    self.pending_center_node_id = Some(node_id);
                    self.structure_panel_mode = StructurePanelMode::SelectedNode;
                    self.status_message =
                        Some(format!("Selected feature_id={feature_id} (node {node_id})"));
                    self.error_message = None;
                } else {
                    self.error_message =
                        Some(format!("Feature ID not found in network: {feature_id}"));
                }
                ui.ctx().request_repaint();
            }
        }
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Spectral Network");
        ui.separator();

        ui.collapsing("Input", |ui| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label("MGF path (native)");
                ui.text_edit_singleline(&mut self.mgf_path);
                ui.horizontal(|ui| {
                    if ui.button("Load path").clicked() {
                        self.load_from_path();
                    }

                    if ui.button("Pick file").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("MGF", &["mgf"])
                            .pick_file()
                    {
                        self.mgf_path = path.display().to_string();
                        self.load_from_path();
                    }
                });
            }

            #[cfg(target_arch = "wasm32")]
            {
                if ui.button("Upload MGF").clicked() {
                    self.start_upload_dialog();
                }
                if self.upload_promise.is_some() {
                    ui.label("Waiting for file selection...");
                    ctx.request_repaint();
                }
            }

            if let Some(source) = &self.source_label {
                ui.label(format!("Source: {source}"));
            }
            ui.label(format!("Parsed spectra: {}", self.spectra.len()));
            if let Some(stats) = self.parse_stats {
                ui.label(format!(
                    "Accepted={} / Blocks={} / MissingName={} / MissingPrecursor={} / TooFew={} / TooMany={} / DuplicateMz={}",
                    stats.accepted,
                    stats.ions_blocks,
                    stats.dropped_missing_name,
                    stats.dropped_missing_precursor_mz,
                    stats.dropped_too_few_peaks,
                    stats.dropped_too_many_peaks,
                    stats.dropped_duplicate_mz,
                ));
            }
        });

        ui.separator();
        ui.collapsing("Attributes TSV", |ui| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.horizontal(|ui| {
                    if ui.button("Load node attributes TSV").clicked() {
                        self.pick_and_load_attributes(true);
                    }
                    if ui.button("Load edge attributes TSV").clicked() {
                        self.pick_and_load_attributes(false);
                    }
                });
            }

            #[cfg(target_arch = "wasm32")]
            {
                ui.small("TSV attribute loading is currently native-only.");
            }

            Self::draw_attribute_table_setup(
                ui,
                "Node attributes",
                &mut self.node_attributes,
                true,
                &mut self.node_attr_match_field,
            );
            Self::draw_attribute_table_setup(
                ui,
                "Edge attributes",
                &mut self.edge_attributes,
                false,
                &mut self.node_attr_match_field,
            );

            ui.group(|ui| {
                ui.label("Structure depiction");
                egui::ComboBox::from_label("Structure view mode")
                    .selected_text(self.structure_panel_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.structure_panel_mode,
                            StructurePanelMode::SelectedNode,
                            StructurePanelMode::SelectedNode.label(),
                        );
                        ui.selectable_value(
                            &mut self.structure_panel_mode,
                            StructurePanelMode::SelectedSet,
                            StructurePanelMode::SelectedSet.label(),
                        );
                    });
                if let Some(table) = &self.node_attributes {
                    let selected_text = self
                        .depict_smiles_column
                        .and_then(|idx| table.table.columns.get(idx))
                        .cloned()
                        .unwrap_or_else(|| "<none>".to_string());
                    egui::ComboBox::from_label("SMILES column")
                        .selected_text(selected_text)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.depict_smiles_column, None, "<none>");
                            for (idx, col) in table.table.columns.iter().enumerate() {
                                ui.selectable_value(&mut self.depict_smiles_column, Some(idx), col);
                            }
                        });
                    self.structure_caption_columns
                        .retain(|idx| *idx < table.table.columns.len());
                    ui.separator();
                    ui.label("Text columns shown under each depicted structure");
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .show(ui, |ui| {
                            for (idx, col) in table.table.columns.iter().enumerate() {
                                if Some(idx) == self.depict_smiles_column {
                                    continue;
                                }
                                let mut selected = self.structure_caption_columns.contains(&idx);
                                if ui.checkbox(&mut selected, col).changed() {
                                    if selected {
                                        if !self.structure_caption_columns.contains(&idx) {
                                            self.structure_caption_columns.push(idx);
                                        }
                                    } else {
                                        self.structure_caption_columns
                                            .retain(|existing| *existing != idx);
                                    }
                                }
                            }
                        });
                    self.structure_caption_columns.sort_unstable();
                    ui.small(
                        "Selected SMILES values are rendered in the right panel via api.naturalproducts.net.",
                    );
                } else {
                    ui.small("Load a node attributes TSV to choose a SMILES column.");
                }
                ui.small(
                    "Modes are exclusive: either depict one selected node or depict structures from rectangle-selected nodes.",
                );
            });

            ui.separator();
            ui.checkbox(
                &mut self.show_node_attributes_panel,
                "Show node attributes table panel",
            );
            egui::ComboBox::from_label("Table docking")
                .selected_text(self.node_attr_panel_dock.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.node_attr_panel_dock,
                        NodeAttrPanelDock::Bottom,
                        NodeAttrPanelDock::Bottom.label(),
                    );
                    ui.selectable_value(
                        &mut self.node_attr_panel_dock,
                        NodeAttrPanelDock::Right,
                        NodeAttrPanelDock::Right.label(),
                    );
                    ui.selectable_value(
                        &mut self.node_attr_panel_dock,
                        NodeAttrPanelDock::Detached,
                        NodeAttrPanelDock::Detached.label(),
                    );
                });
            match self.node_attr_panel_dock {
                NodeAttrPanelDock::Bottom => {
                    ui.horizontal(|ui| {
                        ui.label("Bottom panel height");
                        ui.add(
                            egui::DragValue::new(&mut self.node_attr_bottom_height)
                                .range(140.0..=1200.0)
                                .speed(1.0),
                        );
                    });
                }
                NodeAttrPanelDock::Right => {
                    ui.horizontal(|ui| {
                        ui.label("Right panel width");
                        ui.add(
                            egui::DragValue::new(&mut self.node_attr_right_width)
                                .range(220.0..=1800.0)
                                .speed(1.0),
                        );
                    });
                }
                NodeAttrPanelDock::Detached => {}
            }
        });

        let style_preview = self.network.as_ref().and_then(|network| {
            let visible =
                visible_node_set_for_view(network, self.component_selection, self.hide_singletons);
            self.node_attribute_coloring(network, &visible)
        });

        ui.separator();
        ui.collapsing("Style", |ui| {
            ui.checkbox(
                &mut self.color_nodes_by_attribute,
                "Color nodes from node attribute column",
            );
            if self.color_nodes_by_attribute {
                if let Some(table) = &self.node_attributes {
                    if table.table.columns.is_empty() {
                        ui.small("Loaded node table has no columns.");
                    } else {
                        if self.node_color_attr_column >= table.table.columns.len() {
                            self.node_color_attr_column = 0;
                        }
                        let selected_column = table
                            .table
                            .columns
                            .get(self.node_color_attr_column)
                            .cloned()
                            .unwrap_or_else(|| "<invalid>".to_string());
                        egui::ComboBox::from_label("Color column")
                            .selected_text(selected_column)
                            .show_ui(ui, |ui| {
                                for (idx, col_name) in table.table.columns.iter().enumerate() {
                                    ui.selectable_value(&mut self.node_color_attr_column, idx, col_name);
                                }
                            });
                        ui.small(
                            "Categorical values use a qualitative palette; numeric values use a divergent gradient.",
                        );

                        if let Some(preview) = style_preview.as_ref() {
                            match &preview.mode {
                                NodeColorMode::Categorical { legend } => {
                                    ui.checkbox(
                                        &mut self.show_categorical_legend,
                                        "Show categorical legend",
                                    );
                                    if self.show_categorical_legend {
                                        if legend.is_empty() {
                                            ui.small("No category values found in visible nodes.");
                                        } else {
                                            ui.separator();
                                            ui.small("Legend");
                                            egui::ScrollArea::vertical()
                                                .max_height(180.0)
                                                .show(ui, |ui| {
                                                    for (category, color) in legend {
                                                        ui.horizontal(|ui| {
                                                            ui.colored_label(*color, "■");
                                                            ui.label(category);
                                                        });
                                                    }
                                                });
                                        }
                                    }
                                }
                                NodeColorMode::Continuous => {
                                    ui.small("Selected column is numeric; continuous palette applied.");
                                }
                            }
                        }
                    }
                } else {
                    ui.small("Load a node attributes TSV to enable attribute-based colors.");
                }
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Metric")
                .selected_text(self.selected_metric.label())
                .show_ui(ui, |ui| {
                    for metric in SimilarityMetric::ALL {
                        ui.selectable_value(&mut self.selected_metric, metric, metric.label());
                    }
                });

            let can_start = !self.is_computing() && !self.spectra.is_empty();
            if ui
                .add_enabled(
                    can_start,
                    egui::Button::new(format!("Run {}", self.selected_metric.label())),
                )
                .clicked()
            {
                self.start_compute();
            }
            if self.is_computing() && ui.button("Cancel").clicked() {
                self.cancel_compute();
            }
        });

        ui.collapsing("Similarity Params (required)", |ui| {
            ui.horizontal(|ui| {
                ui.label("tolerance");
                ui.text_edit_singleline(&mut self.tolerance_input);
            });
            ui.horizontal(|ui| {
                ui.label("mz_power");
                ui.text_edit_singleline(&mut self.mz_power_input);
            });
            ui.horizontal(|ui| {
                ui.label("intensity_power");
                ui.text_edit_singleline(&mut self.intensity_power_input);
            });
        });

        if self.is_computing()
            && let Some((done, total)) = self.compute_progress()
        {
            let frac = if total == 0 {
                0.0
            } else {
                done as f32 / total as f32
            };
            ui.add(egui::ProgressBar::new(frac).text(format!("{done}/{total}")));
        }

        ui.separator();
        ui.label("Network Controls");
        let mut changed = false;
        let mut layout_changed = false;
        ui.horizontal(|ui| {
            ui.label("Similarity threshold");
            changed |= ui
                .add(egui::Slider::new(&mut self.threshold, 0.0..=1.0).show_value(false))
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.threshold)
                        .range(0.0..=1.0)
                        .speed(0.01),
                )
                .changed();
        });
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.top_k)
                    .range(1..=500)
                    .prefix("top-k "),
            )
            .changed();
        let hide_singletons_changed = ui
            .checkbox(&mut self.hide_singletons, "Hide singleton nodes")
            .changed();
        layout_changed |= ui
            .add(egui::Slider::new(&mut self.node_force, 0.1..=5.0).text("Node force"))
            .changed();
        layout_changed |= ui
            .add(egui::Slider::new(&mut self.edge_force, 0.1..=5.0).text("Edge force"))
            .changed();

        if changed {
            self.threshold = self.threshold.clamp(0.0, 1.0);
            self.rebuild_network();
        }
        if hide_singletons_changed && let Some(network) = &self.network {
            self.selected_node_id = keep_selected_if_visible(
                self.selected_node_id,
                network,
                self.component_selection,
                self.hide_singletons,
            );
            self.layout_running = false;
            self.request_fit_view = true;
        }
        if layout_changed {
            let _ = self.relayout_visible(50);
        }

        ui.horizontal(|ui| {
            let run_label = if self.layout_running {
                "Pause layout"
            } else {
                "Run layout"
            };
            if ui.button(run_label).clicked() {
                self.layout_running = !self.layout_running;
                self.layout_low_motion_streak = 0;
                if self.layout_running {
                    ctx.request_repaint();
                }
            }
            if ui.button("Stabilize").clicked() {
                let _ = self.relayout_visible(self.stabilize_iterations);
                self.layout_running = false;
                self.layout_low_motion_streak = 0;
            }
            ui.add(
                egui::DragValue::new(&mut self.stabilize_iterations)
                    .range(10..=800)
                    .prefix("iters "),
            );
        });
        ui.horizontal(|ui| {
            if ui.button("Fit full network").clicked() {
                self.layout_running = false;
                self.component_selection = ComponentSelection::All;
                self.request_fit_view = true;
            }
        });
        ui.small(format!(
            "Layout: {} | mean displacement: {:.5}",
            if self.layout_running {
                "running"
            } else {
                "paused"
            },
            self.layout_mean_displacement
        ));

        if self.layout_running {
            ctx.request_repaint();
        }
        if let Some(network) = &self.network {
            let visible_now =
                visible_node_set_for_view(network, self.component_selection, self.hide_singletons);
            ui.small(format!(
                "Active threshold: {:.3} | active edges: {}",
                self.threshold,
                network
                    .edges
                    .iter()
                    .filter(|e| visible_now.contains(&e.source) && visible_now.contains(&e.target))
                    .count()
            ));
        }

        if let Some(network) = &self.network {
            let mut selection = self.component_selection;
            egui::ComboBox::from_label("Component scope")
                .selected_text(selection_label(network, selection))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selection, ComponentSelection::All, "Full network");
                    ui.selectable_value(
                        &mut selection,
                        ComponentSelection::Largest,
                        "Largest component",
                    );
                    for (idx, nodes) in network.components.iter().enumerate() {
                        ui.selectable_value(
                            &mut selection,
                            ComponentSelection::Component(idx),
                            format!("Component {idx} ({} nodes)", nodes.len()),
                        );
                    }
                });

            if selection != self.component_selection {
                self.component_selection = selection;
                self.selected_node_id = keep_selected_if_visible(
                    self.selected_node_id,
                    network,
                    self.component_selection,
                    self.hide_singletons,
                );
                self.layout_running = false;
                self.request_fit_view = true;
            }
        }

        ui.separator();
        ui.collapsing("Export", |ui| {
            let can_export = self.network.is_some();
            if ui
                .add_enabled(can_export, egui::Button::new("Export nodes/edges CSV"))
                .clicked()
                && let Some(network) = &self.network
            {
                let (nodes_csv, edges_csv) = export_csv_strings(network, self.component_selection);

                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        match crate::export::save_csvs_to_directory(&dir, &nodes_csv, &edges_csv) {
                            Ok(()) => {
                                self.status_message = Some(format!(
                                    "Exported nodes.csv and edges.csv into {}",
                                    dir.display()
                                ));
                            }
                            Err(err) => {
                                self.error_message = Some(err);
                            }
                        }
                    }
                }

                #[cfg(target_arch = "wasm32")]
                {
                    if let Err(err) = crate::export::download_csv_files(&nodes_csv, &edges_csv) {
                        self.error_message = Some(err);
                    } else {
                        self.status_message = Some("Triggered CSV download".to_string());
                    }
                }
            }
        });

        if let Some(status) = &self.status_message {
            ui.separator();
            ui.label(status);
        }

        if let Some(err) = &self.error_message {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(190, 50, 50), err);
        }
    }

    fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        if self.network.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label(format!(
                    "Load spectra and run {} to render a spectral network.",
                    self.selected_metric.label()
                ));
            });
            return;
        }

        let visible_set = {
            let network = self.network.as_ref().expect("network presence checked");
            visible_node_set_for_view(network, self.component_selection, self.hide_singletons)
        };
        let visible = {
            let network = self.network.as_ref().expect("network presence checked");
            visible_node_ids_for_view(network, self.component_selection, self.hide_singletons)
        };
        let canvas_size = ui.available_size_before_wrap();

        if self.positions.is_empty() {
            let network = self.network.as_ref().expect("network presence checked");
            let initial = force_directed_layout(
                network,
                &visible,
                &self.positions,
                120,
                self.node_force,
                self.edge_force,
            );
            self.layout_mean_displacement = initial.mean_displacement;
            for (node_id, pos) in initial.positions {
                self.positions.insert(node_id, pos);
            }
        }
        if self.request_fit_view {
            self.fit_view_to_visible_nodes(canvas_size, &visible);
            self.request_fit_view = false;
        }
        if let Some(node_id) = self.pending_center_node_id.take()
            && visible_set.contains(&node_id)
        {
            let component_members = self.network.as_ref().and_then(|net| {
                let component_id = net
                    .nodes
                    .iter()
                    .find(|n| n.id == node_id)
                    .map(|n| n.component_id)?;
                net.components.get(component_id).cloned()
            });
            let focused = if let Some(component_members) = component_members {
                self.focus_view_on_node_set(canvas_size, &component_members, node_id)
            } else {
                self.center_view_on_node(canvas_size, node_id)
            };
            if !focused {
                self.pending_center_node_id = Some(node_id);
            }
        }

        let network = self.network.as_ref().expect("network presence checked");
        let node_coloring = self.node_attribute_coloring(network, &visible_set);
        let interaction = draw_network(
            ui,
            network,
            &self.positions,
            &visible_set,
            node_coloring.as_ref().map(|coloring| &coloring.fills),
            &mut self.view_state,
            self.selected_node_id,
        );
        self.hovered_node_id = interaction.hovered_node_id;
        self.view_state.pan += interaction.pan_delta;
        if let Some(box_ids) = interaction.box_selected_node_ids {
            if box_ids.is_empty() {
                self.table_node_filter = None;
                self.status_message =
                    Some("Rectangle selection is empty; table filter cleared".to_string());
            } else {
                let filter: HashSet<usize> = box_ids.iter().copied().collect();
                self.table_node_filter = Some(filter);
                self.structure_panel_mode = StructurePanelMode::SelectedSet;
                self.status_message = Some(format!(
                    "Rectangle selected {} node(s); table filtered",
                    box_ids.len()
                ));
                if box_ids.len() == 1 {
                    self.selected_node_id = box_ids.first().copied();
                    self.pending_center_node_id = self.selected_node_id;
                } else if self
                    .selected_node_id
                    .is_some_and(|selected| !box_ids.contains(&selected))
                {
                    self.selected_node_id = None;
                }
            }
            ui.ctx().request_repaint();
        }
        if let Some((node_id, delta)) = interaction.dragged_node
            && let Some(pos) = self.positions.get_mut(&node_id)
        {
            pos[0] += delta[0];
            pos[1] += delta[1];
            ui.ctx().request_repaint();
        }
        if interaction.fit_full_network_requested {
            self.layout_running = false;
            self.component_selection = ComponentSelection::All;
            self.pending_center_node_id = None;
            self.request_fit_view = true;
            ui.ctx().request_repaint();
        } else if let Some(node_id) = interaction.clicked_node_id {
            self.selected_node_id = Some(node_id);
            self.pending_center_node_id = Some(node_id);
            self.structure_panel_mode = StructurePanelMode::SelectedNode;
            ui.ctx().request_repaint();
        } else if interaction.clicked_empty_canvas {
            self.selected_node_id = None;
        }
        if self.layout_running
            && interaction.dragged_node.is_none()
            && interaction.pan_delta.length_sq() <= f32::EPSILON
            && !visible.is_empty()
        {
            let relaxed = force_directed_layout(
                network,
                &visible,
                &self.positions,
                2,
                self.node_force,
                self.edge_force,
            );
            self.layout_mean_displacement = relaxed.mean_displacement;
            for (node_id, pos) in relaxed.positions {
                self.positions.insert(node_id, pos);
            }
            if self.layout_mean_displacement <= LAYOUT_STOP_EPSILON {
                self.layout_low_motion_streak += 1;
            } else {
                self.layout_low_motion_streak = 0;
            }
            if self.layout_low_motion_streak >= LAYOUT_STOP_STREAK {
                self.layout_running = false;
                self.layout_low_motion_streak = 0;
                self.status_message = Some(format!(
                    "Layout auto-paused (mean displacement {:.5})",
                    self.layout_mean_displacement
                ));
            } else {
                ui.ctx().request_repaint();
            }
        } else if self.layout_running {
            self.layout_low_motion_streak = 0;
        }

        ui.separator();
        let visible_nodes = visible.clone();
        let visible_edges = network
            .edges
            .iter()
            .filter(|e| visible_set.contains(&e.source) && visible_set.contains(&e.target))
            .count();
        ui.label(format!(
            "Visible: {} node(s), {} edge(s), {} component(s) total",
            visible_nodes.len(),
            visible_edges,
            network.components.len()
        ));

        if let Some(node_id) = self.hovered_node_id
            && let Some(node) = network.nodes.iter().find(|n| n.id == node_id)
        {
            ui.label(format!(
                "Hover node {} | label={} | raw={} | precursor_mz={:.4} | peaks={} | degree={} | component={}",
                node.id,
                node.label,
                node.raw_name,
                node.precursor_mz,
                node.num_peaks,
                node.degree,
                node.component_id
            ));
        }
    }

    fn draw_selected_node_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Node Properties");
        ui.separator();

        let Some(network) = &self.network else {
            ui.label("No network loaded.");
            return;
        };

        let selected_node = self.selected_visible_node(network);
        if let Some(node) = selected_node.as_ref() {
            ui.monospace(format!("Node index: {}", node.id));
            ui.horizontal(|ui| {
                ui.monospace("Feature ID:");
                if let Some(feature_id) = node.feature_id.as_deref() {
                    if ui.link(feature_id).clicked() {
                        self.pending_center_node_id = Some(node.id);
                    }
                } else {
                    ui.monospace("n/a");
                }
            });
            ui.monospace(format!("Display label: {}", node.label));
            ui.monospace(format!("Raw name: {}", node.raw_name));
            ui.monospace(format!(
                "Parent mass (precursor m/z): {:.6}",
                node.precursor_mz
            ));
            ui.monospace(format!("Peak count: {}", node.num_peaks));
            ui.monospace(format!("Degree: {}", node.degree));
            ui.monospace(format!("Component: {}", node.component_id));
            ui.monospace(format!("Scans: {}", node.scans.as_deref().unwrap_or("n/a")));
            ui.monospace(format!(
                "Filename: {}",
                node.filename.as_deref().unwrap_or("n/a")
            ));
            ui.monospace(format!(
                "Source scan USI: {}",
                node.source_scan_usi.as_deref().unwrap_or("n/a")
            ));
            ui.monospace(format!(
                "Featurelist feature ID: {}",
                node.featurelist_feature_id.as_deref().unwrap_or("n/a")
            ));
        } else if self.selected_node_id.is_some() {
            ui.label("Selected node is not visible in current scope.");
        } else {
            ui.label("Select a node in the graph.");
        }

        ui.separator();
        ui.label("Structure");
        if self.node_attributes.is_none() {
            ui.small("No node attributes table loaded.");
            return;
        }
        let Some(smiles_col_idx) = self.depict_smiles_column else {
            ui.small("Select a SMILES column in the left panel.");
            return;
        };
        let Some(col_name) = self
            .node_attributes
            .as_ref()
            .and_then(|table| table.table.columns.get(smiles_col_idx))
            .cloned()
        else {
            ui.small("Selected SMILES column is out of range.");
            return;
        };

        ui.horizontal(|ui| {
            ui.label("Mode:");
            egui::ComboBox::from_id_salt("structure_mode_right_panel")
                .selected_text(self.structure_panel_mode.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.structure_panel_mode,
                        StructurePanelMode::SelectedNode,
                        StructurePanelMode::SelectedNode.label(),
                    );
                    ui.selectable_value(
                        &mut self.structure_panel_mode,
                        StructurePanelMode::SelectedSet,
                        StructurePanelMode::SelectedSet.label(),
                    );
                });
        });

        match self.structure_panel_mode {
            StructurePanelMode::SelectedNode => {
                self.show_selection_structures_detached = false;
                let Some(node) = selected_node.as_ref() else {
                    ui.small("Select a visible node to depict its structure.");
                    return;
                };
                let Some(smiles) = self.selected_node_smiles(node) else {
                    ui.small(format!(
                        "No SMILES found for this node using column '{col_name}'."
                    ));
                    return;
                };

                ui.small(format!("SMILES column: {col_name}"));
                ui.small(format!("SMILES: {smiles}"));
                let img_width = ui.available_width().clamp(180.0, 480.0);
                let img_height = (img_width * 0.7).clamp(120.0, 360.0);
                let depict_uri =
                    naturalproducts_depict_uri(&smiles, img_width as u32, img_height as u32);
                self.ensure_depiction_request(depict_uri.clone(), depict_uri.clone());
                self.draw_depiction_widget(ui, &depict_uri, egui::vec2(img_width, img_height));
                ui.hyperlink_to("Open structure in browser", depict_uri);
            }
            StructurePanelMode::SelectedSet => {
                ui.horizontal(|ui| {
                    ui.small(format!("SMILES column: {col_name}"));
                    if ui.button("Open in window").clicked() {
                        self.show_selection_structures_detached = true;
                    }
                });
                if let Some(filter_len) = self.table_node_filter.as_ref().map(HashSet::len) {
                    ui.small(format!("Rectangle-selected nodes: {filter_len}"));
                }
                ui.add_space(4.0);
                self.draw_selected_structures_gallery(ui);
            }
        }
    }
}

fn categorical_color(index: usize) -> egui::Color32 {
    const PALETTE: [egui::Color32; 12] = [
        egui::Color32::from_rgb(31, 119, 180),
        egui::Color32::from_rgb(255, 127, 14),
        egui::Color32::from_rgb(44, 160, 44),
        egui::Color32::from_rgb(214, 39, 40),
        egui::Color32::from_rgb(148, 103, 189),
        egui::Color32::from_rgb(140, 86, 75),
        egui::Color32::from_rgb(227, 119, 194),
        egui::Color32::from_rgb(127, 127, 127),
        egui::Color32::from_rgb(188, 189, 34),
        egui::Color32::from_rgb(23, 190, 207),
        egui::Color32::from_rgb(57, 106, 177),
        egui::Color32::from_rgb(218, 124, 48),
    ];
    PALETTE[index % PALETTE.len()]
}

fn diverging_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let low = egui::Color32::from_rgb(49, 54, 149);
    let mid = egui::Color32::from_rgb(246, 246, 246);
    let high = egui::Color32::from_rgb(165, 0, 38);
    if t <= 0.5 {
        lerp_color(low, mid, t * 2.0)
    } else {
        lerp_color(mid, high, (t - 0.5) * 2.0)
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 { (x as f32 + (y as f32 - x as f32) * t).round() as u8 };
    egui::Color32::from_rgba_unmultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

fn naturalproducts_depict_uri(smiles: &str, width: u32, height: u32) -> String {
    let encoded_smiles = urlencoding::encode(smiles);
    format!(
        "https://api.naturalproducts.net/latest/depict/2D_enhanced?smiles={encoded_smiles}&width={width}&height={height}&style=bow&zoom=2.2&abbreviate=off"
    )
}

fn detect_depict_image_extension(content_type: Option<&str>, bytes: &[u8]) -> Option<&'static str> {
    let ctype = content_type.unwrap_or_default().to_ascii_lowercase();
    if ctype.contains("svg") {
        return Some("svg");
    }
    if ctype.contains("png") {
        return Some("png");
    }
    if ctype.contains("jpeg") || ctype.contains("jpg") {
        return Some("jpg");
    }
    if ctype.contains("webp") {
        return Some("webp");
    }

    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("png");
    }
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return Some("jpg");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("webp");
    }
    let probe = &bytes[..bytes.len().min(512)];
    let probe_text = String::from_utf8_lossy(probe).to_ascii_lowercase();
    if probe_text.contains("<svg") {
        return Some("svg");
    }
    None
}

fn stable_hash(input: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

fn compare_node_attr_values(left: &str, right: &str) -> Ordering {
    let left_trimmed = left.trim();
    let right_trimmed = right.trim();
    let left_numeric = left_trimmed.parse::<f64>();
    let right_numeric = right_trimmed.parse::<f64>();
    if let (Ok(left_number), Ok(right_number)) = (left_numeric, right_numeric)
        && let Some(ordering) = left_number.partial_cmp(&right_number)
    {
        return ordering;
    }

    let left_lower = left_trimmed.to_ascii_lowercase();
    let right_lower = right_trimmed.to_ascii_lowercase();
    left_lower
        .cmp(&right_lower)
        .then_with(|| left_trimmed.cmp(right_trimmed))
}

fn sort_header_label(
    current: NodeAttrSortColumn,
    ascending: bool,
    target: NodeAttrSortColumn,
    base: &str,
) -> String {
    if current == target {
        if ascending {
            format!("{base} (asc)")
        } else {
            format!("{base} (desc)")
        }
    } else {
        base.to_string()
    }
}

fn normalized_column_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn default_structure_caption_columns(columns: &[String], smiles_col: Option<usize>) -> Vec<usize> {
    let mut picks = Vec::new();
    let mut seen = HashSet::new();
    for (idx, name) in columns.iter().enumerate() {
        if Some(idx) == smiles_col {
            continue;
        }
        let normalized = normalized_column_name(name);
        let preferred = normalized.contains("name")
            || normalized.contains("canopus")
            || normalized.contains("class")
            || normalized.contains("annotation")
            || normalized.contains("compound")
            || normalized.contains("superclass");
        if preferred && seen.insert(idx) {
            picks.push(idx);
        }
        if picks.len() >= 4 {
            return picks;
        }
    }

    for (idx, _name) in columns.iter().enumerate() {
        if Some(idx) == smiles_col || seen.contains(&idx) {
            continue;
        }
        picks.push(idx);
        if picks.len() >= 2 {
            break;
        }
    }
    picks
}

impl eframe::App for SpectralApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_arch = "wasm32")]
        self.poll_upload_dialog();

        self.poll_compute(ctx);
        self.poll_depiction(ctx);

        egui::SidePanel::left("controls_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                self.draw_controls(ui, ctx);
            });

        egui::SidePanel::right("selected_node_panel")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                self.draw_selected_node_panel(ui);
            });

        if self.node_attr_panel_dock == NodeAttrPanelDock::Right {
            egui::SidePanel::right("node_attributes_side_panel")
                .resizable(false)
                .exact_width(if self.show_node_attributes_panel {
                    self.node_attr_right_width
                } else {
                    58.0
                })
                .show(ctx, |ui| {
                    self.draw_node_attributes_panel(ui, "right");
                });
        }

        if self.node_attr_panel_dock == NodeAttrPanelDock::Bottom {
            egui::TopBottomPanel::bottom("node_attributes_bottom_panel")
                .resizable(false)
                .exact_height(if self.show_node_attributes_panel {
                    self.node_attr_bottom_height
                } else {
                    34.0
                })
                .show(ctx, |ui| {
                    self.draw_node_attributes_panel(ui, "bottom");
                });
        }

        if self.node_attr_panel_dock == NodeAttrPanelDock::Detached
            && self.show_node_attributes_panel
        {
            let mut open = true;
            egui::Window::new("Node Attributes Table")
                .id(egui::Id::new("node_attributes_detached_window"))
                .resizable(true)
                .default_size(egui::vec2(980.0, 340.0))
                .open(&mut open)
                .show(ctx, |ui| {
                    self.draw_node_attributes_panel(ui, "detached");
                });
            if !open {
                self.show_node_attributes_panel = false;
            }
        }

        if self.show_selection_structures_detached
            && self.structure_panel_mode == StructurePanelMode::SelectedSet
        {
            let mut open = true;
            egui::Window::new("Selected Structures")
                .id(egui::Id::new("selected_structures_window"))
                .resizable(true)
                .default_size(egui::vec2(920.0, 560.0))
                .open(&mut open)
                .show(ctx, |ui| {
                    self.draw_selected_structures_gallery(ui);
                });
            if !open {
                self.show_selection_structures_detached = false;
            }
        } else if self.show_selection_structures_detached {
            self.show_selection_structures_detached = false;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_canvas(ui);
        });
    }
}

fn selection_label(network: &SpectralNetwork, selection: ComponentSelection) -> String {
    match selection {
        ComponentSelection::All => "Full network".to_string(),
        ComponentSelection::Largest => "Largest component".to_string(),
        ComponentSelection::Component(idx) => {
            let size = network.components.get(idx).map_or(0, |nodes| nodes.len());
            format!("Component {idx} ({size} nodes)")
        }
    }
}

fn keep_selected_if_visible(
    selected_node_id: Option<usize>,
    network: &SpectralNetwork,
    selection: ComponentSelection,
    hide_singletons: bool,
) -> Option<usize> {
    let visible = visible_node_set_for_view(network, selection, hide_singletons);
    selected_node_id.filter(|selected| visible.contains(selected))
}

fn visible_node_set_for_view(
    network: &SpectralNetwork,
    selection: ComponentSelection,
    hide_singletons: bool,
) -> std::collections::HashSet<usize> {
    let node_by_id: std::collections::HashMap<usize, &crate::network::NetworkNode> =
        network.nodes.iter().map(|n| (n.id, n)).collect();

    let mut visible: std::collections::HashSet<usize> = match selection {
        ComponentSelection::All => network.nodes.iter().map(|n| n.id).collect(),
        ComponentSelection::Largest => {
            let largest_component = largest_component_for_view(network, hide_singletons);
            network
                .nodes
                .iter()
                .filter(|n| Some(n.component_id) == largest_component)
                .map(|n| n.id)
                .collect()
        }
        ComponentSelection::Component(component_id) => network
            .nodes
            .iter()
            .filter(|n| n.component_id == component_id)
            .map(|n| n.id)
            .collect(),
    };

    if hide_singletons {
        visible.retain(|id| node_by_id.get(id).is_some_and(|n| n.degree > 0));
    }
    visible
}

fn largest_component_for_view(network: &SpectralNetwork, hide_singletons: bool) -> Option<usize> {
    let node_by_id: std::collections::HashMap<usize, &crate::network::NetworkNode> =
        network.nodes.iter().map(|n| (n.id, n)).collect();
    network
        .components
        .iter()
        .enumerate()
        .map(|(component_id, members)| {
            let count = members
                .iter()
                .filter(|id| {
                    node_by_id
                        .get(id)
                        .is_some_and(|node| !hide_singletons || node.degree > 0)
                })
                .count();
            (component_id, count)
        })
        .filter(|(_, count)| *count > 0)
        .max_by_key(|(component_id, count)| (*count, std::cmp::Reverse(*component_id)))
        .map(|(component_id, _)| component_id)
}

fn visible_node_ids_for_view(
    network: &SpectralNetwork,
    selection: ComponentSelection,
    hide_singletons: bool,
) -> Vec<usize> {
    let mut visible: Vec<usize> = visible_node_set_for_view(network, selection, hide_singletons)
        .into_iter()
        .collect();
    visible.sort_unstable();
    visible
}

#[cfg(test)]
mod tests {
    use crate::compute::PairScore;
    use crate::io::SpectrumMeta;
    use crate::network::{ComponentSelection, build_network};

    use super::{keep_selected_if_visible, visible_node_ids_for_view};

    fn meta(id: usize) -> SpectrumMeta {
        SpectrumMeta {
            id,
            label: format!("s{id}"),
            raw_name: format!("raw{id}"),
            feature_id: Some(format!("f{id}")),
            scans: None,
            filename: None,
            source_scan_usi: None,
            featurelist_feature_id: None,
            precursor_mz: 100.0 + id as f64,
            num_peaks: 10,
        }
    }

    #[test]
    fn selection_survives_when_node_stays_visible() {
        let metas = vec![meta(0), meta(1), meta(2)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.9,
                matches: 2,
            },
            PairScore {
                left: 1,
                right: 2,
                score: 0.9,
                matches: 2,
            },
        ];
        let network = build_network(&metas, &scores, 0.2, 5);
        assert_eq!(
            keep_selected_if_visible(Some(1), &network, ComponentSelection::All, false),
            Some(1)
        );
    }

    #[test]
    fn selection_clears_when_filtered_out_by_component_scope() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.95,
                matches: 2,
            },
            PairScore {
                left: 2,
                right: 3,
                score: 0.95,
                matches: 2,
            },
        ];
        let network = build_network(&metas, &scores, 0.2, 5);
        assert_eq!(
            keep_selected_if_visible(Some(3), &network, ComponentSelection::Component(0), false),
            None
        );
    }

    #[test]
    fn hide_singletons_removes_isolated_nodes_from_visible_set() {
        let metas = vec![meta(0), meta(1), meta(2)];
        let scores = vec![PairScore {
            left: 0,
            right: 1,
            score: 0.95,
            matches: 2,
        }];
        let network = build_network(&metas, &scores, 0.2, 5);

        let visible = visible_node_ids_for_view(&network, ComponentSelection::All, true);
        assert_eq!(visible, vec![0, 1]);
    }

    #[test]
    fn largest_component_ignores_singleton_components_when_hidden() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3), meta(4)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.95,
                matches: 2,
            },
            PairScore {
                left: 1,
                right: 2,
                score: 0.95,
                matches: 2,
            },
        ];
        let network = build_network(&metas, &scores, 0.2, 5);

        let visible = visible_node_ids_for_view(&network, ComponentSelection::Largest, true);
        assert_eq!(visible, vec![0, 1, 2]);
    }
}
