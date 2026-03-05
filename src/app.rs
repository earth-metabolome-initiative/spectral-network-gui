use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use eframe::egui;

#[cfg(target_arch = "wasm32")]
use crate::compute::NativeComputeHandle;
use crate::compute::{
    ComputeMessage, ComputeParams, IncrementalComputeState, IncrementalStep, PairScore,
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

pub struct SpectralApp {
    #[cfg(not(target_arch = "wasm32"))]
    mgf_path: String,
    source_label: Option<String>,
    parse_stats: Option<ParseStats>,
    spectra: Vec<SpectrumRecord>,

    tolerance_input: String,
    mz_power_input: String,
    intensity_power_input: String,

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

    #[cfg(target_arch = "wasm32")]
    upload_promise: Option<poll_promise::Promise<Result<UploadedFile, String>>>,
}

impl SpectralApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            mgf_path: DEFAULT_MGF_PATH.to_string(),
            source_label: None,
            parse_stats: None,
            spectra: Vec::new(),
            tolerance_input: "0.02".to_string(),
            mz_power_input: "0".to_string(),
            intensity_power_input: "1".to_string(),
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
            },
            hovered_node_id: None,
            selected_node_id: None,
            native_compute: None,
            incremental_compute: None,
            status_message: None,
            error_message: None,
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
        self.status_message = Some("Computing CosineGreedy scores...".to_string());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native_compute = Some(start_native_compute(self.spectra.clone(), params));
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.incremental_compute =
                Some(IncrementalComputeState::new(self.spectra.clone(), params));
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
        ui.label("Metric: CosineGreedy");

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

        ui.separator();
        ui.collapsing("Compute", |ui| {
            let can_start = !self.is_computing() && !self.spectra.is_empty();
            if ui
                .add_enabled(can_start, egui::Button::new("Run CosineGreedy"))
                .clicked()
            {
                self.start_compute();
            }

            if self.is_computing() {
                if ui.button("Cancel").clicked() {
                    self.cancel_compute();
                }

                if let Some((done, total)) = self.compute_progress() {
                    let frac = if total == 0 {
                        0.0
                    } else {
                        done as f32 / total as f32
                    };
                    ui.add(egui::ProgressBar::new(frac).text(format!("{done}/{total}")));
                }
            }
        });

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
            if ui.button("Fit to view").clicked() {
                self.layout_running = false;
                self.request_fit_view = true;
            }
            if ui.button("Fit full network").clicked() {
                self.layout_running = false;
                self.component_selection = ComponentSelection::All;
                self.request_fit_view = true;
            }
            if ui.button("Reset view").clicked() {
                self.layout_running = false;
                self.view_state.pan = egui::Vec2::ZERO;
                self.view_state.zoom = 1.0;
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
                ui.label("Load spectra and run CosineGreedy to render a spectral network.");
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

        let network = self.network.as_ref().expect("network presence checked");
        let interaction = draw_network(
            ui,
            network,
            &self.positions,
            &visible_set,
            &mut self.view_state,
            self.selected_node_id,
        );
        self.hovered_node_id = interaction.hovered_node_id;
        self.view_state.pan += interaction.pan_delta;
        if let Some((node_id, delta)) = interaction.dragged_node
            && let Some(pos) = self.positions.get_mut(&node_id)
        {
            pos[0] += delta[0];
            pos[1] += delta[1];
            ui.ctx().request_repaint();
        }
        if let Some(node_id) = interaction.clicked_node_id {
            self.selected_node_id = Some(node_id);
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

    fn draw_selected_node_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Node Properties");
        ui.separator();

        let Some(network) = &self.network else {
            ui.label("No network loaded.");
            return;
        };

        let Some(node_id) = self.selected_node_id else {
            ui.label("Select a node in the graph.");
            return;
        };

        let visible_set =
            visible_node_set_for_view(network, self.component_selection, self.hide_singletons);
        if !visible_set.contains(&node_id) {
            ui.label("Selected node is not visible in current scope.");
            return;
        }

        let Some(node) = network.nodes.iter().find(|n| n.id == node_id) else {
            ui.label("Selected node is not visible in current scope.");
            return;
        };

        ui.monospace(format!("Node index: {}", node.id));
        ui.monospace(format!(
            "Feature ID: {}",
            node.feature_id.as_deref().unwrap_or("n/a")
        ));
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
    }
}

impl eframe::App for SpectralApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_arch = "wasm32")]
        self.poll_upload_dialog();

        self.poll_compute(ctx);

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
