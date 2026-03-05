use std::collections::{HashMap, HashSet};

use egui::{Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::network::SpectralNetwork;

#[derive(Default)]
pub struct GraphViewState {
    pub pan: Vec2,
    pub zoom: f32,
    pub dragging_node_id: Option<usize>,
    pub box_select_start: Option<Pos2>,
    pub box_select_current: Option<Pos2>,
}

pub struct GraphInteraction {
    pub hovered_node_id: Option<usize>,
    pub clicked_node_id: Option<usize>,
    pub clicked_empty_canvas: bool,
    pub fit_full_network_requested: bool,
    pub pan_delta: Vec2,
    pub dragged_node: Option<(usize, [f32; 2])>,
    pub box_selected_node_ids: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct NodeScreenGeom {
    id: usize,
    pos: Pos2,
    radius: f32,
}

pub fn draw_network(
    ui: &mut Ui,
    network: &SpectralNetwork,
    positions: &HashMap<usize, [f32; 2]>,
    visible: &HashSet<usize>,
    view_state: &mut GraphViewState,
    selected_node_id: Option<usize>,
) -> GraphInteraction {
    if view_state.zoom <= 0.0 {
        view_state.zoom = 1.0;
    }

    let available = ui.available_size_before_wrap();
    let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
    let rect = response.rect;

    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > f32::EPSILON {
            let old_zoom = view_state.zoom;
            let zoom_factor = (scroll * 0.002).exp().clamp(0.5, 2.0);
            let new_zoom = (view_state.zoom * zoom_factor).clamp(0.01, 20.0);
            if let Some(cursor) = response
                .hover_pos()
                .or_else(|| ui.input(|i| i.pointer.latest_pos()))
            {
                view_state.pan =
                    pan_for_cursor_zoom(rect.center(), cursor, view_state.pan, old_zoom, new_zoom);
            }
            view_state.zoom = new_zoom;
        }
    }

    let center = rect.center() + view_state.pan;
    let scale = (rect.width().min(rect.height()) * 0.4 * view_state.zoom).max(1e-6);

    let to_screen = |point: [f32; 2]| -> Pos2 {
        Pos2 {
            x: center.x + point[0] * scale,
            y: center.y + point[1] * scale,
        }
    };

    for edge in &network.edges {
        if !visible.contains(&edge.source) || !visible.contains(&edge.target) {
            continue;
        }
        let Some(&p1_raw) = positions.get(&edge.source) else {
            continue;
        };
        let Some(&p2_raw) = positions.get(&edge.target) else {
            continue;
        };
        let p1 = to_screen(p1_raw);
        let p2 = to_screen(p2_raw);
        let alpha = (edge.score.clamp(0.0, 1.0) * 200.0 + 20.0) as u8;
        let width = 0.4 + edge.score as f32 * 2.0;
        painter.line_segment(
            [p1, p2],
            Stroke::new(width, Color32::from_rgba_unmultiplied(70, 90, 110, alpha)),
        );
    }

    let mut node_geometry: Vec<NodeScreenGeom> = Vec::new();

    for node in &network.nodes {
        if !visible.contains(&node.id) {
            continue;
        }
        let Some(&raw_pos) = positions.get(&node.id) else {
            continue;
        };
        let pos = to_screen(raw_pos);
        if !Rect::from_center_size(rect.center(), rect.size() * 1.5).contains(pos) {
            continue;
        }

        let radius = (2.0 + (node.degree as f32).sqrt() * 0.8).clamp(2.5, 10.0);
        painter.circle_filled(pos, radius, component_color(node.component_id));
        node_geometry.push(NodeScreenGeom {
            id: node.id,
            pos,
            radius,
        });
        if selected_node_id == Some(node.id) {
            painter.circle_stroke(
                pos,
                radius + 2.5,
                Stroke::new(1.8, Color32::from_rgb(255, 210, 80)),
            );
        }
    }

    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.0, Color32::from_gray(60)),
        egui::StrokeKind::Outside,
    );

    let clicked_primary = response.clicked_by(PointerButton::Primary);
    let fit_full_network_requested = response.double_clicked_by(PointerButton::Primary);
    let pointer_latest = ui.input(|i| i.pointer.latest_pos());
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    let primary_down = ui.input(|i| i.pointer.primary_down());
    let primary_released = ui.input(|i| i.pointer.primary_released());
    if primary_pressed {
        let picked_node =
            pointer_latest.and_then(|pointer| hit_test_node(pointer, &node_geometry, 4.0));
        view_state.dragging_node_id = picked_node;
        if picked_node.is_some() {
            view_state.box_select_start = None;
            view_state.box_select_current = None;
        } else {
            view_state.box_select_start = pointer_latest;
            view_state.box_select_current = pointer_latest;
        }
    }
    if primary_down
        && view_state.dragging_node_id.is_none()
        && view_state.box_select_start.is_some()
    {
        view_state.box_select_current = pointer_latest;
    }

    let mut box_selected_node_ids: Option<Vec<usize>> = None;
    if primary_released
        && let (Some(start), Some(end)) = (
            view_state.box_select_start.take(),
            view_state.box_select_current.take(),
        )
        && start.distance(end) >= 4.0
    {
        let select_rect = Rect::from_two_pos(start, end);
        let mut selected = node_geometry
            .iter()
            .filter(|node| select_rect.contains(node.pos))
            .map(|node| node.id)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        selected.dedup();
        box_selected_node_ids = Some(selected);
    }
    if !primary_down {
        view_state.dragging_node_id = None;
    }

    let pointer_pos = if clicked_primary {
        response
            .interact_pointer_pos()
            .or(pointer_latest)
            .or_else(|| response.hover_pos())
    } else {
        response.hover_pos()
    };
    let pointer_delta = ui.input(|i| i.pointer.delta());
    let pan_delta = if response.dragged_by(PointerButton::Secondary) {
        pointer_delta
    } else {
        Vec2::ZERO
    };
    let dragged_node = if primary_down {
        if let Some(node_id) = view_state.dragging_node_id {
            let delta = ui.input(|i| i.pointer.delta());
            if delta.length_sq() > 0.0 {
                Some((node_id, [delta.x / scale, delta.y / scale]))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if primary_down
        && view_state.dragging_node_id.is_none()
        && let (Some(start), Some(end)) =
            (view_state.box_select_start, view_state.box_select_current)
        && start.distance(end) >= 2.0
    {
        let select_rect = Rect::from_two_pos(start, end);
        painter.rect_filled(
            select_rect,
            0.0,
            Color32::from_rgba_unmultiplied(90, 140, 220, 28),
        );
        painter.rect_stroke(
            select_rect,
            0.0,
            Stroke::new(1.0, Color32::from_rgb(90, 140, 220)),
            egui::StrokeKind::Outside,
        );
    }

    resolve_interaction(
        pointer_pos,
        clicked_primary,
        fit_full_network_requested,
        pan_delta,
        dragged_node,
        box_selected_node_ids,
        &node_geometry,
    )
}

fn pan_for_cursor_zoom(
    rect_center: Pos2,
    cursor: Pos2,
    pan: Vec2,
    old_zoom: f32,
    new_zoom: f32,
) -> Vec2 {
    if old_zoom <= f32::EPSILON {
        return pan;
    }
    let ratio = new_zoom / old_zoom;
    let cursor_from_center = cursor - rect_center;
    pan * ratio + cursor_from_center * (1.0 - ratio)
}

fn hit_test_node(pointer: Pos2, nodes: &[NodeScreenGeom], pick_padding: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for node in nodes {
        let distance = pointer.distance(node.pos);
        if distance > node.radius + pick_padding {
            continue;
        }
        match best {
            Some((_, best_dist)) if distance >= best_dist => {}
            _ => best = Some((node.id, distance)),
        }
    }
    best.map(|(id, _)| id)
}

fn resolve_interaction(
    pointer_pos: Option<Pos2>,
    clicked_primary: bool,
    fit_full_network_requested: bool,
    pan_delta: Vec2,
    dragged_node: Option<(usize, [f32; 2])>,
    box_selected_node_ids: Option<Vec<usize>>,
    node_geometry: &[NodeScreenGeom],
) -> GraphInteraction {
    let hovered_node_id =
        pointer_pos.and_then(|pointer| hit_test_node(pointer, node_geometry, 4.0));
    let clicked_node_id = if clicked_primary {
        hovered_node_id
    } else {
        None
    };
    let clicked_empty_canvas = clicked_primary && clicked_node_id.is_none();
    GraphInteraction {
        hovered_node_id,
        clicked_node_id,
        clicked_empty_canvas,
        fit_full_network_requested,
        pan_delta,
        dragged_node,
        box_selected_node_ids,
    }
}

fn component_color(component_id: usize) -> Color32 {
    let hue = ((component_id as f32 * 0.137) % 1.0) * 360.0;
    let hsva = egui::ecolor::Hsva {
        h: hue / 360.0,
        s: 0.72,
        v: 0.86,
        a: 1.0,
    };
    Color32::from(hsva)
}

#[cfg(test)]
mod tests {
    use egui::{Pos2, Vec2};

    use super::{NodeScreenGeom, hit_test_node, pan_for_cursor_zoom, resolve_interaction};

    #[test]
    fn hit_test_selects_nearest_node_inside_pick_radius() {
        let nodes = vec![
            NodeScreenGeom {
                id: 1,
                pos: Pos2::new(10.0, 10.0),
                radius: 6.0,
            },
            NodeScreenGeom {
                id: 2,
                pos: Pos2::new(14.0, 10.0),
                radius: 6.0,
            },
        ];

        let picked = hit_test_node(Pos2::new(12.9, 10.0), &nodes, 4.0);
        assert_eq!(picked, Some(2));
    }

    #[test]
    fn hit_test_returns_none_for_empty_area() {
        let nodes = vec![NodeScreenGeom {
            id: 1,
            pos: Pos2::new(10.0, 10.0),
            radius: 4.0,
        }];
        assert_eq!(hit_test_node(Pos2::new(50.0, 50.0), &nodes, 4.0), None);
    }

    #[test]
    fn right_drag_emits_pan_delta_without_selection() {
        let nodes = vec![NodeScreenGeom {
            id: 7,
            pos: Pos2::new(20.0, 20.0),
            radius: 5.0,
        }];

        let interaction =
            resolve_interaction(None, false, false, Vec2::new(3.0, -2.0), None, None, &nodes);

        assert_eq!(interaction.clicked_node_id, None);
        assert!(!interaction.clicked_empty_canvas);
        assert!(!interaction.fit_full_network_requested);
        assert_eq!(interaction.pan_delta, Vec2::new(3.0, -2.0));
        assert!(interaction.box_selected_node_ids.is_none());
    }

    #[test]
    fn primary_background_drag_does_not_pan() {
        let nodes = vec![NodeScreenGeom {
            id: 1,
            pos: Pos2::new(0.0, 0.0),
            radius: 3.0,
        }];

        let interaction =
            resolve_interaction(None, false, false, Vec2::ZERO, None, Some(vec![1]), &nodes);
        assert_eq!(interaction.clicked_node_id, None);
        assert!(!interaction.fit_full_network_requested);
        assert_eq!(interaction.pan_delta, Vec2::ZERO);
        assert_eq!(interaction.box_selected_node_ids, Some(vec![1]));
    }

    #[test]
    fn cursor_zoom_keeps_cursor_anchor() {
        let rect_center = Pos2::new(100.0, 100.0);
        let cursor = Pos2::new(150.0, 100.0);
        let pan = Vec2::new(0.0, 0.0);

        let pan_zoom_in = pan_for_cursor_zoom(rect_center, cursor, pan, 1.0, 2.0);
        let pan_zoom_out = pan_for_cursor_zoom(rect_center, cursor, pan, 2.0, 1.0);

        assert_eq!(pan_zoom_in, Vec2::new(-50.0, 0.0));
        assert_eq!(pan_zoom_out, Vec2::new(25.0, 0.0));
    }
}
