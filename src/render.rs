use std::collections::{HashMap, HashSet};

use egui::{Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::network::SpectralNetwork;

#[derive(Default)]
pub struct GraphViewState {
    pub pan: Vec2,
    pub zoom: f32,
    pub dragging_node_id: Option<usize>,
}

pub struct GraphInteraction {
    pub hovered_node_id: Option<usize>,
    pub clicked_node_id: Option<usize>,
    pub clicked_empty_canvas: bool,
    pub pan_delta: Vec2,
    pub dragged_node: Option<(usize, [f32; 2])>,
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

    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > f32::EPSILON {
            let zoom_factor = (scroll * 0.002).exp().clamp(0.5, 2.0);
            view_state.zoom = (view_state.zoom * zoom_factor).clamp(0.01, 20.0);
        }
    }

    let rect = response.rect;
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
    let pointer_latest = ui.input(|i| i.pointer.latest_pos());
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    let primary_down = ui.input(|i| i.pointer.primary_down());
    if primary_pressed {
        view_state.dragging_node_id =
            pointer_latest.and_then(|pointer| hit_test_node(pointer, &node_geometry, 4.0));
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
    let pan_delta = if response.dragged_by(PointerButton::Secondary) {
        ui.input(|i| i.pointer.delta())
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

    resolve_interaction(
        pointer_pos,
        clicked_primary,
        pan_delta,
        dragged_node,
        &node_geometry,
    )
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
    pan_delta: Vec2,
    dragged_node: Option<(usize, [f32; 2])>,
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
        pan_delta,
        dragged_node,
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

    use super::{NodeScreenGeom, hit_test_node, resolve_interaction};

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

        let interaction = resolve_interaction(None, false, Vec2::new(3.0, -2.0), None, &nodes);

        assert_eq!(interaction.clicked_node_id, None);
        assert!(!interaction.clicked_empty_canvas);
        assert_eq!(interaction.pan_delta, Vec2::new(3.0, -2.0));
    }
}
