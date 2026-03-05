use std::collections::{BTreeMap, HashMap, HashSet};

use crate::network::SpectralNetwork;

#[derive(Debug, Default)]
pub struct LayoutResult {
    pub positions: HashMap<usize, [f32; 2]>,
    pub mean_displacement: f32,
}

struct ComponentLayout {
    component_id: usize,
    positions: HashMap<usize, [f32; 2]>,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

impl ComponentLayout {
    fn width(&self) -> f32 {
        (self.max_x - self.min_x).max(0.4)
    }

    fn height(&self) -> f32 {
        (self.max_y - self.min_y).max(0.4)
    }

    fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

pub fn force_directed_layout(
    network: &SpectralNetwork,
    visible_node_ids: &[usize],
    previous_positions: &HashMap<usize, [f32; 2]>,
    iterations: usize,
    node_force: f32,
    edge_force: f32,
) -> LayoutResult {
    if visible_node_ids.is_empty() {
        return LayoutResult::default();
    }

    let visible_set: HashSet<usize> = visible_node_ids.iter().copied().collect();
    let mut members_by_component: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for node in &network.nodes {
        if visible_set.contains(&node.id) {
            members_by_component
                .entry(node.component_id)
                .or_default()
                .push(node.id);
        }
    }
    for members in members_by_component.values_mut() {
        members.sort_unstable();
    }

    let mut component_edges: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    let component_by_id: HashMap<usize, usize> = network
        .nodes
        .iter()
        .map(|n| (n.id, n.component_id))
        .collect();
    for edge in &network.edges {
        if !visible_set.contains(&edge.source) || !visible_set.contains(&edge.target) {
            continue;
        }
        let Some(&left_component) = component_by_id.get(&edge.source) else {
            continue;
        };
        let Some(&right_component) = component_by_id.get(&edge.target) else {
            continue;
        };
        if left_component != right_component {
            continue;
        }
        component_edges
            .entry(left_component)
            .or_default()
            .push((edge.source, edge.target));
    }

    let mut component_layouts = Vec::new();
    for (component_id, members) in members_by_component {
        let edges = component_edges.remove(&component_id).unwrap_or_default();
        component_layouts.push(layout_component(
            component_id,
            &members,
            &edges,
            previous_positions,
            iterations,
            node_force,
            edge_force,
        ));
    }

    let packed_positions = pack_components(component_layouts);
    let mean_displacement =
        mean_displacement(&packed_positions, previous_positions, visible_node_ids);

    LayoutResult {
        positions: packed_positions,
        mean_displacement,
    }
}

fn layout_component(
    component_id: usize,
    members: &[usize],
    edges: &[(usize, usize)],
    previous_positions: &HashMap<usize, [f32; 2]>,
    iterations: usize,
    node_force: f32,
    edge_force: f32,
) -> ComponentLayout {
    let mut index_by_id = HashMap::new();
    for (idx, node_id) in members.iter().copied().enumerate() {
        index_by_id.insert(node_id, idx);
    }

    let bounds = (members.len() as f32).sqrt() * 1.2 + 1.2;
    let mut positions: Vec<[f32; 2]> = members
        .iter()
        .map(|node_id| {
            previous_positions.get(node_id).copied().unwrap_or_else(|| {
                let p = seeded_position(*node_id);
                [p[0] * bounds * 0.7, p[1] * bounds * 0.7]
            })
        })
        .collect();

    if members.len() > 1 {
        let mut local_edges = Vec::new();
        for (left, right) in edges {
            if let (Some(&a), Some(&b)) = (index_by_id.get(left), index_by_id.get(right)) {
                local_edges.push((a, b));
            }
        }
        run_fr_iterations(
            &mut positions,
            &local_edges,
            iterations,
            bounds,
            node_force.max(0.01),
            edge_force.max(0.01),
        );
    }

    center_positions(&mut positions);

    let mut out = HashMap::new();
    for (idx, node_id) in members.iter().copied().enumerate() {
        out.insert(node_id, positions[idx]);
    }
    let (min_x, max_x, min_y, max_y) = bbox(out.values().copied());

    ComponentLayout {
        component_id,
        positions: out,
        min_x,
        max_x,
        min_y,
        max_y,
    }
}

fn run_fr_iterations(
    positions: &mut [[f32; 2]],
    edges: &[(usize, usize)],
    iterations: usize,
    bounds: f32,
    repulsion: f32,
    attraction: f32,
) {
    let n = positions.len();
    let area = (n as f32).max(1.0) * 2.2;
    let k = (area / n as f32).sqrt().max(0.01);
    let iter_count = iterations.max(1);

    for step in 0..iter_count {
        let mut disp = vec![[0.0f32, 0.0f32]; n];

        for a in 0..n {
            for b in (a + 1)..n {
                let dx = positions[a][0] - positions[b][0];
                let dy = positions[a][1] - positions[b][1];
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);
                let force = repulsion * (k * k) / dist;
                let fx = dx / dist * force;
                let fy = dy / dist * force;
                disp[a][0] += fx;
                disp[a][1] += fy;
                disp[b][0] -= fx;
                disp[b][1] -= fy;
            }
        }

        for &(a, b) in edges {
            let dx = positions[a][0] - positions[b][0];
            let dy = positions[a][1] - positions[b][1];
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);
            let force = attraction * (dist * dist) / k;
            let fx = dx / dist * force;
            let fy = dy / dist * force;
            disp[a][0] -= fx;
            disp[a][1] -= fy;
            disp[b][0] += fx;
            disp[b][1] += fy;
        }

        let temp = (0.12 * bounds) * (1.0 - step as f32 / iter_count as f32).max(0.01);
        for i in 0..n {
            let dx = disp[i][0];
            let dy = disp[i][1];
            let mag = (dx * dx + dy * dy).sqrt().max(1e-6);
            let limited = mag.min(temp);
            positions[i][0] = (positions[i][0] + dx / mag * limited) * 0.98;
            positions[i][1] = (positions[i][1] + dy / mag * limited) * 0.98;
            positions[i][0] = positions[i][0].clamp(-bounds, bounds);
            positions[i][1] = positions[i][1].clamp(-bounds, bounds);
        }
    }
}

fn center_positions(positions: &mut [[f32; 2]]) {
    if positions.is_empty() {
        return;
    }
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for pos in positions.iter() {
        cx += pos[0];
        cy += pos[1];
    }
    cx /= positions.len() as f32;
    cy /= positions.len() as f32;

    for pos in positions.iter_mut() {
        pos[0] -= cx;
        pos[1] -= cy;
    }
}

fn bbox(points: impl Iterator<Item = [f32; 2]>) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for [x, y] in points {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (min_x, max_x, min_y, max_y)
    }
}

fn pack_components(mut components: Vec<ComponentLayout>) -> HashMap<usize, [f32; 2]> {
    if components.is_empty() {
        return HashMap::new();
    }

    components.sort_by(|a, b| {
        b.area()
            .total_cmp(&a.area())
            .then(a.component_id.cmp(&b.component_id))
    });

    let total_area: f32 = components.iter().map(ComponentLayout::area).sum();
    let row_limit = total_area.sqrt().max(2.0) * 1.7;
    let gap = 0.6f32;
    let bbox_padding = 0.3f32;

    let mut cursor_x = 0.0f32;
    let mut cursor_y = 0.0f32;
    let mut row_height = 0.0f32;
    let mut packed = HashMap::new();

    for component in components {
        let width = component.width() + bbox_padding * 2.0;
        let height = component.height() + bbox_padding * 2.0;

        if cursor_x > 0.0 && cursor_x + width > row_limit {
            cursor_x = 0.0;
            cursor_y += row_height + gap;
            row_height = 0.0;
        }

        let tx = cursor_x + bbox_padding - component.min_x;
        let ty = cursor_y + bbox_padding - component.min_y;
        for (node_id, [x, y]) in component.positions {
            packed.insert(node_id, [x + tx, y + ty]);
        }

        cursor_x += width + gap;
        row_height = row_height.max(height);
    }

    let (min_x, max_x, min_y, max_y) = bbox(packed.values().copied());
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    for pos in packed.values_mut() {
        pos[0] -= center_x;
        pos[1] -= center_y;
    }

    packed
}

fn mean_displacement(
    positions: &HashMap<usize, [f32; 2]>,
    previous_positions: &HashMap<usize, [f32; 2]>,
    visible_node_ids: &[usize],
) -> f32 {
    if visible_node_ids.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for node_id in visible_node_ids {
        let Some(current) = positions.get(node_id) else {
            continue;
        };
        let prev = previous_positions.get(node_id).unwrap_or(current);
        let dx = current[0] - prev[0];
        let dy = current[1] - prev[1];
        sum += (dx * dx + dy * dy).sqrt();
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn seeded_position(id: usize) -> [f32; 2] {
    let mut x = id as u64 ^ 0x9E3779B97F4A7C15;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;

    let lo = (x & 0xffff) as f32 / 65535.0;
    let hi = ((x >> 16) & 0xffff) as f32 / 65535.0;
    [lo * 2.0 - 1.0, hi * 2.0 - 1.0]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::compute::PairScore;
    use crate::io::SpectrumMeta;
    use crate::network::build_network;

    use super::force_directed_layout;

    fn meta(id: usize) -> SpectrumMeta {
        SpectrumMeta {
            id,
            label: format!("s{id}"),
            raw_name: format!("raw{id}"),
            feature_id: None,
            scans: None,
            filename: None,
            source_scan_usi: None,
            featurelist_feature_id: None,
            precursor_mz: 100.0 + id as f64,
            num_peaks: 10,
        }
    }

    #[test]
    fn component_packing_separates_disconnected_components() {
        let metas = vec![meta(0), meta(1), meta(2), meta(3)];
        let scores = vec![
            PairScore {
                left: 0,
                right: 1,
                score: 0.9,
                matches: 1,
            },
            PairScore {
                left: 2,
                right: 3,
                score: 0.9,
                matches: 1,
            },
        ];
        let network = build_network(&metas, &scores, 0.5, 5);
        let result = force_directed_layout(&network, &[0, 1, 2, 3], &HashMap::new(), 50, 1.0, 1.0);

        let c0 = [
            result.positions[&0][0] + result.positions[&1][0],
            result.positions[&0][1] + result.positions[&1][1],
        ];
        let c1 = [
            result.positions[&2][0] + result.positions[&3][0],
            result.positions[&2][1] + result.positions[&3][1],
        ];
        let dx = c0[0] - c1[0];
        let dy = c0[1] - c1[1];
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist > 0.6);
    }
}
