use egui::Color32;

/// Color scales for category wedges and their corresponding legend entries.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PiePalette {
    Sequential,
    Diverging,
    #[default]
    Contrasting,
}

impl PiePalette {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sequential => "Sequential",
            Self::Diverging => "Diverging",
            Self::Contrasting => "Contrasting",
        }
    }

    pub fn color(self, category_index: usize, category_count: usize) -> Color32 {
        match self {
            Self::Sequential => {
                // Every channel rises, so even a two-category scale has ordered brightness.
                let position = category_position(category_index, category_count);
                interpolate([19, 57, 111], [93, 182, 216], position)
            }
            Self::Diverging => {
                // A colored, rather than near-white, midpoint remains visible in thin wedges.
                let position = category_position(category_index, category_count);
                if position <= 0.5 {
                    interpolate([35, 92, 157], [190, 163, 111], position * 2.0)
                } else {
                    interpolate([190, 163, 111], [175, 57, 52], (position - 0.5) * 2.0)
                }
            }
            Self::Contrasting => contrasting_color(category_index),
        }
    }
}

fn category_position(index: usize, count: usize) -> f32 {
    if count <= 1 {
        0.5
    } else {
        index.min(count - 1) as f32 / (count - 1) as f32
    }
}

fn interpolate(start: [u8; 3], end: [u8; 3], position: f32) -> Color32 {
    let channel = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * position).round() as u8;
    Color32::from_rgb(
        channel(start[0], end[0]),
        channel(start[1], end[1]),
        channel(start[2], end[2]),
    )
}

fn contrasting_color(index: usize) -> Color32 {
    // High-contrast, moderately dark categorical colors; avoid pastel or white wedges.
    // The first colors draw on color-vision-deficiency-friendly blue/orange/teal families.
    const COLORS: [[u8; 3]; 12] = [
        [0, 103, 166],  // blue
        [207, 83, 12],  // vermilion
        [0, 137, 102],  // teal
        [151, 76, 153], // purple
        [176, 131, 0],  // amber
        [74, 157, 197], // sky blue
        [188, 69, 108], // raspberry
        [65, 111, 51],  // green
        [138, 87, 52],  // brown
        [67, 91, 146],  // slate blue
        [206, 111, 85], // coral
        [104, 75, 134], // violet
    ];
    if let Some(&[r, g, b]) = COLORS.get(index) {
        return Color32::from_rgb(r, g, b);
    }

    // More categories get new hues rather than wrapping immediately to an identical
    // swatch. Integer golden-angle stepping keeps adjacent categories far apart even
    // for indices too large to represent exactly in f32.
    let hue = (index as u32).wrapping_mul(2_654_435_761) as f32 / 4_294_967_296.0;
    let chroma = 0.55_f32;
    let sector = hue * 6.0;
    let secondary = chroma * (1.0 - ((sector % 2.0) - 1.0).abs());
    let (r, g, b) = match sector as u32 {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };
    let channel = |value: f32| ((value + 0.135) * 255.0).round() as u8;
    Color32::from_rgb(channel(r), channel(g), channel(b))
}

#[cfg(test)]
mod tests {
    use super::PiePalette;

    #[test]
    fn neighboring_categories_remain_distinguishable() {
        for palette in [
            PiePalette::Sequential,
            PiePalette::Diverging,
            PiePalette::Contrasting,
        ] {
            for count in 2..=12 {
                for index in 1..count {
                    assert_ne!(
                        palette.color(index - 1, count),
                        palette.color(index, count),
                        "{palette:?}, {index}/{count}"
                    );
                }
            }
            if palette == PiePalette::Contrasting {
                for index in 1..32 {
                    assert_ne!(palette.color(index - 1, 32), palette.color(index, 32));
                }
            }
        }
    }

    #[test]
    fn sequential_brightness_increases_and_diverging_endpoints_differ() {
        let mut previous = 0_u32;
        for index in 0..7 {
            let color = PiePalette::Sequential.color(index, 7);
            let brightness =
                2126 * color.r() as u32 + 7152 * color.g() as u32 + 722 * color.b() as u32;
            assert!(brightness > previous);
            previous = brightness;
        }
        assert_ne!(
            PiePalette::Diverging.color(0, 3),
            PiePalette::Diverging.color(2, 3)
        );
    }
}
