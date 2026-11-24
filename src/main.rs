#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Spectral Network GUI",
        native_options,
        Box::new(|cc| Ok(Box::new(spectral_network_gui::app::SpectralApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
