use std::sync::Arc;

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

pub use spectral_matcher::load_mgf_bytes;
#[cfg(target_arch = "wasm32")]
pub use spectral_matcher::load_mgf_file_for_wasm;
#[cfg(not(target_arch = "wasm32"))]
pub use spectral_matcher::load_mgf_path;
pub use spectral_matcher::model::{
    LoadedSpectra, ParseStats, SpectrumMetadata as SpectrumMeta, SpectrumRecord,
};
#[cfg(not(target_arch = "wasm32"))]
pub use spectral_matcher::{NativeLoadHandle, NativeLoadMessage, start_native_mgf_load};

pub fn spectrum_record_from_parts(
    meta: SpectrumMeta,
    peaks: &[(f64, f64)],
) -> Result<SpectrumRecord, String> {
    let mut spectrum = GenericSpectrum::<f64, f64>::with_capacity(meta.precursor_mz, peaks.len())
        .map_err(|err| format!("failed to allocate spectrum for node {}: {err}", meta.id))?;
    for (mz, intensity) in peaks {
        spectrum
            .add_peak(*mz, *intensity)
            .map_err(|err| format!("failed to add peak for node {}: {err}", meta.id))?;
    }
    Ok(SpectrumRecord {
        meta,
        peaks: Arc::new(peaks.to_vec()),
        spectrum: Arc::new(spectrum),
        payload: (),
    })
}
