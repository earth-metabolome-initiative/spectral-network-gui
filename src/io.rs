use std::sync::Arc;

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

pub use spectral_matcher::model::{ParseStats, SpectrumMetadata as SpectrumMeta, SpectrumRecord};

const MIN_SPECTRUM_ALLOCATION_PRECURSOR_MZ: f64 = 5.486e-4;

pub fn spectrum_record_from_parts(
    meta: SpectrumMeta,
    peaks: &[(f64, f64)],
) -> Result<SpectrumRecord, String> {
    let allocation_precursor_mz = if meta.precursor_mz.is_finite() {
        meta.precursor_mz.max(MIN_SPECTRUM_ALLOCATION_PRECURSOR_MZ)
    } else {
        meta.precursor_mz
    };
    let mut spectrum = GenericSpectrum::with_capacity(allocation_precursor_mz, peaks.len())
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

pub fn spectrum_record_from_metadata(meta: SpectrumMeta) -> Result<SpectrumRecord, String> {
    spectrum_record_from_parts(meta, &[])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{SpectrumMeta, spectrum_record_from_parts};

    fn metadata_with_precursor(precursor_mz: f64) -> SpectrumMeta {
        SpectrumMeta {
            id: 0,
            spectrum_id: "0".to_string(),
            label: "zero_precursor".to_string(),
            raw_name: "zero_precursor".to_string(),
            feature_id: Some("0".to_string()),
            scans: None,
            filename: None,
            source_scan_usi: None,
            featurelist_feature_id: None,
            headers: BTreeMap::new(),
            precursor_mz,
            num_peaks: 0,
        }
    }

    #[test]
    fn zero_precursor_mz_loads_without_changing_metadata() {
        let record = spectrum_record_from_parts(metadata_with_precursor(0.0), &[])
            .expect("zero precursor metadata should load");

        assert_eq!(record.meta.precursor_mz, 0.0);
    }
}
