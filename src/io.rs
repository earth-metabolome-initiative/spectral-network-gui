#[cfg(target_arch = "wasm32")]
use std::io::BufRead;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParseStats {
    pub ions_blocks: usize,
    pub accepted: usize,
    pub dropped_missing_name: usize,
    pub dropped_missing_precursor_mz: usize,
    pub dropped_too_few_peaks: usize,
    pub dropped_too_many_peaks: usize,
    pub dropped_nonpositive_intensity_peaks: usize,
    pub dropped_duplicate_mz: usize,
}

#[derive(Clone, Debug)]
struct ParsedSpectrum {
    raw_name: String,
    feature_id: Option<String>,
    scans: Option<String>,
    filename: Option<String>,
    source_scan_usi: Option<String>,
    featurelist_feature_id: Option<String>,
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct SpectrumMeta {
    pub id: usize,
    pub label: String,
    pub raw_name: String,
    pub feature_id: Option<String>,
    pub scans: Option<String>,
    pub filename: Option<String>,
    pub source_scan_usi: Option<String>,
    pub featurelist_feature_id: Option<String>,
    pub precursor_mz: f64,
    pub num_peaks: usize,
}

#[derive(Clone)]
pub struct SpectrumRecord {
    pub meta: SpectrumMeta,
    pub spectrum: Arc<GenericSpectrum<f64, f64>>,
}

#[derive(Clone)]
pub struct LoadedSpectra {
    pub source_label: String,
    pub spectra: Vec<SpectrumRecord>,
    pub stats: ParseStats,
}

pub fn load_mgf_path(
    path: &Path,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<LoadedSpectra, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use spectral_cosine_similarity::mgf_parser::parse_mgf;

        let parsed = parse_mgf(path, min_peaks, max_peaks);
        let spectra = to_records(
            parsed
                .spectra
                .into_iter()
                .map(|s| ParsedSpectrum {
                    raw_name: s.raw_name,
                    feature_id: s.feature_id,
                    scans: s.scans,
                    filename: s.filename,
                    source_scan_usi: s.source_scan_usi,
                    featurelist_feature_id: s.featurelist_feature_id,
                    precursor_mz: s.precursor_mz,
                    peaks: s.peaks,
                })
                .collect(),
        )?;
        Ok(LoadedSpectra {
            source_label: path.display().to_string(),
            spectra,
            stats: ParseStats {
                ions_blocks: parsed.stats.ions_blocks,
                accepted: parsed.stats.accepted,
                dropped_missing_name: parsed.stats.dropped_missing_name,
                dropped_missing_precursor_mz: parsed.stats.dropped_missing_precursor_mz,
                dropped_too_few_peaks: parsed.stats.dropped_too_few_peaks,
                dropped_too_many_peaks: parsed.stats.dropped_too_many_peaks,
                dropped_nonpositive_intensity_peaks: parsed
                    .stats
                    .dropped_nonpositive_intensity_peaks,
                dropped_duplicate_mz: parsed.stats.dropped_duplicate_mz,
            },
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, min_peaks, max_peaks);
        Err("Path-based loading is unavailable on wasm; use upload instead".to_string())
    }
}

pub fn load_mgf_bytes(
    source_label: &str,
    bytes: &[u8],
    min_peaks: usize,
    max_peaks: usize,
) -> Result<LoadedSpectra, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use spectral_cosine_similarity::mgf_parser::parse_mgf_reader;

        let parsed = parse_mgf_reader(Cursor::new(bytes), min_peaks, max_peaks);
        let spectra = to_records(
            parsed
                .spectra
                .into_iter()
                .map(|s| ParsedSpectrum {
                    raw_name: s.raw_name,
                    feature_id: s.feature_id,
                    scans: s.scans,
                    filename: s.filename,
                    source_scan_usi: s.source_scan_usi,
                    featurelist_feature_id: s.featurelist_feature_id,
                    precursor_mz: s.precursor_mz,
                    peaks: s.peaks,
                })
                .collect(),
        )?;
        Ok(LoadedSpectra {
            source_label: source_label.to_string(),
            spectra,
            stats: ParseStats {
                ions_blocks: parsed.stats.ions_blocks,
                accepted: parsed.stats.accepted,
                dropped_missing_name: parsed.stats.dropped_missing_name,
                dropped_missing_precursor_mz: parsed.stats.dropped_missing_precursor_mz,
                dropped_too_few_peaks: parsed.stats.dropped_too_few_peaks,
                dropped_too_many_peaks: parsed.stats.dropped_too_many_peaks,
                dropped_nonpositive_intensity_peaks: parsed
                    .stats
                    .dropped_nonpositive_intensity_peaks,
                dropped_duplicate_mz: parsed.stats.dropped_duplicate_mz,
            },
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let (spectra, stats) = parse_mgf_reader_local(Cursor::new(bytes), min_peaks, max_peaks)?;
        let spectra = to_records(spectra)?;
        Ok(LoadedSpectra {
            source_label: source_label.to_string(),
            spectra,
            stats,
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub fn load_mgf_file_for_wasm(
    source_label: &str,
    file: web_sys::File,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<poll_promise::Promise<Result<LoadedSpectra, String>>, String> {
    use wasm_bindgen_futures::JsFuture;

    let source_label = source_label.to_string();
    Ok(poll_promise::Promise::spawn_local(async move {
        let array_buffer = JsFuture::from(file.array_buffer())
            .await
            .map_err(|err| format!("failed to read file: {err:?}"))?;
        let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();
        load_mgf_bytes(&source_label, &bytes, min_peaks, max_peaks)
    }))
}

#[cfg(target_arch = "wasm32")]
fn parse_mgf_reader_local<R: BufRead>(
    reader: R,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<(Vec<ParsedSpectrum>, ParseStats), String> {
    let mut spectra = Vec::new();
    let mut stats = ParseStats::default();
    let mut name: Option<String> = None;
    let mut title: Option<String> = None;
    let mut compound_name: Option<String> = None;
    let mut featurelist_feature_id: Option<String> = None;
    let mut feature_id: Option<String> = None;
    let mut source_scan_usi: Option<String> = None;
    let mut scans: Option<String> = None;
    let mut filename: Option<String> = None;
    let mut precursor_mz: Option<f64> = None;
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    let mut in_ions = false;

    for line in reader.lines() {
        let line = line.map_err(|err| format!("failed to read line: {err}"))?;
        let line = line.trim();

        if line == "BEGIN IONS" {
            stats.ions_blocks += 1;
            name = None;
            title = None;
            compound_name = None;
            featurelist_feature_id = None;
            feature_id = None;
            source_scan_usi = None;
            scans = None;
            filename = None;
            precursor_mz = None;
            peaks.clear();
            in_ions = true;
        } else if line == "END IONS" {
            let raw_name = preferred_name(
                &name,
                &title,
                &compound_name,
                &feature_id,
                &featurelist_feature_id,
                &source_scan_usi,
                &scans,
            );
            match (raw_name, precursor_mz.take()) {
                (None, _) => {
                    stats.dropped_missing_name += 1;
                }
                (Some(_), None) => {
                    stats.dropped_missing_precursor_mz += 1;
                }
                (Some(_), Some(_)) if peaks.len() < min_peaks => {
                    stats.dropped_too_few_peaks += 1;
                }
                (Some(_), Some(_)) if peaks.len() > max_peaks => {
                    stats.dropped_too_many_peaks += 1;
                }
                (Some(raw_name), Some(pmz)) => {
                    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
                    if peaks.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                        stats.dropped_duplicate_mz += 1;
                    } else {
                        spectra.push(ParsedSpectrum {
                            raw_name,
                            feature_id: feature_id.clone(),
                            scans: scans.clone(),
                            filename: filename.clone(),
                            source_scan_usi: source_scan_usi.clone(),
                            featurelist_feature_id: featurelist_feature_id.clone(),
                            precursor_mz: pmz,
                            peaks: std::mem::take(&mut peaks),
                        });
                        stats.accepted += 1;
                    }
                }
            }
            in_ions = false;
        } else if in_ions {
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim().to_uppercase();
                let val = val.trim();
                match key.as_str() {
                    "NAME" => name = Some(val.to_string()),
                    "TITLE" => title = Some(val.to_string()),
                    "COMPOUND_NAME" => compound_name = Some(val.to_string()),
                    "FEATURELIST_FEATURE_ID" => featurelist_feature_id = Some(val.to_string()),
                    "FEATURE_ID" => feature_id = Some(val.to_string()),
                    "FILENAME" => filename = Some(val.to_string()),
                    "SOURCE_SCAN_USI" => source_scan_usi = Some(val.to_string()),
                    "SCANS" => scans = Some(val.to_string()),
                    "PEPMASS" | "PRECURSOR_MZ" => {
                        if let Some(first) = val.split_whitespace().next()
                            && let Ok(mz) = first.parse::<f64>()
                        {
                            precursor_mz = Some(mz);
                        }
                    }
                    _ => {}
                }
            } else if line.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2
                    && let (Ok(mz), Ok(intensity)) =
                        (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                {
                    if intensity > 0.0 {
                        peaks.push((mz, intensity));
                    } else {
                        stats.dropped_nonpositive_intensity_peaks += 1;
                    }
                }
            }
        }
    }

    Ok((spectra, stats))
}

#[cfg(target_arch = "wasm32")]
fn preferred_name(
    name: &Option<String>,
    title: &Option<String>,
    compound_name: &Option<String>,
    feature_id: &Option<String>,
    featurelist_feature_id: &Option<String>,
    source_scan_usi: &Option<String>,
    scans: &Option<String>,
) -> Option<String> {
    name.clone()
        .or_else(|| title.clone())
        .or_else(|| compound_name.clone())
        .or_else(|| feature_id.clone())
        .or_else(|| featurelist_feature_id.clone())
        .or_else(|| source_scan_usi.clone())
        .or_else(|| scans.clone())
}

fn sanitize_name(raw_name: &str) -> String {
    let mut name = raw_name.to_string();
    name = name.trim_matches('"').trim_matches('\'').to_string();

    let name: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();

    let mut result = String::new();
    let mut prev_underscore = true;
    for c in name.chars() {
        if c == '_' {
            if !prev_underscore {
                result.push('_');
            }
            prev_underscore = true;
        } else {
            result.push(c);
            prev_underscore = false;
        }
    }
    let result = result.trim_end_matches('_').to_string();

    if result.starts_with(|c: char| c.is_ascii_digit()) {
        format!("n{result}")
    } else {
        result
    }
}

fn to_records(parsed: Vec<ParsedSpectrum>) -> Result<Vec<SpectrumRecord>, String> {
    let mut out = Vec::with_capacity(parsed.len());
    for (idx, spec) in parsed.into_iter().enumerate() {
        let mut spectrum =
            GenericSpectrum::<f64, f64>::with_capacity(spec.precursor_mz, spec.peaks.len())
                .map_err(|err| format!("failed to create spectrum capacity: {err:?}"))?;
        for (mz, intensity) in &spec.peaks {
            spectrum
                .add_peak(*mz, *intensity)
                .map_err(|err| format!("failed to add peak: {err:?}"))?;
        }

        let mut label = sanitize_name(&spec.raw_name);
        if label.is_empty() {
            label = format!("spectrum_{idx}");
        }

        out.push(SpectrumRecord {
            meta: SpectrumMeta {
                id: idx,
                label,
                raw_name: spec.raw_name,
                feature_id: spec.feature_id,
                scans: spec.scans,
                filename: spec.filename,
                source_scan_usi: spec.source_scan_usi,
                featurelist_feature_id: spec.featurelist_feature_id,
                precursor_mz: spec.precursor_mz,
                num_peaks: spec.peaks.len(),
            },
            spectrum: Arc::new(spectrum),
        });
    }
    Ok(out)
}
