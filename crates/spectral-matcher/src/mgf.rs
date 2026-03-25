use std::collections::BTreeMap;
use std::io::{BufRead, Cursor};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc, SpectrumMut};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::model::{LoadedSpectra, ParseStats, SpectrumMetadata, SpectrumRecord};

#[derive(Clone, Debug)]
struct ParsedSpectrum {
    raw_name: String,
    feature_id: Option<String>,
    scans: Option<String>,
    filename: Option<String>,
    source_scan_usi: Option<String>,
    featurelist_feature_id: Option<String>,
    headers: BTreeMap<String, String>,
    precursor_mz: f64,
    peaks: Vec<(f64, f64)>,
}

#[cfg(not(target_arch = "wasm32"))]
pub enum NativeLoadMessage {
    Finished(LoadedSpectra),
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
pub struct NativeLoadHandle {
    total_bytes: u64,
    processed_bytes: Arc<AtomicU64>,
    accepted: Arc<AtomicUsize>,
    ions_blocks: Arc<AtomicUsize>,
    rx: mpsc::Receiver<NativeLoadMessage>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeLoadHandle {
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn processed_bytes(&self) -> u64 {
        self.processed_bytes.load(Ordering::Relaxed)
    }

    pub fn accepted(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }

    pub fn ions_blocks(&self) -> usize {
        self.ions_blocks.load(Ordering::Relaxed)
    }

    pub fn try_recv(&self) -> Option<NativeLoadMessage> {
        self.rx.try_recv().ok()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_mgf_path(path: &Path, min_peaks: usize, max_peaks: usize) -> Result<LoadedSpectra, String> {
    use std::fs::File;
    use std::io::BufReader;

    let file =
        File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let (parsed, stats) = parse_mgf_reader_local(BufReader::new(file), min_peaks, max_peaks)?;
    let spectra = to_records(parsed)?;
    Ok(LoadedSpectra {
        source_label: path.display().to_string(),
        spectra,
        stats,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_native_mgf_load(
    path: &Path,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<NativeLoadHandle, String> {
    use std::fs::File;
    use std::io::BufReader;

    let source_label = path.display().to_string();
    let total_bytes = std::fs::metadata(path)
        .map_err(|err| format!("cannot stat {}: {err}", path.display()))?
        .len();
    let file = File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let processed_bytes = Arc::new(AtomicU64::new(0));
    let accepted = Arc::new(AtomicUsize::new(0));
    let ions_blocks = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();

    let progress_bytes = Arc::clone(&processed_bytes);
    let progress_accepted = Arc::clone(&accepted);
    let progress_ions = Arc::clone(&ions_blocks);

    std::thread::spawn(move || {
        let reader = BufReader::new(file);
        let (parsed, stats) = match parse_mgf_reader_local_with_progress(
            reader,
            min_peaks,
            max_peaks,
            |processed, stats| {
                progress_bytes.store(processed.min(total_bytes), Ordering::Relaxed);
                progress_accepted.store(stats.accepted, Ordering::Relaxed);
                progress_ions.store(stats.ions_blocks, Ordering::Relaxed);
            },
        ) {
            Ok(result) => result,
            Err(err) => {
                let _ = tx.send(NativeLoadMessage::Failed(err));
                return;
            }
        };

        let spectra = match to_records(parsed) {
            Ok(spectra) => spectra,
            Err(err) => {
                let _ = tx.send(NativeLoadMessage::Failed(err));
                return;
            }
        };

        progress_bytes.store(total_bytes, Ordering::Relaxed);
        progress_accepted.store(stats.accepted, Ordering::Relaxed);
        progress_ions.store(stats.ions_blocks, Ordering::Relaxed);

        let _ = tx.send(NativeLoadMessage::Finished(LoadedSpectra {
            source_label,
            spectra,
            stats,
        }));
    });

    Ok(NativeLoadHandle {
        total_bytes,
        processed_bytes,
        accepted,
        ions_blocks,
        rx,
    })
}

pub fn load_mgf_bytes(
    source_label: &str,
    bytes: &[u8],
    min_peaks: usize,
    max_peaks: usize,
) -> Result<LoadedSpectra, String> {
    let (parsed, stats) = parse_mgf_reader_local(Cursor::new(bytes), min_peaks, max_peaks)?;
    let spectra = to_records(parsed)?;
    Ok(LoadedSpectra {
        source_label: source_label.to_string(),
        spectra,
        stats,
    })
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

fn parse_mgf_reader_local<R: BufRead>(
    reader: R,
    min_peaks: usize,
    max_peaks: usize,
) -> Result<(Vec<ParsedSpectrum>, ParseStats), String> {
    parse_mgf_reader_local_with_progress(reader, min_peaks, max_peaks, |_, _| {})
}

fn parse_mgf_reader_local_with_progress<R: BufRead, F: FnMut(u64, ParseStats)>(
    mut reader: R,
    min_peaks: usize,
    max_peaks: usize,
    mut on_progress: F,
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
    let mut headers = BTreeMap::new();
    let mut precursor_mz: Option<f64> = None;
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    let mut in_ions = false;
    let mut line = String::new();
    let mut processed_bytes = 0_u64;

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|err| format!("failed to read line: {err}"))?;
        if bytes_read == 0 {
            break;
        }
        processed_bytes = processed_bytes.saturating_add(bytes_read as u64);
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
            headers.clear();
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
                    peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
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
                            headers: headers.clone(),
                            precursor_mz: pmz,
                            peaks: std::mem::take(&mut peaks),
                        });
                        stats.accepted += 1;
                    }
                }
            }
            in_ions = false;
            on_progress(processed_bytes, stats);
        } else if in_ions {
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim().to_uppercase();
                let val = val.trim();
                headers.insert(key.clone(), val.to_string());
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
                let mut parts = line.split_whitespace();
                if let (Some(mz), Some(intensity)) = (parts.next(), parts.next())
                    && let (Ok(mz), Ok(intensity)) = (mz.parse::<f64>(), intensity.parse::<f64>())
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

    on_progress(processed_bytes, stats);
    Ok((spectra, stats))
}

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

fn parsed_spectrum_to_record(idx: usize, spec: ParsedSpectrum) -> Result<SpectrumRecord, String> {
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

    Ok(SpectrumRecord {
        meta: SpectrumMetadata {
            id: idx,
            label,
            raw_name: spec.raw_name,
            feature_id: spec.feature_id,
            scans: spec.scans,
            filename: spec.filename,
            source_scan_usi: spec.source_scan_usi,
            featurelist_feature_id: spec.featurelist_feature_id,
            headers: spec.headers,
            precursor_mz: spec.precursor_mz,
            num_peaks: spec.peaks.len(),
        },
        peaks: Arc::new(spec.peaks),
        spectrum: Arc::new(spectrum),
        payload: (),
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn to_records(parsed: Vec<ParsedSpectrum>) -> Result<Vec<SpectrumRecord>, String> {
    parsed
        .into_par_iter()
        .enumerate()
        .map(|(idx, spec)| parsed_spectrum_to_record(idx, spec))
        .collect()
}

#[cfg(target_arch = "wasm32")]
fn to_records(parsed: Vec<ParsedSpectrum>) -> Result<Vec<SpectrumRecord>, String> {
    parsed
        .into_iter()
        .enumerate()
        .map(|(idx, spec)| parsed_spectrum_to_record(idx, spec))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::load_mgf_bytes;

    #[test]
    fn preserves_arbitrary_headers_with_normalized_keys() {
        let input = br#"
BEGIN IONS
name=Example
Adduct=[M+H]+
custom_field=abc
PEPMASS=123.45
10 100
20 50
END IONS
"#;

        let loaded = load_mgf_bytes("test.mgf", input, 2, 100).expect("mgf should parse");
        let meta = &loaded.spectra[0].meta;
        assert_eq!(meta.headers.get("NAME"), Some(&"Example".to_string()));
        assert_eq!(meta.headers.get("ADDUCT"), Some(&"[M+H]+".to_string()));
        assert_eq!(meta.headers.get("CUSTOM_FIELD"), Some(&"abc".to_string()));
        assert_eq!(meta.headers.get("PEPMASS"), Some(&"123.45".to_string()));
    }

    #[test]
    fn keeps_fixed_fields_alongside_headers() {
        let input = br#"
BEGIN IONS
TITLE=Title A
FEATURE_ID=feat-1
FEATURELIST_FEATURE_ID=featlist-2
SCANS=42
FILENAME=file.mgf
SOURCE_SCAN_USI=mzspec:abc
PRECURSOR_MZ=321.0
10 100
20 50
END IONS
"#;

        let loaded = load_mgf_bytes("test.mgf", input, 2, 100).expect("mgf should parse");
        let meta = &loaded.spectra[0].meta;
        assert_eq!(meta.raw_name, "Title A");
        assert_eq!(meta.feature_id.as_deref(), Some("feat-1"));
        assert_eq!(meta.featurelist_feature_id.as_deref(), Some("featlist-2"));
        assert_eq!(meta.scans.as_deref(), Some("42"));
        assert_eq!(meta.filename.as_deref(), Some("file.mgf"));
        assert_eq!(meta.source_scan_usi.as_deref(), Some("mzspec:abc"));
    }
}
