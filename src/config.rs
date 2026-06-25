#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
pub const DEFAULT_CONFIG_FILE: &str = "spectral-network-gui.toml";

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct RawAppConfig {
    lotus_metadata_path: Option<String>,
    #[serde(default = "default_auto_load_lotus_metadata")]
    auto_load_lotus_metadata: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct AppConfig {
    pub lotus_metadata_path: Option<PathBuf>,
    pub auto_load_lotus_metadata: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn default_auto_load_lotus_metadata() -> bool {
    true
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_default_config() -> Result<Option<AppConfig>, String> {
    let path = PathBuf::from(DEFAULT_CONFIG_FILE);
    if !path.exists() {
        return Ok(None);
    }
    load_config_path(&path).map(Some)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_config_path(path: &Path) -> Result<AppConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    let raw: RawAppConfig =
        toml::from_str(&text).map_err(|err| format!("cannot parse {}: {err}", path.display()))?;
    let base_dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let lotus_metadata_path = raw.lotus_metadata_path.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            let candidate = PathBuf::from(trimmed);
            Some(if candidate.is_absolute() {
                candidate
            } else {
                base_dir.join(candidate)
            })
        }
    });
    Ok(AppConfig {
        lotus_metadata_path,
        auto_load_lotus_metadata: raw.auto_load_lotus_metadata,
    })
}
