#[cfg(not(target_arch = "wasm32"))]
use std::env;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
use spectral_matcher::{
    NetworkBuildParams, ParseConfig, SearchQueryKey, SearchRequest, build_network_artifact,
    run_search_request, save_json_to_path, save_tsv_to_path, serve,
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
struct SearchBatchConfig {
    jobs: Vec<SearchJobConfig>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
struct SearchJobConfig {
    name: Option<String>,
    query_mgf: PathBuf,
    library_mgf: PathBuf,
    output_json: PathBuf,
    output_tsv: Option<PathBuf>,
    #[serde(default)]
    parse: ParseConfig,
    search: SearchConfig,
    #[serde(default)]
    output: SearchOutputConfig,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
struct SearchConfig {
    metric: spectral_matcher::SimilarityMetric,
    tolerance: f64,
    mz_power: f64,
    intensity_power: f64,
    parent_mass_tolerance: f64,
    min_matched_peaks: usize,
    min_similarity_threshold: f64,
    top_n: usize,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize, Default)]
struct SearchOutputConfig {
    query_key: Option<SearchQueryKey>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
struct NetworkBatchConfig {
    jobs: Vec<NetworkJobConfig>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Deserialize)]
struct NetworkJobConfig {
    name: Option<String>,
    input_mgf: PathBuf,
    output_json: PathBuf,
    output_csv_dir: Option<PathBuf>,
    #[serde(default)]
    parse: ParseConfig,
    build: NetworkBuildParams,
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Err(err) = run(env::args()) {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        eprintln!("spectral-matcher CLI is unavailable on wasm");
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run<I>(mut args: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let _exe = args.next();
    let Some(command) = args.next() else {
        return Err("usage: spectral-matcher <serve|search|network> ...".to_string());
    };
    match command.as_str() {
        "serve" => {
            let bind = parse_bind_arg(args)?;
            serve(&bind)
        }
        "search" => {
            let path = parse_config_arg(args)?;
            run_search_config(Path::new(&path))
        }
        "network" => {
            let path = parse_config_arg(args)?;
            run_network_config(Path::new(&path))
        }
        other => Err(format!("unsupported command '{other}'")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_bind_arg<I>(mut args: I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    let Some(flag) = args.next() else {
        return Ok("127.0.0.1:8787".to_string());
    };
    if flag != "--bind" {
        return Err(format!("unsupported argument '{flag}', expected --bind"));
    }
    let Some(bind) = args.next() else {
        return Err("missing bind address after --bind".to_string());
    };
    if args.next().is_some() {
        return Err("unexpected extra arguments".to_string());
    }
    Ok(bind)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_config_arg<I>(mut args: I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    let Some(flag) = args.next() else {
        return Err("missing --config <path>".to_string());
    };
    if flag != "--config" {
        return Err(format!("unsupported argument '{flag}', expected --config"));
    }
    let Some(path) = args.next() else {
        return Err("missing config path after --config".to_string());
    };
    if args.next().is_some() {
        return Err("unexpected extra arguments".to_string());
    }
    Ok(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_search_config(path: &Path) -> Result<(), String> {
    let raw =
        std::fs::read_to_string(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let config: SearchBatchConfig =
        toml::from_str(&raw).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    if config.jobs.is_empty() {
        return Err("config must contain at least one [[jobs]] entry".to_string());
    }
    for (idx, job) in config.jobs.into_iter().enumerate() {
        let label = job
            .name
            .clone()
            .unwrap_or_else(|| format!("job {}", idx + 1));
        run_search_job(&label, job)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_search_job(label: &str, job: SearchJobConfig) -> Result<(), String> {
    let request = SearchRequest {
        query_source_label: job.query_mgf.display().to_string(),
        query_mgf_text: None,
        query_mgf_path: Some(job.query_mgf.display().to_string()),
        library_source_label: job.library_mgf.display().to_string(),
        library_mgf_text: None,
        library_mgf_path: Some(job.library_mgf.display().to_string()),
        parse: job.parse,
        search: spectral_matcher::LibrarySearchParams {
            compute: spectral_matcher::ComputeParams {
                metric: job.search.metric,
                tolerance: job.search.tolerance,
                mz_power: job.search.mz_power,
                intensity_power: job.search.intensity_power,
                top_n_peaks: None,
            },
            parent_mass_tolerance: job.search.parent_mass_tolerance,
            min_matched_peaks: job.search.min_matched_peaks,
            min_similarity_threshold: job.search.min_similarity_threshold,
            top_n: job.search.top_n,
        },
        taxonomy: None,
        query_key: job.output.query_key,
    };
    let artifact = run_search_request(request)
        .map_err(|err| format!("{label}: search failed: {err}"))?;
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("{label}: failed to serialize JSON output: {err}"))?;
    save_json_to_path(&job.output_json, &json)
        .map_err(|err| format!("{label}: failed to write JSON output: {err}"))?;
    if let Some(path) = job.output_tsv {
        save_tsv_to_path(&path, &artifact.tsv)
            .map_err(|err| format!("{label}: failed to write TSV output: {err}"))?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_network_config(path: &Path) -> Result<(), String> {
    let raw =
        std::fs::read_to_string(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let config: NetworkBatchConfig =
        toml::from_str(&raw).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    if config.jobs.is_empty() {
        return Err("config must contain at least one [[jobs]] entry".to_string());
    }
    for (idx, job) in config.jobs.into_iter().enumerate() {
        let label = job
            .name
            .clone()
            .unwrap_or_else(|| format!("job {}", idx + 1));
        run_network_job(&label, job)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_network_job(label: &str, job: NetworkJobConfig) -> Result<(), String> {
    let request = spectral_matcher::NetworkRequest {
        source_label: job.input_mgf.display().to_string(),
        mgf_text: None,
        mgf_path: Some(job.input_mgf.display().to_string()),
        parse: job.parse,
        build: job.build,
    };
    let artifact = build_network_artifact(request)
        .map_err(|err| format!("{label}: network build failed: {err}"))?;
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("{label}: failed to serialize network JSON: {err}"))?;
    save_json_to_path(&job.output_json, &json)
        .map_err(|err| format!("{label}: failed to write network JSON: {err}"))?;
    if let Some(dir) = job.output_csv_dir {
        save_network_csvs(&dir, &artifact)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn save_network_csvs(dir: &Path, artifact: &spectral_matcher::NetworkArtifact) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let nodes_path = dir.join("nodes.csv");
    let edges_path = dir.join("edges.csv");

    let mut nodes_csv =
        String::from("node_id,label,raw_name,precursor_mz,num_peaks,component_id,degree\n");
    for node in &artifact.network.nodes {
        nodes_csv.push_str(&format!(
            "{},{},{},{:.6},{},{},{}\n",
            node.id,
            escape_csv(&node.label),
            escape_csv(&node.raw_name),
            node.precursor_mz,
            node.num_peaks,
            node.component_id,
            node.degree
        ));
    }
    let mut edges_csv = String::from("source,target,score,matches\n");
    for edge in &artifact.network.edges {
        edges_csv.push_str(&format!(
            "{},{},{:.8},{}\n",
            edge.source, edge.target, edge.score, edge.matches
        ));
    }
    std::fs::write(&nodes_path, nodes_csv)
        .map_err(|err| format!("failed to write {}: {err}", nodes_path.display()))?;
    std::fs::write(&edges_path, edges_csv)
        .map_err(|err| format!("failed to write {}: {err}", edges_path.display()))?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        value.to_string()
    }
}
