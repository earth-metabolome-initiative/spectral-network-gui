#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("spectral_matcher_{label}_{nanos}"));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn write_file(path: &PathBuf, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn sample_mgf(name: &str) -> String {
    format!(
        "BEGIN IONS\nNAME={name}\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n"
    )
}

#[test]
fn search_cli_writes_json_and_optional_tsv() {
    let dir = temp_dir("search_json_tsv");
    let query = dir.join("query.mgf");
    let library = dir.join("library.mgf");
    let config = dir.join("config.toml");
    let output_json = dir.join("out/result.json");
    let output_tsv = dir.join("out/result.tsv");
    write_file(&query, &sample_mgf("query"));
    write_file(&library, &sample_mgf("library"));
    write_file(
        &config,
        &format!(
            r#"
[[jobs]]
name = "test"
query_mgf = "{}"
library_mgf = "{}"
output_json = "{}"
output_tsv = "{}"

[jobs.parse]
min_peaks = 1
max_peaks = 1000

[jobs.search]
metric = "CosineGreedy"
tolerance = 0.2
mz_power = 0.0
intensity_power = 1.0
parent_mass_tolerance = 1.0
min_matched_peaks = 1
min_similarity_threshold = 0.0
top_n = 1
"#,
            query.display(),
            library.display(),
            output_json.display(),
            output_tsv.display(),
        ),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_spectral-matcher"))
        .arg("search")
        .arg("--config")
        .arg(&config)
        .output()
        .expect("run cli");
    assert!(output.status.success(), "{output:?}");

    let json = fs::read_to_string(output_json).expect("json output");
    let tsv = fs::read_to_string(output_tsv).expect("tsv output");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert_eq!(parsed["result"]["hits"].as_array().map(Vec::len), Some(1));
    assert!(tsv.starts_with("query_export_key\tquery_node_id"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn search_cli_runs_multiple_jobs() {
    let dir = temp_dir("search_batch");
    let query = dir.join("query.mgf");
    let library = dir.join("library.mgf");
    let config = dir.join("config.toml");
    let output_one = dir.join("out/one.json");
    let output_two = dir.join("out/two.json");
    write_file(&query, &sample_mgf("query"));
    write_file(&library, &sample_mgf("library"));
    write_file(
        &config,
        &format!(
            r#"
[[jobs]]
name = "one"
query_mgf = "{}"
library_mgf = "{}"
output_json = "{}"

[jobs.parse]
min_peaks = 1
max_peaks = 1000

[jobs.search]
metric = "CosineGreedy"
tolerance = 0.2
mz_power = 0.0
intensity_power = 1.0
parent_mass_tolerance = 1.0
min_matched_peaks = 1
min_similarity_threshold = 0.0
top_n = 1

[[jobs]]
name = "two"
query_mgf = "{}"
library_mgf = "{}"
output_json = "{}"

[jobs.parse]
min_peaks = 1
max_peaks = 1000

[jobs.search]
metric = "CosineGreedy"
tolerance = 0.2
mz_power = 0.0
intensity_power = 1.0
parent_mass_tolerance = 1.0
min_matched_peaks = 1
min_similarity_threshold = 0.0
top_n = 1
"#,
            query.display(),
            library.display(),
            output_one.display(),
            query.display(),
            library.display(),
            output_two.display(),
        ),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_spectral-matcher"))
        .arg("search")
        .arg("--config")
        .arg(&config)
        .output()
        .expect("run cli");
    assert!(output.status.success(), "{output:?}");
    assert!(output_one.exists());
    assert!(output_two.exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn network_cli_writes_network_json_and_csvs() {
    let dir = temp_dir("network");
    let input = dir.join("query.mgf");
    let config = dir.join("config.toml");
    let output_json = dir.join("out/network.json");
    let output_csv_dir = dir.join("out/csv");
    write_file(
        &input,
        concat!(
            "BEGIN IONS\nNAME=a\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n",
            "BEGIN IONS\nNAME=b\nPEPMASS=100.1\n10 100\n20 80\n30 50\nEND IONS\n"
        ),
    );
    write_file(
        &config,
        &format!(
            r#"
[[jobs]]
name = "network"
input_mgf = "{}"
output_json = "{}"
output_csv_dir = "{}"

[jobs.parse]
min_peaks = 1
max_peaks = 1000

[jobs.build.compute]
metric = "CosineGreedy"
tolerance = 0.2
mz_power = 0.0
intensity_power = 1.0

[jobs.build]
threshold = 0.0
top_k = 5
"#,
            input.display(),
            output_json.display(),
            output_csv_dir.display(),
        ),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_spectral-matcher"))
        .arg("network")
        .arg("--config")
        .arg(&config)
        .output()
        .expect("run network cli");
    assert!(output.status.success(), "{output:?}");
    assert!(output_json.exists());
    assert!(output_csv_dir.join("nodes.csv").exists());
    assert!(output_csv_dir.join("edges.csv").exists());
    let _ = fs::remove_dir_all(dir);
}
