#![cfg(not(target_arch = "wasm32"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

use spectral_matcher::{NetworkBuildParams, NetworkRequest, ParseConfig, SearchRequest, SimilarityMetric};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind temp listener")
        .local_addr()
        .expect("local addr")
        .port()
}

fn start_server() -> String {
    let port = free_port();
    let bind = format!("127.0.0.1:{port}");
    let url = format!("http://{bind}");
    thread::spawn({
        let bind = bind.clone();
        move || {
            spectral_matcher::serve(&bind).expect("serve");
        }
    });
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        if request_json::<serde_json::Value>(&url, "GET", "/v1/health", None).is_ok() {
            return url;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("server failed to start");
}

fn request_json<T: serde::de::DeserializeOwned>(
    base_url: &str,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<T, String> {
    let without_scheme = base_url.strip_prefix("http://").ok_or("bad url")?;
    let mut stream = TcpStream::connect(without_scheme).map_err(|err| err.to_string())?;
    let body = body.unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {without_scheme}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|err| err.to_string())?;
    if !body.is_empty() {
        stream.write_all(body).map_err(|err| err.to_string())?;
    }
    let mut response = Vec::new();
    stream.read_to_end(&mut response).map_err(|err| err.to_string())?;
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("bad response")?;
    let header = String::from_utf8(response[..split].to_vec()).map_err(|err| err.to_string())?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or("missing status")?
        .parse::<u16>()
        .map_err(|err| err.to_string())?;
    if !(200..300).contains(&status) {
        return Err(String::from_utf8_lossy(&response[split + 4..]).to_string());
    }
    serde_json::from_slice(&response[split + 4..]).map_err(|err| err.to_string())
}

#[test]
fn health_and_sync_network_endpoints_work() {
    let url = start_server();
    let payload = serde_json::to_vec(&NetworkRequest {
        source_label: "query.mgf".to_string(),
        mgf_text: Some(concat!(
            "BEGIN IONS\nNAME=a\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n",
            "BEGIN IONS\nNAME=b\nPEPMASS=100.1\n10 100\n20 80\n30 50\nEND IONS\n"
        )
        .to_string()),
        mgf_path: None,
        parse: ParseConfig {
            min_peaks: 1,
            max_peaks: 1000,
        },
        build: NetworkBuildParams {
            compute: spectral_matcher::ComputeParams {
                metric: SimilarityMetric::CosineGreedy,
                tolerance: 0.2,
                mz_power: 0.0,
                intensity_power: 1.0,
                top_n_peaks: None,
            },
            threshold: 0.0,
            top_k: 5,
        },
    })
    .expect("serialize");

    let artifact: spectral_matcher::NetworkArtifact =
        request_json(&url, "POST", "/v1/network", Some(&payload)).expect("network response");
    assert_eq!(artifact.network.nodes.len(), 2);
}

#[test]
fn async_search_job_endpoints_work() {
    let url = start_server();
    let payload = serde_json::to_vec(&SearchRequest {
        query_source_label: "query.mgf".to_string(),
        query_mgf_text: Some(
            "BEGIN IONS\nNAME=q\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n"
                .to_string(),
        ),
        query_mgf_path: None,
        library_source_label: "library.mgf".to_string(),
        library_mgf_text: Some(
            "BEGIN IONS\nNAME=l\nPEPMASS=100.0\n10 100\n20 80\n30 50\nEND IONS\n"
                .to_string(),
        ),
        library_mgf_path: None,
        parse: ParseConfig {
            min_peaks: 1,
            max_peaks: 1000,
        },
        search: spectral_matcher::LibrarySearchParams {
            compute: spectral_matcher::ComputeParams {
                metric: SimilarityMetric::CosineGreedy,
                tolerance: 0.2,
                mz_power: 0.0,
                intensity_power: 1.0,
                top_n_peaks: None,
            },
            parent_mass_tolerance: 1.0,
            min_matched_peaks: 1,
            min_similarity_threshold: 0.0,
            top_n: 1,
        },
        taxonomy: None,
        query_key: Some(spectral_matcher::SearchQueryKey::FeatureId),
    })
    .expect("serialize");

    let created: spectral_matcher::JobCreatedResponse =
        request_json(&url, "POST", "/v1/library-search/jobs", Some(&payload))
            .expect("create job");
    let started = Instant::now();
    loop {
        let status: spectral_matcher::JobStatusResponse = request_json(
            &url,
            "GET",
            &format!("/v1/jobs/{}", created.job_id),
            None,
        )
        .expect("job status");
        if status.status == spectral_matcher::JobStatus::Finished {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(5), "job timed out");
        thread::sleep(Duration::from_millis(100));
    }
    let result: spectral_matcher::MatcherJobResult = request_json(
        &url,
        "GET",
        &format!("/v1/jobs/{}/result", created.job_id),
        None,
    )
    .expect("job result");
    match result {
        spectral_matcher::MatcherJobResult::LibrarySearch(artifact) => {
            assert_eq!(artifact.result.hits.len(), 1);
        }
        _ => panic!("unexpected result kind"),
    }
}
