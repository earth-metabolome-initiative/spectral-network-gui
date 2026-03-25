#![cfg(not(target_arch = "wasm32"))]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use spectral_matcher::{
    JobCreatedResponse, JobProgress, JobProgressStage, JobStatus, JobStatusResponse,
    MatcherJobResult, NetworkArtifact, NetworkRequest, SearchArtifact, SearchRequest,
};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8787";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);
const JOB_TIMEOUT: Duration = Duration::from_secs(14_400);
const POLL_INTERVAL: Duration = Duration::from_millis(400);
const MAX_LOG_ENTRIES: usize = 200;

pub type SharedMatcherLog = Arc<MatcherLogStore>;

#[derive(Default)]
pub struct MatcherLogStore {
    entries: Mutex<VecDeque<String>>,
}

impl MatcherLogStore {
    pub fn snapshot(&self) -> Vec<String> {
        self.entries
            .lock()
            .map(|entries| entries.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    fn push(&self, message: String) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.push_back(format!("[{}] {message}", timestamp_string()));
            while entries.len() > MAX_LOG_ENTRIES {
                let _ = entries.pop_front();
            }
        }
    }
}

pub struct NativeMatcherHandle<T> {
    status: Arc<Mutex<String>>,
    progress: Arc<Mutex<Option<(f32, String)>>>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Result<T, String>>,
    log: SharedMatcherLog,
    cancel_request: Arc<dyn Fn() + Send + Sync>,
}

impl<T> NativeMatcherHandle<T> {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Ok(mut status) = self.status.lock() {
            *status = "Cancellation requested".to_string();
        }
        self.log.push("GUI -> matcher | cancel requested".to_string());
        (self.cancel_request)();
    }

    pub fn status_text(&self) -> String {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|_| "Waiting for spectral-matcher".to_string())
    }

    pub fn progress(&self) -> Option<(f32, String)> {
        self.progress
            .lock()
            .ok()
            .and_then(|progress| progress.clone())
    }

    pub fn try_recv(&self) -> Option<Result<T, String>> {
        self.rx.try_recv().ok()
    }
}

pub fn default_base_url() -> &'static str {
    DEFAULT_BASE_URL
}

pub fn new_matcher_log() -> SharedMatcherLog {
    Arc::new(MatcherLogStore::default())
}

pub fn start_native_network_request(
    base_url: String,
    request: NetworkRequest,
    log: SharedMatcherLog,
) -> NativeMatcherHandle<NetworkArtifact> {
    let job_id = Arc::new(Mutex::new(None));
    let source_ref = request
        .mgf_path
        .as_deref()
        .map(|path| format!("source_path={path}"))
        .unwrap_or_else(|| format!("source={}", request.source_label));
    let request_summary = format!(
        "GUI -> matcher | POST /v1/network/jobs | {source_ref} | spectra_build threshold={:.3} top_k={}",
        request.build.threshold,
        request.build.top_k
    );
    let job_id_for_request = Arc::clone(&job_id);
    start_native_job(base_url, log, job_id, move |client| {
        client.log.push(request_summary);
        let created: JobCreatedResponse = client.post_json("/v1/network/jobs", &request)?;
        if let Ok(mut slot) = job_id_for_request.lock() {
            *slot = Some(created.job_id);
        }
        client
            .log
            .push(format!("matcher -> GUI | 202 /v1/network/jobs | job_id={}", created.job_id));
        let result = client.wait_for_job(created.job_id, "network")?;
        match result {
            MatcherJobResult::Network(artifact) => Ok(artifact),
            MatcherJobResult::LibrarySearch(_) => {
                Err("spectral-matcher returned the wrong job result kind".to_string())
            }
        }
    })
}

pub fn start_native_search_request(
    base_url: String,
    request: SearchRequest,
    log: SharedMatcherLog,
) -> NativeMatcherHandle<SearchArtifact> {
    let job_id = Arc::new(Mutex::new(None));
    let query_ref = request
        .query_mgf_path
        .as_deref()
        .map(|path| format!("query_path={path}"))
        .unwrap_or_else(|| format!("query={}", request.query_source_label));
    let library_ref = request
        .library_mgf_path
        .as_deref()
        .map(|path| format!("library_path={path}"))
        .unwrap_or_else(|| format!("library={}", request.library_source_label));
    let request_summary = format!(
        "GUI -> matcher | POST /v1/library-search/jobs | {query_ref} | {library_ref} | top_n={}",
        request.search.top_n
    );
    let job_id_for_request = Arc::clone(&job_id);
    start_native_job(base_url, log, job_id, move |client| {
        client.log.push(request_summary);
        let created: JobCreatedResponse = client.post_json("/v1/library-search/jobs", &request)?;
        if let Ok(mut slot) = job_id_for_request.lock() {
            *slot = Some(created.job_id);
        }
        client.log.push(format!(
            "matcher -> GUI | 202 /v1/library-search/jobs | job_id={}",
            created.job_id
        ));
        let result = client.wait_for_job(created.job_id, "library search")?;
        match result {
            MatcherJobResult::LibrarySearch(artifact) => Ok(artifact),
            MatcherJobResult::Network(_) => {
                Err("spectral-matcher returned the wrong job result kind".to_string())
            }
        }
    })
}

fn start_native_job<T, F>(
    base_url: String,
    log: SharedMatcherLog,
    job_id: Arc<Mutex<Option<u64>>>,
    run_job: F,
) -> NativeMatcherHandle<T>
where
    T: Send + 'static,
    F: FnOnce(&mut LocalMatcherClient) -> Result<T, String> + Send + 'static,
{
    let status = Arc::new(Mutex::new("Waiting for spectral-matcher".to_string()));
    let progress = Arc::new(Mutex::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let status_for_thread = Arc::clone(&status);
    let progress_for_thread = Arc::clone(&progress);
    let cancel_for_thread = Arc::clone(&cancel);
    let log_for_thread = Arc::clone(&log);
    let log_for_outcome = Arc::clone(&log);
    let base_url_for_cancel = base_url.clone();
    let job_id_for_cancel = Arc::clone(&job_id);
    let log_for_cancel = Arc::clone(&log);
    let cancel_request: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        let Some(job_id) = job_id_for_cancel.lock().ok().and_then(|slot| *slot) else {
            return;
        };
        let base_url = base_url_for_cancel.clone();
        let log = Arc::clone(&log_for_cancel);
        std::thread::spawn(move || {
            if let Err(err) = send_cancel_request(&base_url, job_id) {
                log.push(format!(
                    "GUI -> matcher | cancel job_id={job_id} failed | {err}"
                ));
            } else {
                log.push(format!("GUI -> matcher | POST /v1/jobs/{job_id}/cancel"));
            }
        });
    });

    std::thread::spawn(move || {
        let outcome = (|| {
            let mut client = LocalMatcherClient::new(
                base_url,
                status_for_thread,
                progress_for_thread,
                cancel_for_thread,
                log_for_thread,
            )?;
            client.ensure_available()?;
            run_job(&mut client)
        })();
        if let Err(err) = &outcome {
            log_for_outcome.push(format!("matcher -> GUI | error | {err}"));
        } else {
            log_for_outcome.push("matcher -> GUI | request completed".to_string());
        }
        let _ = tx.send(outcome);
    });

    NativeMatcherHandle {
        status,
        progress,
        cancel,
        rx,
        log,
        cancel_request,
    }
}

struct LocalMatcherClient {
    endpoint: Endpoint,
    status: Arc<Mutex<String>>,
    progress: Arc<Mutex<Option<(f32, String)>>>,
    cancel: Arc<AtomicBool>,
    log: SharedMatcherLog,
}

impl LocalMatcherClient {
    fn new(
        base_url: String,
        status: Arc<Mutex<String>>,
        progress: Arc<Mutex<Option<(f32, String)>>>,
        cancel: Arc<AtomicBool>,
        log: SharedMatcherLog,
    ) -> Result<Self, String> {
        let endpoint = Endpoint::parse(&base_url)?;
        log.push(format!("GUI | matcher base URL = http://{}", endpoint.socket_addr));
        Ok(Self {
            endpoint,
            status,
            progress,
            cancel,
            log,
        })
    }

    fn ensure_available(&mut self) -> Result<(), String> {
        self.set_status("Checking local spectral-matcher");
        self.log.push("GUI -> matcher | GET /v1/health".to_string());
        if self.health_check().is_ok() {
            self.set_status("Connected to local spectral-matcher");
            self.log
                .push("matcher -> GUI | 200 /v1/health | local matcher already running".to_string());
            return Ok(());
        }

        self.set_status("Starting local spectral-matcher");
        self.log
            .push(format!("GUI | attempting local matcher spawn on {}", self.endpoint.bind_address()));
        let _child = spawn_local_matcher(&self.endpoint.bind_address())?;
        let started = Instant::now();
        while started.elapsed() < HEALTH_TIMEOUT {
            self.check_cancelled()?;
            if self.health_check().is_ok() {
                self.set_status("Connected to local spectral-matcher");
                self.log.push(format!(
                    "matcher -> GUI | 200 /v1/health | local matcher became healthy after {} ms",
                    started.elapsed().as_millis()
                ));
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(200));
        }

        Err("spectral-matcher did not become healthy after startup".to_string())
    }

    fn wait_for_job(&mut self, job_id: u64, label: &str) -> Result<MatcherJobResult, String> {
        let started = Instant::now();
        let mut last_status = None;
        loop {
            self.check_cancelled()?;
            if started.elapsed() > JOB_TIMEOUT {
                return Err(format!("timed out while waiting for {label} job {job_id}"));
            }

            let status: JobStatusResponse =
                self.get_json(&format!("/v1/jobs/{job_id}"))?;
            self.set_progress(status.progress.as_ref(), label);
            if last_status != Some(status.status) {
                self.log.push(format!(
                    "matcher -> GUI | GET /v1/jobs/{job_id} | status={:?}",
                    status.status
                ));
                last_status = Some(status.status);
            }
            match status.status {
                JobStatus::Queued => self.set_status(&format!("Queued {label} job {job_id}")),
                JobStatus::Running => self.set_status(&format!("Running {label} job {job_id}")),
                JobStatus::Finished => {
                    self.set_status(&format!("Loading {label} result"));
                    self.set_progress(Some(&JobProgress {
                        stage: JobProgressStage::Finalizing,
                        completed: 1,
                        total: 1,
                    }), label);
                    self.log.push(format!(
                        "GUI -> matcher | GET /v1/jobs/{job_id}/result"
                    ));
                    return self.get_json(&format!("/v1/jobs/{job_id}/result"));
                }
                JobStatus::Failed => {
                    return Err(status
                        .error
                        .unwrap_or_else(|| format!("{label} job {job_id} failed")));
                }
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn health_check(&self) -> Result<(), String> {
        let _: serde_json::Value = self.get_json("/v1/health")?;
        Ok(())
    }

    fn post_json<TReq: serde::Serialize, TResp: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        payload: &TReq,
    ) -> Result<TResp, String> {
        let body = serde_json::to_vec(payload)
            .map_err(|err| format!("failed to serialize request JSON: {err}"))?;
        let response = self.request("POST", path, Some(&body))?;
        self.log.push(format!(
            "matcher -> GUI | {} {} | {} bytes",
            response.status, path, response.body.len()
        ));
        decode_json_response(response)
    }

    fn get_json<TResp: serde::de::DeserializeOwned>(&self, path: &str) -> Result<TResp, String> {
        let response = self.request("GET", path, None)?;
        self.log.push(format!(
            "matcher -> GUI | {} {} | {} bytes",
            response.status, path, response.body.len()
        ));
        decode_json_response(response)
    }

    fn request(&self, method: &str, path: &str, body: Option<&[u8]>) -> Result<HttpResponse, String> {
        let mut stream = TcpStream::connect(self.endpoint.socket_addr.as_str())
            .map_err(|err| format!("failed to connect to spectral-matcher: {err}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|err| format!("failed to set read timeout: {err}"))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|err| format!("failed to set write timeout: {err}"))?;

        let path = format!("{}{}", self.endpoint.base_path, path);
        let body = body.unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.endpoint.host_port,
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|err| format!("failed to write request headers: {err}"))?;
        if !body.is_empty() {
            stream
                .write_all(body)
                .map_err(|err| format!("failed to write request body: {err}"))?;
        }
        stream
            .flush()
            .map_err(|err| format!("failed to flush request: {err}"))?;

        let mut bytes = Vec::new();
        stream
            .read_to_end(&mut bytes)
            .map_err(|err| format!("failed to read response: {err}"))?;
        HttpResponse::parse(&bytes)
    }

    fn set_status(&self, value: &str) {
        if let Ok(mut status) = self.status.lock() {
            *status = value.to_string();
        }
    }

    fn set_progress(&self, progress: Option<&JobProgress>, label: &str) {
        let mapped = progress.map(|progress| map_progress(progress, label));
        if let Ok(mut slot) = self.progress.lock() {
            *slot = mapped;
        }
    }

    fn check_cancelled(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) {
            Err("matcher request cancelled".to_string())
        } else {
            Ok(())
        }
    }
}

fn map_progress(progress: &JobProgress, label: &str) -> (f32, String) {
    let done = progress.completed.min(progress.total.max(1));
    let total = progress.total.max(1);
    let ratio = done as f32 / total as f32;
    let pct = ratio * 100.0;
    match progress.stage {
        JobProgressStage::Queued => (0.0, format!("Queued {label} job")),
        JobProgressStage::LoadingSpectra => (
            0.05 * ratio,
            format!("Loading spectra ({pct:.1}%)"),
        ),
        JobProgressStage::LoadingQuery => (
            0.05 * ratio,
            format!("Loading query spectra ({pct:.1}%)"),
        ),
        JobProgressStage::LoadingLibrary => {
            (
                0.05 + 0.15 * ratio,
                format!("Loading library spectra ({pct:.1}%)"),
            )
        }
        JobProgressStage::Scoring => (
            (0.20 + 0.75 * ratio).clamp(0.0, 0.95),
            format!("Scoring similarities: {done}/{total} ({pct:.1}%)"),
        ),
        JobProgressStage::BuildingNetwork => (0.97, "Building spectral network".to_string()),
        JobProgressStage::Finalizing => (
            (0.95 + 0.04 * ratio).clamp(0.95, 0.99),
            format!("Finalizing {label} ({pct:.1}%)"),
        ),
    }
}

fn send_cancel_request(base_url: &str, job_id: u64) -> Result<(), String> {
    let endpoint = Endpoint::parse(base_url)?;
    let mut stream = TcpStream::connect(endpoint.socket_addr.as_str())
        .map_err(|err| format!("failed to connect to spectral-matcher: {err}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|err| format!("failed to set read timeout: {err}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|err| format!("failed to set write timeout: {err}"))?;
    let path = format!("{}/v1/jobs/{job_id}/cancel", endpoint.base_path);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        endpoint.host_port,
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|err| format!("failed to write cancel request: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("failed to flush cancel request: {err}"))?;
    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read cancel response: {err}"))?;
    let response = HttpResponse::parse(&bytes)?;
    if (200..300).contains(&response.status) || response.status == 409 {
        Ok(())
    } else {
        Err(format!("cancel request returned HTTP {}", response.status))
    }
}

fn spawn_local_matcher(bind: &str) -> Result<Child, String> {
    if let Ok(explicit) = std::env::var("SPECTRAL_MATCHER_BIN") {
        return spawn_command(Command::new(explicit), bind);
    }

    if let Ok(current_exe) = std::env::current_exe() {
        let sibling = current_exe
            .parent()
            .map(|parent| parent.join("spectral-matcher"))
            .filter(|path| path.exists());
        if let Some(path) = sibling {
            return spawn_command(Command::new(path), bind);
        }
    }

    if let Ok(child) = spawn_command(Command::new("spectral-matcher"), bind) {
        return Ok(child);
    }

    let repo_root = std::env::current_dir()
        .ok()
        .filter(|dir| dir.join("Cargo.toml").exists())
        .map(PathBuf::from);
    if let Some(dir) = repo_root {
        let mut command = Command::new("cargo");
        command.current_dir(dir);
        command.arg("run").arg("-p").arg("spectral-matcher").arg("--");
        return spawn_command(command, bind);
    }

    Err("failed to locate a spectral-matcher executable".to_string())
}

fn spawn_command(mut command: Command, bind: &str) -> Result<Child, String> {
    command.arg("serve").arg("--bind").arg(bind);
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    command
        .spawn()
        .map_err(|err| format!("failed to start spectral-matcher: {err}"))
}

fn timestamp_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => format!("{}.{:03}", duration.as_secs(), duration.subsec_millis()),
        Err(_) => "0.000".to_string(),
    }
}

struct Endpoint {
    socket_addr: String,
    host_port: String,
    base_path: String,
}

impl Endpoint {
    fn parse(base_url: &str) -> Result<Self, String> {
        let without_scheme = base_url
            .strip_prefix("http://")
            .ok_or_else(|| "matcher base URL must start with http://".to_string())?;
        let (host_port, base_path) = match without_scheme.split_once('/') {
            Some((host_port, rest)) => (host_port.to_string(), format!("/{}", rest.trim_matches('/'))),
            None => (without_scheme.to_string(), String::new()),
        };
        if host_port.is_empty() {
            return Err("matcher base URL is missing host".to_string());
        }
        Ok(Self {
            socket_addr: host_port.clone(),
            host_port,
            base_path,
        })
    }

    fn bind_address(&self) -> String {
        self.socket_addr.clone()
    }
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

impl HttpResponse {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let split = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or("malformed HTTP response")?;
        let header_text = String::from_utf8(bytes[..split].to_vec())
            .map_err(|_| "HTTP response headers are not valid UTF-8".to_string())?;
        let status = header_text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .ok_or("missing HTTP response status")?
            .parse::<u16>()
            .map_err(|_| "invalid HTTP response status".to_string())?;
        Ok(Self {
            status,
            body: bytes[split + 4..].to_vec(),
        })
    }
}

fn decode_json_response<T: serde::de::DeserializeOwned>(response: HttpResponse) -> Result<T, String> {
    if (200..300).contains(&response.status) {
        serde_json::from_slice(&response.body)
            .map_err(|err| format!("failed to decode matcher response: {err}"))
    } else {
        let value = serde_json::from_slice::<serde_json::Value>(&response.body).ok();
        let message = value
            .as_ref()
            .and_then(|json| json.get("error"))
            .and_then(|err| err.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                String::from_utf8_lossy(&response.body)
                    .trim()
                    .to_string()
            });
        Err(if message.is_empty() {
            format!("spectral-matcher returned HTTP {}", response.status)
        } else {
            format!("spectral-matcher returned HTTP {}: {message}", response.status)
        })
    }
}
