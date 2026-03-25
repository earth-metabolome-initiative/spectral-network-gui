# spectral-network-gui

`egui/eframe` desktop + web GUI for inspecting spectral networks and library-search results produced by the local `spectral-matcher` service.

## Install

Install the Rust toolchain first:

```bash
rustup toolchain install stable
```

Build the native workspace binaries from the repository root:

```bash
cargo build
```

For web builds, also install the wasm target and `trunk`:

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk
```

## Native run

```bash
cargo run -p spectral-matcher -- serve
cargo run -p spectral-network-gui
```

## Native run (release)

```bash
cargo run --release -p spectral-matcher -- serve
cargo run --release -p spectral-network-gui
```

The native GUI defaults to `http://127.0.0.1:8787` for `spectral-matcher`.

## Matcher CLI

```bash
cargo run -p spectral-matcher -- search --config path/to/search.toml
cargo run -p spectral-matcher -- network --config path/to/network.toml
```

On native, the default query input path is:

- `./fixtures/mapp_batch_00231.mgf`

## Web build/check

```bash
cargo check -p spectral-network-gui --target wasm32-unknown-unknown --locked
```

Optional local web serve (requires `trunk`):

```bash
cd spectral-network-gui
trunk serve
```
