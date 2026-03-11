# spectral-network-gui

`egui/eframe` desktop + web GUI for spectral networking with `CosineGreedy`.

## Native run

```bash
cargo run -p spectral-network-gui
```

## Native run (release)

```bash
cargo run --release -p spectral-network-gui
```

On native, the default input path is:

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
