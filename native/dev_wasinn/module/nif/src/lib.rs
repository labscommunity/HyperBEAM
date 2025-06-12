use std::sync::{Mutex, OnceLock};
use std::thread;

use rustler::NifResult;

static SERVER_STARTED: OnceLock<Mutex<bool>> = OnceLock::new();

#[rustler::nif(name = "load_http_server")]
fn load_http_server(port: u16, path: String) -> NifResult<String> {
    let handle_cell = SERVER_STARTED.get_or_init(|| Mutex::new(false));

    let mut lock = handle_cell.lock().unwrap();
    if *lock {
        return Ok("Server already running".into());
    }

    thread::spawn(move || {
        let runtime = http_server::create_runtime().expect("Tokio runtime not created!");
        let _ = runtime.block_on(http_server::start_server(3002, "../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm", "llama3.2-1b-instruct-fp32", http_server::ExecutionTarget::GPU));
    });

    *lock = true;
    Ok("Server started".into())
}

rustler::init!("dev_wasinn_nif");
