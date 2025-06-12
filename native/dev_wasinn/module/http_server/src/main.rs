use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env()
		.unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

    let runtime = http_server::create_runtime().expect("Tokio runtime not created!");
    let _ = runtime.block_on(http_server::start_server(3002, "../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm", "llama3.2-1b-instruct-fp32", http_server::ExecutionTarget::GPU));
    Ok(())
}

