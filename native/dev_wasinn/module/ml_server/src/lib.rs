pub mod chatbot;
pub mod registry;
pub mod utils;

use tokio::runtime::{Builder, Runtime};

pub fn create_runtime() -> anyhow::Result<Runtime>
{
    Builder::new_multi_thread()
        .enable_all()
        .worker_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .build()
        .map_err(Into::into)
}
