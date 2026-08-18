use clap::Parser;
use llamarust_core::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    llamarust_qt::launch_qt_gui(&Config::parse()).await
}
