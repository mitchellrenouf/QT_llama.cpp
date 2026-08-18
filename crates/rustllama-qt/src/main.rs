use clap::Parser;
use rustllama_core::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rustllama_qt::launch_qt_gui(&Config::parse()).await
}
