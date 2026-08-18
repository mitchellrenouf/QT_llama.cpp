use clap::Parser;
use mrml_core::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mrml_qt::launch_qt_gui(&Config::parse()).await
}
