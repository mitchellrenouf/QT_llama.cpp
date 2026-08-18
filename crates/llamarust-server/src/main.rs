use clap::Parser;
use llamarust_core::{Config, GemmaAgent};
use llamarust_server::ApiServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let agent = GemmaAgent::new(config.clone());
    ApiServer::new(agent.get_client_arc(), config.port).run().await
}
