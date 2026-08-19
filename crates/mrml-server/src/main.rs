use mrml_core::{Config, MrmlAgent};
use mrml_server::ApiServer;

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let agent = MrmlAgent::new(config.clone());
    ApiServer::new(agent.get_client_arc(), config.port).run()
}
