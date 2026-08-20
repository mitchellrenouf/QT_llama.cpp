#![no_std]
#![cfg_attr(not(test), no_main)]

use mrml_agent::{Config, MrmlAgent};
use mrml_server::ApiServer;

fn application_main() -> mrml_error::Result<()> {
    let config = Config::parse();
    let agent = MrmlAgent::new(config.clone());
    ApiServer::new(agent.get_client_arc(), config.port).run()
}

mrml_runtime::mrml_entrypoint!(application_main);
