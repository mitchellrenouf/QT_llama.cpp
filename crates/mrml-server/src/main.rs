#![no_std]
#![cfg_attr(not(test), no_main)]

use mrml_agent::{Config, MrmlAgent};
use mrml_server::ApiServer;

fn application_main() -> mrml_error::Result<()> {
    let config = Config::parse();
    let agent = MrmlAgent::new(config.clone());
    let certificate = mrml_runtime::environment_variable("MRML_TLS_CERT")
        .ok_or_else(|| mrml_error::message("MRML_TLS_CERT must name the HTTPS certificate PEM"))?;
    let private_key = mrml_runtime::environment_variable("MRML_TLS_KEY")
        .ok_or_else(|| mrml_error::message("MRML_TLS_KEY must name the HTTPS private-key PEM"))?;
    let api_token = mrml_runtime::environment_variable("MRML_API_TOKEN").ok_or_else(|| {
        mrml_error::message("MRML_API_TOKEN must contain a high-entropy bearer token")
    })?;
    let certificate = mrml_runtime::read_file(&certificate)?;
    let private_key = mrml_runtime::read_file(&private_key)?;
    ApiServer::new(agent.get_client_arc(), config.port)
        .with_bearer_token(api_token)?
        .with_tls_pem(&certificate, &private_key)?
        .run()
}

mrml_runtime::mrml_entrypoint!(application_main);
