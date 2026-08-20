use mrml_error::{Result, anyhow};
use mrml_runtime::Text as String;
use mrml_runtime::{Command, PipedChild, Shared, SpinMutex, Text, Vector};
use serde_json::Value;

use crate::{DynTool, Tool};

pub struct McpClient {
    child: SpinMutex<PipedChild>,
    req_id: SpinMutex<u64>,
}

impl McpClient {
    pub async fn spawn(command: &str, args: &[&str]) -> Result<Shared<Self>> {
        let child = Command::new(command)
            .args(args.iter().copied())
            .spawn_piped()?;

        let client = Shared::new(Self {
            child: SpinMutex::new(child),
            req_id: SpinMutex::new(1),
        });

        // Initialize MCP connection
        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mrml",
                "version": "0.4.0"
            }
        });

        let _ = client.call_method("initialize", Some(init_params)).await?;

        Ok(client)
    }

    pub async fn call_method(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = {
            let mut id_guard = self.req_id.lock();
            let current = *id_guard;
            *id_guard += 1;
            current
        };

        let mut request = serde_json::object([
            ("jsonrpc", "2.0".into()),
            ("id", id.into()),
            ("method", method.into()),
        ]);
        if let Some(params) = params {
            request["params"] = params;
        }
        let mut req_str = serde_json::to_string(&request)?;
        req_str.push('\n');

        let line = {
            let mut child = self.child.lock();
            child.write_all(req_str.as_bytes())?;
            child.read_line()?
        };

        let resp: Value = serde_json::from_str(&line)?;
        if let Some(err) = resp.get("error").filter(|value| !value.is_null()) {
            return Err(anyhow!("MCP Error: {}", err));
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| anyhow!("Empty result from MCP server"))
    }

    pub async fn list_tools(client: &Shared<Self>) -> Result<Vector<Shared<dyn DynTool>>> {
        let res = client.call_method("tools/list", None).await?;
        let mut tools: Vector<Shared<dyn DynTool>> = Vector::new();

        if let Some(tools_arr) = res.get("tools").and_then(|t| t.as_array()) {
            for t in tools_arr {
                if let (Some(name), Some(desc)) = (
                    t.get("name").and_then(|n| n.as_str()),
                    t.get("description").and_then(|d| d.as_str()),
                ) {
                    let schema = t.get("inputSchema").cloned().unwrap_or(serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }));

                    let mcp_tool = McpTool {
                        client: client.clone(),
                        name: name.into(),
                        description: desc.into(),
                        parameters: schema,
                    };
                    tools.push(Shared::new(mcp_tool));
                }
            }
        }

        Ok(tools)
    }
}

pub struct McpTool {
    client: Shared<McpClient>,
    name: Text,
    description: Text,
    parameters: Value,
}

impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(&self, _workspace_root: &str, args: Value) -> Result<String> {
        let params = serde_json::object([("name", self.name.as_str().into()), ("arguments", args)]);

        let res = self.client.call_method("tools/call", Some(params)).await?;
        if let Some(content_arr) = res.get("content").and_then(|c| c.as_array()) {
            let mut out = String::new();
            for item in content_arr {
                if let Some(txt) = item.get("text").and_then(|t| t.as_str()) {
                    out.push_str(txt);
                }
            }
            Ok(out)
        } else {
            Ok(serde_json::stringify(&res).as_str().into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pipes_initialize_and_list_mcp_tools() {
        #[cfg(windows)]
        let (program, arguments): (&str, &[&str]) = (
            "powershell.exe",
            &[
                "-NoProfile",
                "-Command",
                "$null=[Console]::In.ReadLine(); [Console]::Out.WriteLine('{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}'); $null=[Console]::In.ReadLine(); [Console]::Out.WriteLine('{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}')",
            ],
        );
        #[cfg(unix)]
        let (program, arguments): (&str, &[&str]) = (
            "sh",
            &[
                "-c",
                "IFS= read -r init; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}'; IFS= read -r list; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}'",
            ],
        );
        crate::block_on(async {
            let client = McpClient::spawn(program, arguments).await.unwrap();
            let tools = McpClient::list_tools(&client).await.unwrap();
            assert!(tools.is_empty());
        });
    }
}
