use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::{DynTool, Tool};

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

pub struct McpClient {
    stdin: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
    _child: Child,
    req_id: Mutex<u64>,
}

impl McpClient {
    pub async fn spawn(command: &str, args: &[&str]) -> Result<Arc<Self>> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to open MCP stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to open MCP stdout"))?;
        let reader = BufReader::new(stdout);

        let client = Arc::new(Self {
            stdin: Mutex::new(stdin),
            reader: Mutex::new(reader),
            _child: child,
            req_id: Mutex::new(1),
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
            let mut id_guard = self.req_id.lock().await;
            let current = *id_guard;
            *id_guard += 1;
            current
        };

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let mut req_str = serde_json::to_string(&req)?;
        req_str.push('\n');

        {
            let mut stdin_guard = self.stdin.lock().await;
            stdin_guard.write_all(req_str.as_bytes()).await?;
            stdin_guard.flush().await?;
        }

        let mut line = String::new();
        {
            let mut reader_guard = self.reader.lock().await;
            reader_guard.read_line(&mut line).await?;
        }

        let resp: JsonRpcResponse = serde_json::from_str(&line)?;
        if let Some(err) = resp.error {
            return Err(anyhow!("MCP Error: {}", err));
        }

        resp.result
            .ok_or_else(|| anyhow!("Empty result from MCP server"))
    }

    pub async fn list_tools(self: &Arc<Self>) -> Result<Vec<Arc<dyn DynTool>>> {
        let res = self.call_method("tools/list", None).await?;
        let mut tools: Vec<Arc<dyn DynTool>> = Vec::new();

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
                        client: self.clone(),
                        name: name.to_string(),
                        description: desc.to_string(),
                        parameters: schema,
                    };
                    tools.push(Arc::new(mcp_tool));
                }
            }
        }

        Ok(tools)
    }
}

pub struct McpTool {
    client: Arc<McpClient>,
    name: String,
    description: String,
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

    async fn execute(&self, _workspace_root: &Path, args: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": self.name,
            "arguments": args
        });

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
            Ok(res.to_string())
        }
    }
}
