use crate::types::Shape;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    GetRows,
    Scale,
    RmsNorm,
    MulMat,
    RoPE,
    FlashAttn,
    SiLU,
    Mul,
    Add,
    Softmax,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op: OpType,
    pub shape: Shape,
    pub src_nodes: Vec<usize>,
}

#[derive(Debug, Default)]
pub struct CGraph {
    pub nodes: Vec<Node>,
}

impl CGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, name: &str, op: OpType, shape: Shape, src_nodes: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            name: name.to_string(),
            op,
            shape,
            src_nodes,
        });
        id
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
