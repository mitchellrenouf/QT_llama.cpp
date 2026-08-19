use crate::types::Shape;
use mrml_runtime::{Text, Vector};

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
    pub name: Text,
    pub op: OpType,
    pub shape: Shape,
    pub src_nodes: Vector<usize>,
}

#[derive(Debug, Default)]
pub struct CGraph {
    pub nodes: Vector<Node>,
}

impl CGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vector::new(),
        }
    }

    pub fn add_node(
        &mut self,
        name: &str,
        op: OpType,
        shape: Shape,
        src_nodes: Vector<usize>,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            name: Text::from(name),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_owns_nodes_and_edges_without_rust_alloc() {
        let mut graph = CGraph::new();
        let source = graph.add_node("source", OpType::GetRows, Shape::new_1d(8), Vector::new());
        let output = graph.add_node("output", OpType::Scale, Shape::new_1d(8), [source].into());

        assert_eq!(output, 1);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.nodes[1].name, "output");
        assert_eq!(&graph.nodes[1].src_nodes[..], &[source]);
    }
}
