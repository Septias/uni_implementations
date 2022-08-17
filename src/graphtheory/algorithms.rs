#![allow(unused)]
use itertools::Itertools;
use nalgebra::{SMatrix, SVector};
use std::{collections::HashSet, hash::Hash};

trait Graph {
    fn neighbors(&self, vertex: usize) -> Vec<usize>;
    fn incident_edges(&self, vertex: usize) -> Vec<usize>;
    fn vertexs(&self) -> HashSet<usize>;
    fn get_vertex_pointed_at(&self, vertex: usize, edge: usize) -> Option<usize>;
    fn get_edge(&self, vertex: usize, vertex: usize) -> Option<usize>;
}

trait Saturated {
    fn saturated(&self) -> HashSet<usize>;
    fn saturate(&self, vertex: usize);
    fn unsaturate(&self, vertex: usize);
}

trait Matching {
    fn matched_edges(&self) -> HashSet<usize>;
    fn add_match(&self, edge: usize);
    fn remove_matched(&self, edge: usize);
    fn generate_path(&self) -> Vec<usize>;
    fn get_matched_neigbor(&self, vertex: usize) -> usize;
}

trait Network {
    fn get_sink(&self) -> usize;
    fn get_source(&self) -> usize;
    fn get_capacity(&self, edge: usize);
}

trait Bigraph {
    fn right_vertexes(&self) -> HashSet<usize>;
    fn left_vertexes(&self) -> HashSet<usize>;
}

struct UndirectedGraph<const NODE_COUNT: usize> {
    graph: SMatrix<bool, NODE_COUNT, NODE_COUNT>,
}

fn augementing_paths_alg<T>(graph: T)
where
    T: Graph + Saturated + Matching + Bigraph,
{
    let S = graph
        .left_vertexes()
        .intersection(&graph.saturated())
        .collect_vec();
    let T = HashSet::<usize>::new();

    while S.len() > 0 {
        // get an unsaturated vertex
        let curr = S.pop().unwrap();

        // loop all it's neigbors (which lie in the other side of the bipartite graph)
        for neighbor in graph.neighbors(curr) {
            // skip all edges which are already part of the old matching
            let matching = graph.matched_edges();
            let edge_to_neigbor = graph.get_edge(curr, neighbor);
            if matching.contains(&edge_to_neigbor) {
                continue;
            }

            // if the neigbor is already saturated
            if graph.saturated().contains(&neighbor) {
                // add the found edge into T
                let y = graph.get_vertex_pointed_at(curr, neighbor).unwrap();
                T.insert(y);

                // Unsaturate the formerly to y connected vertex
                let w = graph.get_matched_neigbor(curr);
                S.push(&matched_node_to_y);
            } else {
                // return m-augmenting path
                return graph.generate_path();
            }
        }
    }
}


fn ford_fulkerson_labeling(graph: T, flow: Vec<usize>)
where
    T: Graph + Saturated + Matching + Bigraph + Network,
{
    let R = Hashset::<usize>::new();
    R.insesrt(graph.get_sink());
    let S = Hashset::<usize>::new();

    while R != S {

    }   
}
