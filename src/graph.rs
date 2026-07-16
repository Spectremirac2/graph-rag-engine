use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Deterministic FNV-1a 64-bit hash of a string. Used only to fix a stable,
/// structure-independent node update order for Label Propagation.
fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub community_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub entities: HashMap<String, Entity>,
    pub adj: HashMap<String, Vec<Relationship>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_entity(
        &mut self,
        id: String,
        name: String,
        entity_type: String,
        description: String,
    ) {
        let initial_comm_id = self.entities.len(); // Each starts in its own community
        self.entities.insert(
            id.clone(),
            Entity {
                id,
                name,
                entity_type,
                description,
                community_id: initial_comm_id,
            },
        );
    }

    pub fn add_relationship(
        &mut self,
        source: String,
        target: String,
        relation: String,
        weight: f64,
    ) {
        let edge = Relationship {
            source: source.clone(),
            target: target.clone(),
            relation,
            weight,
        };
        self.adj.entry(source).or_default().push(edge);
    }

    /// Build an UNDIRECTED, weighted neighbor adjacency once, in O(V + E).
    ///
    /// For every directed relationship `s -> t` with weight `w`, weight `w` is
    /// accumulated on both `s`'s and `t`'s neighbor lists (so the graph is treated
    /// as undirected for community detection and modularity). Self-loops
    /// (`s == t`) and edges touching unknown entities are ignored. Parallel edges
    /// between the same pair are summed.
    fn undirected_adjacency(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut adj: HashMap<String, HashMap<String, f64>> = HashMap::new();
        // Ensure every entity has an entry (isolated nodes included).
        for id in self.entities.keys() {
            adj.entry(id.clone()).or_default();
        }
        for edges in self.adj.values() {
            for edge in edges {
                if edge.source == edge.target {
                    continue;
                }
                if !self.entities.contains_key(&edge.source)
                    || !self.entities.contains_key(&edge.target)
                {
                    continue;
                }
                *adj.get_mut(&edge.source)
                    .unwrap()
                    .entry(edge.target.clone())
                    .or_insert(0.0) += edge.weight;
                *adj.get_mut(&edge.target)
                    .unwrap()
                    .entry(edge.source.clone())
                    .or_insert(0.0) += edge.weight;
            }
        }
        adj
    }

    /// Asynchronous Label Propagation Algorithm (LPA) for community detection,
    /// from scratch. Deterministic and reproducible.
    ///
    /// Design:
    /// - The graph is treated as UNDIRECTED: an undirected weighted adjacency is
    ///   precomputed once per call (O(V + E)), so each sweep is O(V + E) rather
    ///   than O(V * E).
    /// - Nodes are updated ASYNCHRONOUSLY in a deterministic order. The order is
    ///   the nodes sorted by a fixed hash of their id (FNV-1a), NOT by the raw id.
    ///   A raw lexicographic order groups structurally-related nodes contiguously
    ///   (e.g. a whole block/community before the next), which lets the first
    ///   block fully consolidate and then act as a concentrated attractor that
    ///   swallows the rest into one "monster community". Hashing decorrelates the
    ///   sweep order from community structure (the role randomized async order
    ///   plays in the LPA literature) while staying fully reproducible: the same
    ///   graph always yields the same order and the same partition.
    /// - Each node initially gets a unique label.
    /// - A node adopts the neighbor community with the largest total incident
    ///   edge weight. TIE-BREAK RULE: on equal total weight, the SMALLEST
    ///   community label id wins. This makes the outcome independent of hash-map
    ///   iteration order.
    /// - A node with no neighbors keeps its own label.
    /// - Iteration stops when a full sweep changes no label, or after
    ///   `max_iterations` sweeps (whichever comes first).
    ///
    /// The resulting labels are written back into each entity's `community_id`.
    pub fn run_community_detection(&mut self, max_iterations: usize) {
        if self.entities.is_empty() {
            return;
        }

        // Stable list of nodes and unique, deterministic initial labels (assigned
        // by raw sorted id, purely for interpretability).
        let mut nodes: Vec<String> = self.entities.keys().cloned().collect();
        nodes.sort();

        let adj = self.undirected_adjacency();

        let mut labels: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            labels.insert(node.clone(), idx);
        }

        for iter in 0..max_iterations {
            // Reshuffle the sweep order every iteration, deterministically: sort by
            // a hash of (iteration, id). Reshuffling each sweep stops any one block
            // from getting a head start and consolidating into an attractor before
            // the others; mixing in `iter` means the order differs sweep to sweep,
            // exactly the role randomized async order plays in the LPA literature -
            // but here it is a pure function of the graph, so still reproducible.
            let mut order = nodes.clone();
            order.sort_by(|a, b| {
                let ka = fnv1a(&format!("{iter}:{a}"));
                let kb = fnv1a(&format!("{iter}:{b}"));
                ka.cmp(&kb).then_with(|| a.cmp(b))
            });

            let mut changed = false;
            for node in &order {
                let neighbors = &adj[node];
                if neighbors.is_empty() {
                    continue; // isolated node keeps its own label
                }

                // Tally total incident weight per neighbor community.
                let mut tally: HashMap<usize, f64> = HashMap::new();
                for (nb, w) in neighbors {
                    *tally.entry(labels[nb]).or_insert(0.0) += *w;
                }

                // Pick the heaviest community; tie-break on smallest label id.
                let mut best_label = usize::MAX;
                let mut best_weight = f64::NEG_INFINITY;
                for (&label, &weight) in &tally {
                    if weight > best_weight || (weight == best_weight && label < best_label) {
                        best_weight = weight;
                        best_label = label;
                    }
                }

                if best_label != labels[node] {
                    labels.insert(node.clone(), best_label);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for (node, label) in labels {
            if let Some(entity) = self.entities.get_mut(&node) {
                entity.community_id = label;
            }
        }
    }

    /// Standard undirected, weighted modularity Q of a partition.
    ///
    /// Q = sum_c [ (in_c / 2m) - (tot_c / 2m)^2 ]
    /// where `in_c` is twice the total weight of edges internal to community `c`
    /// (each internal undirected edge counted once per direction), `tot_c` is the
    /// total weighted degree of nodes in `c`, and `2m` is the sum of all weighted
    /// degrees. Returns 0.0 for an edgeless graph.
    ///
    /// `labels` maps every entity id to its community id.
    pub fn modularity_of(&self, labels: &HashMap<String, usize>) -> f64 {
        let adj = self.undirected_adjacency();

        // Weighted degree of each node and total 2m.
        let mut two_m = 0.0f64;
        let mut degree: HashMap<&str, f64> = HashMap::new();
        for (node, neighbors) in &adj {
            let k: f64 = neighbors.values().sum();
            degree.insert(node.as_str(), k);
            two_m += k;
        }
        if two_m == 0.0 {
            return 0.0;
        }

        // Per-community accumulators.
        let mut in_c: HashMap<usize, f64> = HashMap::new();
        let mut tot_c: HashMap<usize, f64> = HashMap::new();
        for (node, neighbors) in &adj {
            let c = labels[node];
            *tot_c.entry(c).or_insert(0.0) += degree[node.as_str()];
            for (nb, w) in neighbors {
                if labels[nb] == c {
                    *in_c.entry(c).or_insert(0.0) += *w;
                }
            }
        }

        let mut q = 0.0f64;
        for (c, &tot) in &tot_c {
            let inside = *in_c.get(c).unwrap_or(&0.0);
            q += inside / two_m - (tot / two_m).powi(2);
        }
        q
    }

    /// Modularity Q of the current partition stored in each entity's
    /// `community_id`.
    pub fn modularity(&self) -> f64 {
        let labels: HashMap<String, usize> = self
            .entities
            .iter()
            .map(|(id, e)| (id.clone(), e.community_id))
            .collect();
        self.modularity_of(&labels)
    }

    /// Retrieve all descriptions of entities inside a specific community (Global Retrieval).
    pub fn get_community_descriptions(&self, community_id: usize) -> Vec<String> {
        let mut descriptions = Vec::new();
        for entity in self.entities.values() {
            if entity.community_id == community_id {
                descriptions.push(format!("{}: {}", entity.name, entity.description));
            }
        }
        descriptions
    }

    /// Set of entity ids reachable from `start_entity` within `max_hops` DIRECTED
    /// hops (following outgoing edges only), including the start at distance 0.
    ///
    /// Returned in BFS discovery order (which is shortest-distance order). This is
    /// the ground-truth traversal underlying [`multi_hop_search`]; it is exposed so
    /// the retrieval output can be checked against an independent reference.
    pub fn bfs_reachable(&self, start_entity: &str, max_hops: usize) -> Vec<String> {
        let mut order = Vec::new();
        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((start_entity.to_string(), 0));
        visited.insert(start_entity.to_string(), true);

        while let Some((curr, hop)) = queue.pop_front() {
            order.push(curr.clone());
            if hop >= max_hops {
                continue;
            }
            if let Some(edges) = self.adj.get(&curr) {
                for edge in edges {
                    if !visited.contains_key(&edge.target) {
                        visited.insert(edge.target.clone(), true);
                        queue.push_back((edge.target.clone(), hop + 1));
                    }
                }
            }
        }
        order
    }

    /// Performs Multi-hop graph search up to `max_hops` hops (Local Retrieval
    /// context collector). Traversal follows outgoing edges (directed), breadth
    /// first; a node is added to the context the first (shortest-distance) time it
    /// is reached.
    pub fn multi_hop_search(&self, start_entity: &str, max_hops: usize) -> Vec<String> {
        let mut context = Vec::new();
        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        queue.push_back((start_entity.to_string(), 0));
        visited.insert(start_entity.to_string(), true);

        while let Some((curr, hop)) = queue.pop_front() {
            // Add entity details.
            if let Some(entity) = self.entities.get(&curr) {
                context.push(format!(
                    "Entity: {} ({}) - {}",
                    entity.name, entity.entity_type, entity.description
                ));
            }

            if hop >= max_hops {
                continue;
            }

            // Traverse outgoing edges.
            if let Some(edges) = self.adj.get(&curr) {
                for edge in edges {
                    context.push(format!(
                        "Relation: {} --[{}]--> {}",
                        edge.source, edge.relation, edge.target
                    ));
                    if !visited.contains_key(&edge.target) {
                        visited.insert(edge.target.clone(), true);
                        queue.push_back((edge.target.clone(), hop + 1));
                    }
                }
            }
        }

        context
    }

    /// Convenience: the current partition as a map of community id -> sorted
    /// entity ids. Ordering is deterministic (BTreeMap + sorted vectors).
    pub fn communities(&self) -> BTreeMap<usize, Vec<String>> {
        let mut out: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (id, e) in &self.entities {
            out.entry(e.community_id).or_default().push(id.clone());
        }
        for v in out.values_mut() {
            v.sort();
        }
        out
    }
}
