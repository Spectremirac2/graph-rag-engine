use crate::graph::KnowledgeGraph;

#[derive(Default)]
pub struct GraphRagIndexer {
    pub graph: KnowledgeGraph,
}

impl GraphRagIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses structured document lines and adds entities/relationships.
    /// Format: "SourceEntity -- Relation -- TargetEntity (Description of target or source)"
    pub fn ingest_document(&mut self, doc: &str) {
        for line in doc.lines() {
            let line = line.trim();
            if line.is_empty() || !line.contains(" -- ") {
                continue;
            }

            let parts: Vec<&str> = line.split(" -- ").collect();
            if parts.len() >= 3 {
                let source_raw = parts[0].trim();
                let relation = parts[1].trim().to_string();
                let rest = parts[2].trim();

                // Check if there is description in parentheses: e.g. "Google (A search engine company)"
                let mut target_name = rest.to_string();
                let mut description = format!("{} related to {}", target_name, source_raw);

                if let Some(start_paren) = rest.find('(') {
                    if let Some(end_paren) = rest.find(')') {
                        target_name = rest[..start_paren].trim().to_string();
                        description = rest[start_paren + 1..end_paren].trim().to_string();
                    }
                }

                // Add source entity (default description if new)
                if !self.graph.entities.contains_key(source_raw) {
                    self.graph.add_entity(
                        source_raw.to_string(),
                        source_raw.to_string(),
                        "Entity".to_string(),
                        format!("Concept: {}", source_raw),
                    );
                }

                // Add target entity
                self.graph.add_entity(
                    target_name.clone(),
                    target_name.clone(),
                    "Target".to_string(),
                    description,
                );

                // Add relationship
                self.graph
                    .add_relationship(source_raw.to_string(), target_name, relation, 1.0);
            }
        }

        // Run community detection to index node clusters after ingestion.
        // The cap is generous; async LPA converges well before this on small graphs.
        self.graph.run_community_detection(100);
    }

    /// Resolves query and constructs a retrieved context prompt.
    /// If query contains "summary" or "themes", it performs a Global community retrieval.
    /// Otherwise, it performs a Local multi-hop entity retrieval.
    pub fn retrieve_context(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();

        if query_lower.contains("summary")
            || query_lower.contains("theme")
            || query_lower.contains("global")
        {
            // --- GLOBAL RETRIEVAL (Community-based) ---
            // Group entities by community
            let mut community_groups: std::collections::HashMap<usize, Vec<String>> =
                std::collections::HashMap::new();
            for entity in self.graph.entities.values() {
                community_groups
                    .entry(entity.community_id)
                    .or_default()
                    .push(entity.name.clone());
            }

            let mut context = String::from("GLOBAL CONTEXT (Community Clusters):\n");
            for (comm_id, members) in community_groups {
                let desc_list = self.graph.get_community_descriptions(comm_id);
                context.push_str(&format!("Community Cluster {}:\n", comm_id));
                context.push_str(&format!("  Members: {}\n", members.join(", ")));
                context.push_str("  Details:\n");
                for desc in desc_list {
                    context.push_str(&format!("    - {}\n", desc));
                }
            }
            context
        } else {
            // --- LOCAL RETRIEVAL (Entity multi-hop search) ---
            // Simple match to find start entity in query
            let mut start_entity = String::new();
            for entity_id in self.graph.entities.keys() {
                if query_lower.contains(&entity_id.to_lowercase())
                    && entity_id.len() > start_entity.len()
                {
                    start_entity = entity_id.clone();
                }
            }

            if start_entity.is_empty() {
                return "LOCAL CONTEXT: No matching entities found in knowledge graph for query."
                    .to_string();
            }

            let facts = self.graph.multi_hop_search(&start_entity, 3);
            let mut context = format!("LOCAL CONTEXT (Entity Walk for '{}'):\n", start_entity);
            for fact in facts {
                context.push_str(&format!("  * {}\n", fact));
            }
            context
        }
    }
}
