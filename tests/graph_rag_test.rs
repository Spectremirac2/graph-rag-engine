use graph_rag_engine::indexer::GraphRagIndexer;

#[test]
fn test_graph_rag_ingestion_and_retrieval() {
    let mut indexer = GraphRagIndexer::new();

    // ingest a document with structured entities and relationships
    let document = "\
Alice -- works_at -- Google (A software company based in Mountain View)
Google -- founded_by -- LarryPage (Co-founder of Google)
LarryPage -- went_to -- Stanford (A prestigious university in California)
Bob -- member_of -- Stanford (Bob is a PhD student at Stanford)
";

    indexer.ingest_document(document);

    // Verify entity count
    // Entities: Alice, Google, LarryPage, Stanford, Bob
    assert_eq!(indexer.graph.entities.len(), 5);

    // --- PHASE 1: LOCAL RETRIEVAL (Multi-hop Walk) ---
    let local_context = indexer.retrieve_context("Tell me about Alice and her environment");

    // Assert that the retrieval context walked the graph to Google and Stanford!
    assert!(local_context.contains("Alice"));
    assert!(local_context.contains("Google"));
    assert!(local_context.contains("LarryPage"));
    assert!(local_context.contains("Stanford"));

    // --- PHASE 2: GLOBAL RETRIEVAL (Community-based Summary) ---
    let global_context = indexer.retrieve_context("Show me the global summary and themes");

    // Assert that global retrieval aggregates community clusters
    assert!(global_context.contains("GLOBAL CONTEXT"));
    assert!(global_context.contains("Community Cluster"));
}
