# Design

## Determinism is the requirement, and it is what breaks the algorithm

A graph-RAG pipeline retrieves by community. If the same corpus produces a different partition on
each run, then the same question returns different context, an answer that was wrong yesterday
cannot be reproduced today, and no failure downstream can be attributed to anything. Determinism
here is not tidiness; it is the difference between a substrate you can debug and one you cannot.

Label Propagation is the natural algorithm for this — near-linear, no target community count, no
resolution parameter. It is also, in its standard form, **randomized on purpose**. Nodes are
visited in a random asynchronous order, and that randomness is not incidental: it is what stops
the sweep from correlating with the structure it is trying to find.

So the requirement and the algorithm are in direct conflict, and the obvious resolution is wrong.

## Sorting by id produces one monster community

Remove the randomness the simple way — sweep the nodes in sorted order — and the algorithm
degrades in a specific, predictable way.

Sorted identifiers group structurally related nodes contiguously. Entities extracted from the
same document, or sharing a prefix, land next to each other, which is exactly the population that
belongs to one community. The sweep therefore consolidates that block completely before reaching
anything else, and a fully consolidated block is a concentrated attractor: every later node sees a
large, heavily-weighted neighbour community and joins it. The partition collapses toward one
community that has swallowed the graph.

This is a deterministic algorithm producing a reproducibly bad answer, which is worse than a
randomized one producing a decent answer, because it looks stable.

## Hashing the sweep order

The nodes are swept in order of a **fixed FNV-1a hash of the identifier**, not the identifier.

The hash has no meaning and that is the entire point. It decorrelates the sweep order from the
graph's structure — which is the job randomized async order does in the LPA literature — while
being a pure function of the id, so the same graph always yields the same order and therefore the
same partition. Reproducibility comes from the function being fixed; quality comes from the
function being structure-blind.

It is worth being clear about what this is not: it is not a random order, and it is not a good
order. It is an order chosen to be *uncorrelated with the answer*, which is a weaker and
achievable property.

## The tie-break closes the other half

Fixing the sweep order is not enough on its own, because the choice made at each node also has a
free parameter. A node adopts the neighbour community with the largest total incident edge weight,
and when two communities tie, something has to decide.

Left unspecified, that something is hash-map iteration order — which in Rust varies between
processes and would make the whole determinism claim false while every single-process test passed.
This is the failure mode that bites simulations across this portfolio: an unordered container
feeding an ordered decision.

The rule is: **on equal weight, the smallest community label id wins.** Total, deterministic,
and independent of how any map happens to iterate. The comment in `graph.rs` states it in capitals
because it is load-bearing rather than stylistic.

## Community detection has no ground truth, so the test builds the case where it does

The hard part of testing this is that "the right communities" is not a defined quantity for a
general graph. Modularity is a heuristic, not an answer, and comparing against another
implementation only tells you that two heuristics agree.

The suite gets around that by constructing families where the answer *is* defined:

**Disjoint cliques — an exact oracle.** On a random union of cliques, the correct partition is
precisely the connected components, which an independent union-find computes exactly. The
assertion is not approximate: two nodes share a label **iff** they share a component, and every
clique is monochromatic. An unfalsifiable problem becomes an exactly falsifiable one by
restricting the input family.

**Planted partitions — a probabilistic oracle, asserted probabilistically.** Stochastic block
models come with planted labels, but recovery is not guaranteed on any individual graph. So the
test asserts a success *rate* — 39/40, against a 0.90 threshold — rather than per-graph success.
Asserting per-graph would produce a test that fails on a fair sample; asserting nothing would
prove nothing; asserting a rate is the claim that is actually true.

**Modularity monotonicity — metamorphic.** The final partition's Q is at least the singleton
partition's. Weak, and it holds for every graph, which is why it can be checked on arbitrary
random input where the exact oracles cannot.

**Multi-hop against reference BFS — an exact differential.** `multi_hop_search`'s entity set must
equal an independent k-bounded directed BFS, over hop limits spanning both below and above the
graph diameter. Below and above matters: a search that ignored its hop limit would agree with the
reference on every case where the limit exceeded the diameter.

## The anti-triviality guard

All of the above depends on the generator producing the interesting shapes, and a passing run
would look identical if it stopped. So the suite asserts that the corpus actually contained
multi-community graphs, cliques of varying size, disconnected graphs, and hop limits on both sides
of the diameter.

A generator that drifted to emitting single-community graphs would leave every layer above green
while checking almost nothing.

## What was rejected

**Randomized LPA.** The standard algorithm, better on average, and unusable as a retrieval
substrate for the reason in the first section.

**Sorting by identifier.** Covered above. Deterministic and reproducibly wrong.

**Seeding a PRNG for the sweep order.** This would have worked — a seeded shuffle is reproducible
and structure-blind. It costs a generator, a seed to thread through the API, and a seed to record
alongside every stored partition, since the partition is only reproducible if you still have it. A
fixed hash needs none of that: the order is recoverable from the graph alone.

**Louvain or Leiden.** Better partitions and a larger implementation, with a resolution parameter
that turns "what are the communities" into a tuning question. The README says the graph and the
algorithms are hand-implemented with no graph or ML libraries, and that constraint is what makes
the trade-offs in this file visible at all.

## The one dependency

`serde`, for serialisation, and nothing else. It is not a graph library or an ML library, which is
what the README's claim is about — the adjacency structure, the label propagation, the union-find
reference and the BFS reference are all here.

It earns its place by being the boundary rather than the substance: a partition that cannot be
written out and read back is not a substrate anything can be built on, and hand-rolling a
serialisation format would add a parser to maintain and a class of bug that has nothing to do with
graphs. The PRNG in the test suite went the other way and is hand-rolled, because there the
requirement was a *specific* reproducible stream rather than a general capability.

> **Not established by the record.** The monster-community failure is described in `graph.rs` as
> the reason for hashing the sweep order, which reads like something that was observed rather than
> anticipated. If the sorted-order version was actually built and produced that partition, that is
> the more useful account, and the history does not show it.

## What this is not

Not a RAG pipeline. A graph substrate: entity storage, community detection, and multi-hop
retrieval. No embeddings, no chunking, no generation, no LLM anywhere in it.

Not scalable. In-memory, single process, adjacency in hash maps. The algorithms are near-linear
and nothing about the implementation is tuned.

Not a claim that these are the best communities. LPA with a deterministic sweep is a defensible
choice for a reproducible substrate and it is not Leiden.
