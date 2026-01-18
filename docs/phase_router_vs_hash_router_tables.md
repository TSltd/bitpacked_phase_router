| Feature / Aspect                        | Hash Router                                            | Phase Router                                                              |
| --------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| **Column Skew / Load Balance**          | Can produce hotspots if input is structured            | Lower skew; smoother, more balanced distribution                          |
| **Determinism**                         | Seed-deterministic (if keyed hash)                     | Seed-deterministic; fully reproducible                                    |
| **Speed / Throughput**                  | Very fast; minimal computation                         | Slower due to phase rotations and permutations                            |
| **Handling Structured Inputs**          | Poor; input patterns may cause collisions              | Better; decorrelates structured inputs with phase spreading               |
| **Fan-Out / Row Limits**                | Can enforce hard caps trivially                        | Enforces hard caps with randomized top-k selection                        |
| **Memory / Implementation Complexity**  | Simple; lightweight                                    | More complex; requires bit-packed rotations and permutations              |
| **Randomness / Statistical Properties** | Independent key mapping                                | Structured phase overlap preserves degree distribution                    |
| **Use Case Strengths**                  | Raw speed, hash table mapping, basic load distribution | Balanced routing, MoE routing, sparse attention, reproducible experiments |
| **Use Case Limitations**                | Poor for highly structured or adversarial inputs       | Slower; may not be suitable for high-throughput hashing applications      |

| Application / Use Case                | Hash Router Advantage                        | Hash Router Limitation                         | Phase Router Advantage                                   | Phase Router Limitation                    |
| ------------------------------------- | -------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- | ------------------------------------------ |
| **MoE / Sparse Expert Routing**       | Simple, fast mapping of tokens to experts    | Can overload some experts, poor load balance   | Smooth, low-skew routing, respects fan-out, reproducible | Slower, more computation per routing       |
| **Sparse Attention / Graph Networks** | Fast assignment of nodes to attention slots  | Input structure may produce hotspots           | Balanced distribution even for structured inputs         | Higher latency for large matrices          |
| **Hash Table / Key-Value Mapping**    | Very fast key mapping and lookup             | Collisions with structured or adversarial keys | Deterministic mapping, evenly spreads keys               | Slower than standard hash table methods    |
| **Capacity Planning / Simulation**    | Lightweight, easy to simulate many scenarios | Statistical properties may be unreliable       | Reproducible, low-skew, matches expected-degree stats    | Slower; requires bit-packed implementation |
| **Input Anonymization / Shuffling**   | Not directly applicable                      | N/A                                            | Can decorrelate structured input rows, anonymizes data   | Extra computation; optional step           |
