# Operational Blueprint: PolicyProbe Sub-System Architecture

## 1. Primary Objective
Construct a completely deterministic, CPU-bound pipeline to evaluate multi-agent operating rulebooks and synthesize adaptive decision graphs under strict diagnostic budget constraints and adversarial noise parameters. The architecture must strictly bifurcate semantic reasoning from graph traversal methodologies.

## 2. Computational Constraints
- **Hardware Protocol:** Execution environment is restricted to 10 CPU cores and 62 GB RAM. 
- **Absolute GPU Prohibition:** Do not invoke, instantiate, or assume CUDA or any GPU-accelerated dependencies. 
- **Model Parameters:** Limit model selections to quantized, distilled, or compact instruction-tuned CPU-compatible architectures (e.g., DeBERTa-v3-base-mnli). 

## 3. Phase 1: Deterministic State Matrix Generation (Phase_1_Estimator)
Before graph instantiation, you must pre-compute a complete 2D prediction matrix for all `(Candidate, Probe)` pairs. 

### 3.1. Lexical Rulebook Parsing
Implement a deterministic parsing module utilizing robust regular expressions to extract structured operational configurations from all candidate rulebooks. You must map:
- Capability ownership arrays.
- Conditional clauses (e.g., after-hours, international, urgent).
- Compound request precedence.
- Agent-specific fallback directives.

### 3.2. Zero-Shot Intent Classification (NLI)
Abandon hardcoded cosine similarity thresholds. Deploy a Zero-Shot Natural Language Inference (NLI) classifier to map probe requests to capability labels.
- **Preprocessing:** Strip all meta-instructions (e.g., "Immediate handling is requested") and multiple-choice formatting prior to embedding.
- **Classification:** Append an explicit "Other or Unrelated" label to the dynamically extracted capability list to natively handle Out-of-Distribution (OOD) diagnostic queries.

### 3.3. Action Matrix Synthesis
Cross-reference the NLI-classified probe intents and detected conditions against the parsed rulebook logic to populate a matrix dictating the canonical action (`answer_with_X`, `delegate_to_X`, or `refuse`) for every candidate-probe intersection.

## 4. Phase 2: Cost-Aware Graph Synthesis (Phase_2_Search)
Utilize the Phase 1 Matrix to construct the decision tree via Cost-Penalized Information Gain. 

### 4.1. Node Splitting Optimization
For any active subset of candidates $S$, evaluate all remaining diagnostic probes $P$ to identify the optimal split using the following objective function:

$$\text{Score}(P) = \frac{H(S) - \sum_{v \in V} \frac{|S_v|}{|S|} H(S_v)}{C_P^\alpha}$$

Where:
- $H(S)$ is the Shannon entropy of the current candidate subset.
- $V$ is the set of unique predicted actions (branches).
- $S_v$ is the resulting subset of candidates assigned to action $v$.
- $C_P$ is the designated audit cost of the probe.
- $\alpha$ is a hyperparameter ($\approx 0.5 - 1.0$) balancing information gain against budget depletion.

### 4.2. Terminal Leaf Calibration
Upon reaching absolute node isolation (Entropy = 0) or exhausting the `audit_budget` / `maximum_depth`, construct a terminal leaf node. The candidate probabilities must sum perfectly to 1.0. A correctly identified candidate must possess a strictly dominant probability (no ties).

## 5. Phase 3: Adversarial Redundancy (Fault Tolerance)
If the episode parameter designates `noise_budget = 1`, standard graph topologies are critically insufficient. You must engineer 1-Fault Tolerance.

### 5.1. The 2-Separation Mandate
A candidate must not be assigned a probability of 1.0 unless it is algorithmically separated from every other candidate by a minimum of **two distinct probes** along that path. 
- If a candidate is isolated by only a single probe, force a secondary, redundant probe query to corroborate the isolation. 
- Ensure the `__default__` branch maps to subgraphs capable of re-routing corrupted executions back to the correct candidate terminal.

## 6. Output Protocol
The final artifact must be a valid JSON serialization encapsulated within `submission.csv` containing the precise topological mapping of nodes, branches, and probability distributions without exceeding `maximum_nodes` or `audit_budget`.
