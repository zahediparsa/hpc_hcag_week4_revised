# First Assignment

The main goal of the first assignment is to explore and evaluate clustering algorithms alongside Large Language Models (LLMs) to extract and describe the architecture of an existing software system.

The aim is to answer the two research questions (RQs):

1. How do different clustering algorithms vary in their ability to accurately determine the architectural components of a system?
2. How do different prompting techniques impact LLM's ability to accurately describe architectural components based on source code?

Each group is assigned a specific part within a Java project. Table I below outlines these group assignments and provides the necessary project details. For the exact file paths of your assigned components, please refer to [1].

## Table I: Group Projects and Repos

| Group | Project | Part under focus | Code |
|---|---|---|---|
| Groups 1 & 6 & 11 | Tika | Detect & parser | <https://github.com/apache/tika> |
| Groups 2 & 7 & 12 | JClouds | Azureblob | <https://github.com/apache/jclouds> |
| Groups 3 & 8 & 13 | Hadoop MapReduce | Client core | <https://github.com/apache/hadoop> |
| Groups 4 & 9 & 14 | Lucene | Codecs | <https://github.com/apache/lucene> |
| Groups 5 & 10 & 15 | Hadoop Yarn | Resource manager, scheduler, capacity | <https://github.com/apache/hadoop> |

To answer the research questions, we will follow a phased workflow over the coming weeks:

# Week 1: Class Dependency Extraction

1. Compile your assigned Java project to generate the compiled `*.jar` file.

2. Extract system-wide dependencies: Use the ARCADE JavaParser tool to extract all project relationships and generate a master `.rsf` file. For the documentation on the ARCADE tool suite, refer to [2].

3. Filter the dependencies to isolate classes strictly relevant to your group's assigned component (e.g., Server API), resulting in a focused `*.rsf` file.

   A standard `*.rsf` file lists dependencies line-by-line in the format:

   ```text
   depends Source_Package Target_Package
   ```

   Filtering here means extracting only the lines where either the source or the target packages are in the part under focus of your assigned project.

4. Cluster and Tune: Use the ARCADE Clusterer and ARCADE ACDC tools to run the WCA, Limbo, and ACDC (structural) clustering algorithms. Adjust the tool's parameters, if applicable, as needed to optimize your results and generate the final clustered `*.rsf` output [2].

# Week 2: Clusters Evaluation & Initial Exploration of LLMs

## 1. Conduct a comparative evaluation across applied clustering approaches from Week 1

Use the ARCADE a2a (Architecture-to-Architecture) and Cvg (Coverage) metric tools to measure the structural similarity and distance between the two architectures (`.rsf` outputs generated from your clustering phase).

Provide the relative file paths for the respective `.rsf` outputs generated from your clustering phase [2].

Calculate the a2a and cvg among all pairs of clustering algorithms. Determine which clustering algorithm produces most similar clusters.

## 2. Prepare LLMs for prompting

In this step, you will explore and prepare LLMs for prompting.

Each group is assigned specific lightweight & Heavyweight LLMs. Table II below outlines these group LLMs and their full path where they are hosted in the Hugging Face (HF) platform.

## ✨ Table II: Group Light- and Heavy-weight LLMs

| Group | Lightweight Models<br>(Colab / T4 GPU) | Heavyweight Models<br>(HPC / 2x A100) |
|---|---|---|
| Groups 1 & 6 & 11 | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-72B-Instruct` |
| Groups 2 & 7 & 12 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~~`deepseek-ai/DeepSeek-V4-Pro`~~<br>`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`<br>&<br>`deepseek-ai/deepseek-llm-67b-chat` |
| Groups 3 & 8 & 13 | `ibm-granite/granite-3.3-8b-instruct` | `ibm-granite/granite-34b-code-instruct-8k` |
| Groups 4 & 9 & 14 | `mistralai/Mistral-7B-Instruct-v0.3` | ~~`mistralai/Mistral-Large-3-675B-Instruct-2512`~~<br>`mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Groups 5 & 10 & 15 | `ByteDance-Seed/Seed-Coder-8B-Instruct` | ~~`zai-org/GLM-5`~~<br>`nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` |

## First: Prototyping in Google Colab (Browser), Suitable for Lightweight Models

Use the provided Colab notebook [3] to verify your prompt logic before scaling to the cluster.

- Open the provided notebook link and select **File > Save a copy in Drive**.
- Do not paste your HF token into a code cell. Instead, click the **Key** icon (**Secrets**) in the left sidebar, add a new secret named `HF_TOKEN`, and paste your token as the value.
- Modify the `model_name` variable to match your group's assigned Lightweight model.
- Follow the instructions in the script.

## Second: Production Scaling on the (HPC) (Slurm Jobs)

**[Needs access to university HPC: Done]**

Move to the university High Performance Computer (HPC) [4] cluster to process the full part of your assigned project.

- **Script Modification:** Update the `model_name` in your `sample.py` to your assigned Heavyweight model.
- **Token Handling:** The provided `sample.sh` is designed to read your HF token from a local environment variable.
- **Execution:** Ensure you follow the instructions in the comments of the provided scripts [5] to provide your token securely during the `sbatch` submission.

# Week 3: Semantic Clustering & Evaluation

Apply semantic clustering algorithms and once again conduct a comparative evaluation across the different clustering approaches.

This week, you will implement ARC clustering, an algorithm that combines both structural and semantic similarities between source code files. In summary, you should follow these steps:

1. Apply the assigned embedding model for each source code file. The output is a vector for each source code file.

Each group is assigned a specific embedding model(s). Table III lists these group-level models along with their full paths on the Hugging Face (HF) platform.

## Table III: Group Code Embedding Models

| Group | Embedding Model |
|---|---|
| Groups 1 & 6 & 11 | `nomic-ai/nomic-embed-code` |
| Groups 2 & 7 & 12 | `jinaai/jina-code-embeddings-1.5b` |
| Groups 3 & 8 & 13 | `ibm-granite/granite-embedding-english-r2`<br>&<br>`jinaai/jina-code-embeddings-0.5b` |
| Groups 4 & 9 & 14 | `Qodo/Qodo-Embed-1-7B` |
| Groups 5 & 10 & 15 | `nomic-ai/CodeRankEmbed`<br>&<br>`jinaai/jina-code-embeddings-0.5b` |

2. Compute the cosine similarity between embedding vectors to measure the semantic similarity across files. This produces a semantic similarity matrix where each entry (between 0 and 1) indicates the cosine similarity between a pair of files.

3. Construct a matrix representing structural dependencies using the filtered dependency RSF file from Week 1. Then, normalize the matrix so that all values fall between 0 and 1. The result is a normalized structural similarity matrix.

4. Combine structural similarity and semantic similarity matrices in one combined similarity matrix, which contain both the structural dependencies and semantic similarities.

5. Convert the combined similarity matrix to a distance matrix (by subtracting each value from 1), then pass this distance matrix to the Agglomerative clustering algorithm.

   The notebook [6], running in Google Colab, contains code snippets that illustrate how such code is constructed.

6. Apply the comparative evaluation from Week 2 (point 1) on the output clusters from the current week.

# ✨ Week 4: LLM-Based Architectural Recovery

Apply LLMs prompting guidelines to generate architectural titles and a high-level descriptive summary for each identified cluster.

To achieve this week's objective, you will apply the Hierarchical summarization [7], enabling you to provide all source code to your assigned LLM(s)* despite the limitation of window context. Proceed as follows for each cluster:

## 1. Process Leaf Nodes (Files)

Start at the deepest level of your directory structure and work upwards. For every file:

- Pass the raw source code to the LLM.
- Prompt the LLM to extract a semantic summary from the file: **Key functionality, Core logic, Inputs/Outputs, and Dependencies**.
- Save this generated summary.

## 2. Process Branch Nodes (Directories)

Once all files within a specific directory are summarized:

- Gather the summaries of those constituent files and any already-summarized subdirectories.
- Pass this list of summaries to the LLM (**do not pass the raw code!**).
- Prompt the LLM to generate: **A high-level descriptive summary and a title** explaining the module's overall behaviour, architecture, and how the components interact within the cluster.

---

\* Please check Table II if you have been assigned different LLM(s).

---

# References

[1] The provided `Projects_Parts_under_focus.xlsx`

[2] The provided `Arcade documentation.pdf`

[3] `[READ-ONLY]DS4SE26Week2.ipynb`

[4] UPB PC2 WiKi

[5] The provided Slurm sample

[6] `[READ-ONLY]DS4SE26Week3.ipynb`

[7] HCAG: Hierarchical Abstraction and Retrieval-Augmented Generation on Theoretical Repositories with LLMs by Yusen Wu

---

# Tentative Project Roadmap (Subject to Adjustment)

**Note:** The following workflow outlines our goals for the upcoming week(s) and will be written in detail and may be refined as the project progresses.

## Week 5

Finalize your group's research report and presentation.
