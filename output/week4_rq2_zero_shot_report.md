# RQ2 Methodology: Zero-Shot Hierarchical Summarization

## Research Question

**RQ2:** How do prompting techniques impact the LLM's ability to accurately describe architectural components based on source code?

## Prompting Method Used

This Week 4 pipeline uses a **zero-shot prompting methodology**. No examples are included in the prompt. The LLM is instructed directly to summarize source-code files and then aggregate those summaries into higher-level architectural descriptions.

## Hierarchical Process

1. **Leaf/file level:** each Java file is summarized from raw source code using a prompt requesting key functionality, core logic, inputs/outputs, and dependencies.
2. **Directory level:** summaries from files and already-summarized subdirectories are aggregated into directory-level architectural abstractions.
3. **Cluster level:** directory-level summaries are aggregated into the final cluster title and high-level description.

## Required Cluster Description Constraints

Each final cluster description is instructed to include:

- **Components and interactions:** how the distinct parts of the cluster work together.
- **Quality attributes:** non-functional qualities such as scalability, maintainability, reliability, security, or performance.
- **Technology used:** Java, Hadoop MapReduce, APIs, frameworks, or other visible technologies.
- **Conciseness:** fewer than 150 words.

## Clustering Algorithms Covered

The same zero-shot hierarchical summarization process is applied independently to all three clustering algorithms:

- ARC
- ACDC
- LIMBO

The input CSV files for ACDC and LIMBO were prepared as file-level clusters (one row per `.java` file) prior to running this pipeline, as required by the hierarchical summarization methodology. Raw ACDC and LIMBO output can produce inner-class based entries; those were resolved to their enclosing source files before the CSVs were finalised.

## Output Format

The final submission contains one CSV file per clustering algorithm:

- `ARC_hierarchical_summarization_results.csv`
- `ACDC_hierarchical_summarization_results.csv`
- `LIMBO_hierarchical_summarization_results.csv`

Each CSV contains exactly:

```csv
cluster_ID,files,title,description
```
