"""
preprocess_clusters.py
======================
Converts ACDC and LIMBO .rsf files into the same file-level CSV format used by
the ARC clusters_arc.csv (columns: cluster, class, path).

RSF format (both tools):
    contain <cluster_id_or_name> <fully_qualified_class_name>

Inner-class handling (required by Week 4 instructions):
    ACDC and LIMBO often assign inner classes (A$B) to clusters.
    We map every entry to its top-level file (A) using this priority:
      1. If a top-level class has a *direct* (non-inner) entry, that cluster wins.
      2. If only inner-class entries exist, the most frequent cluster wins.
         Ties are broken by the lowest cluster id.

ACDC cluster names are FQNs ending in '.ss' (e.g. org.apache.hadoop.mapreduce.util.ss).
They are mapped to sequential integer ids sorted alphabetically.

Path derivation:
    org.apache.hadoop.mapreduce.<rest> -> <rest/with/slashes>.java
    e.g. org.apache.hadoop.mapreduce.lib.input.FileInputFormat
         -> lib/input/FileInputFormat.java

Usage:
    python preprocess_clusters.py

Outputs written to the same directory as this script:
    clusters_acdc.csv
    clusters_limbo.csv
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
ACDC_RSF = HERE / "rsf_files" / "ACDC_hadoop_mapreduce_focused.rsf"
LIMBO_RSF = HERE / "rsf_files" / "LIMBO_hadoopmapreduce-3.4.1_IL_100_clusters.rsf"

MAPREDUCE_PREFIX = "org.apache.hadoop.mapreduce."


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def fqn_to_path(fqn: str) -> str:
    """Convert a fully-qualified class name to a relative .java path.

    org.apache.hadoop.mapreduce.lib.input.FileInputFormat
        -> lib/input/FileInputFormat.java
    org.apache.hadoop.mapreduce.Job
        -> Job.java
    """
    if fqn.startswith(MAPREDUCE_PREFIX):
        rest = fqn[len(MAPREDUCE_PREFIX):]
    else:
        rest = fqn  # fallback – should not happen for Hadoop MR classes
    return rest.replace(".", "/") + ".java"


def outer_class(fqn: str) -> str:
    """Strip inner-class suffix: 'A$B$C' -> 'A'."""
    return fqn.split("$")[0]


def parse_rsf(path: Path) -> list[tuple[str, str]]:
    """Return list of (raw_cluster, raw_class_fqn) from an RSF file."""
    entries: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3 or parts[0] != "contain":
                continue
            entries.append((parts[1], parts[2]))
    return entries


def resolve_file_clusters(
    entries: list[tuple[str, str]],
    numeric_tiebreak: bool = False,
) -> dict[str, str]:
    """Map each top-level class FQN -> cluster (raw string).

    Priority:
        1. Direct (non-inner) entry for this class.
        2. Most frequent cluster among inner-class entries for this class.
           Ties: smallest cluster id (lexicographic for ACDC, numeric for LIMBO).
    """
    # Collect direct and inner entries per outer class.
    direct: dict[str, str] = {}          # outer_fqn -> cluster (first direct wins)
    inner_clusters: dict[str, list[str]] = defaultdict(list)  # outer_fqn -> [cluster, ...]

    for cluster, cls in entries:
        outer = outer_class(cls)
        if "$" in cls:
            inner_clusters[outer].append(cluster)
        else:
            if outer not in direct:
                direct[outer] = cluster

    # Build final mapping.
    all_outers = set(direct) | set(inner_clusters)
    result: dict[str, str] = {}
    for fqn in all_outers:
        if fqn in direct:
            result[fqn] = direct[fqn]
        else:
            # Only inner-class entries: pick most frequent cluster.
            counter = Counter(inner_clusters[fqn])
            most_common_count = counter.most_common(1)[0][1]
            candidates = [c for c, n in counter.items() if n == most_common_count]
            if numeric_tiebreak:
                result[fqn] = min(candidates, key=lambda c: int(c))
            else:
                result[fqn] = min(candidates)  # stable tie-break

    return result


def build_csv_rows(
    fqn_to_cluster: dict[str, str],
    cluster_name_to_int: dict[str, int] | None,
) -> list[dict]:
    """Produce sorted list of {cluster, class, path} dicts."""
    rows = []
    for fqn, raw_cluster in fqn_to_cluster.items():
        cluster_int = (
            cluster_name_to_int[raw_cluster]
            if cluster_name_to_int
            else int(raw_cluster)
        )
        rows.append(
            {
                "cluster": cluster_int,
                "class": fqn,
                "path": fqn_to_path(fqn),
            }
        )
    rows.sort(key=lambda r: (r["cluster"], r["path"]))
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cluster", "class", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# ACDC
# ---------------------------------------------------------------------------
def process_acdc(rsf_path: Path, out_path: Path) -> None:
    print("\n[ACDC] Parsing RSF ...")
    entries = parse_rsf(rsf_path)
    print(f"  Raw entries: {len(entries)}")

    # Map cluster names (FQNs ending in .ss) to sorted integer ids.
    raw_clusters = sorted({c for c, _ in entries})
    cluster_name_to_int: dict[str, int] = {name: i for i, name in enumerate(raw_clusters)}
    print(f"  Clusters ({len(raw_clusters)}):")
    for name, idx in cluster_name_to_int.items():
        print(f"    {idx:2d}  {name}")

    fqn_to_cluster = resolve_file_clusters(entries)
    print(f"  Unique file-level classes: {len(fqn_to_cluster)}")

    # Report conflict resolution.
    outer_raw: dict[str, list[str]] = defaultdict(list)
    for cluster, cls in entries:
        outer_raw[outer_class(cls)].append(cluster)
    conflicts = {k: set(v) for k, v in outer_raw.items() if len(set(v)) > 1}
    if conflicts:
        print(f"  Cross-cluster conflicts resolved ({len(conflicts)}):")
        for fqn, clusters in sorted(conflicts.items()):
            chosen = fqn_to_cluster[fqn]
            chosen_int = cluster_name_to_int[chosen]
            print(f"    {fqn.replace(MAPREDUCE_PREFIX,'')}: clusters {sorted(clusters)} -> chose {chosen} (id={chosen_int})")

    rows = build_csv_rows(fqn_to_cluster, cluster_name_to_int)
    write_csv(rows, out_path)


# ---------------------------------------------------------------------------
# LIMBO
# ---------------------------------------------------------------------------
def process_limbo(rsf_path: Path, out_path: Path) -> None:
    print("\n[LIMBO] Parsing RSF ...")
    entries = parse_rsf(rsf_path)
    print(f"  Raw entries: {len(entries)}")

    unique_cluster_ids = sorted({int(c) for c, _ in entries})
    print(f"  Cluster ids: {min(unique_cluster_ids)} .. {max(unique_cluster_ids)} ({len(unique_cluster_ids)} total)")

    fqn_to_cluster = resolve_file_clusters(entries, numeric_tiebreak=True)
    print(f"  Unique file-level classes: {len(fqn_to_cluster)}")

    # Report how many conflicts were resolved.
    outer_raw: dict[str, list[str]] = defaultdict(list)
    for cluster, cls in entries:
        outer_raw[outer_class(cls)].append(cluster)
    conflicts = {k: set(v) for k, v in outer_raw.items() if len(set(v)) > 1}
    if conflicts:
        print(f"  Cross-cluster conflicts resolved ({len(conflicts)}):")
        for fqn, clusters in sorted(conflicts.items()):
            chosen = fqn_to_cluster[fqn]
            short = fqn.replace(MAPREDUCE_PREFIX, "")
            print(f"    {short}: clusters {sorted(int(c) for c in clusters)} -> chose {chosen}")

    rows = build_csv_rows(fqn_to_cluster, cluster_name_to_int=None)
    write_csv(rows, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("RSF -> CSV pre-processor (Week 4 HCAG pipeline)")
    print("=" * 60)

    process_acdc(ACDC_RSF, HERE / "input" / "clusters_acdc.csv")
    process_limbo(LIMBO_RSF, HERE / "input" / "clusters_limbo.csv")

    print("\nDone. Files ready for week4_hcag_hpc.py:")
    print(f"  {HERE / 'input' / 'clusters_acdc.csv'}")
    print(f"  {HERE / 'input' / 'clusters_limbo.csv'}")


if __name__ == "__main__":
    main()
