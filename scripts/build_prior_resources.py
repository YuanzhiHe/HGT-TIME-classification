#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import yaml


def build_default_config() -> dict[str, Any]:
    return {
        "prior": {
            "species_taxon_id": "9606",
            "canonical_gene_id": "gene_symbol",
            "filter_to_gene_universe": True,
            "gene_gene_score_threshold": 700,
            "gene_gene_score_sweep": [400, 700, 900],
            "gene_gene_weight_rule": "combined_score / 1000.0",
            "gene_pathway_weight_rule": "1 / sqrt(mapped_gene_count)",
        },
        "mapping": {
            "gene_universe_path": "outputs/scrna/gse161529__reference_v1/gse161529__reference_v1__step-08_gene_panel.tsv",
            "gene_symbol_column": "gene_symbol",
            "ensembl_gene_column": "gene_id",
            "allow_alias_fallback": True,
        },
        "string": {
            "version": "12.0",
            "links_path": "Experiment/datasets/string_v12/raw/9606.protein.links.v12.0.txt.gz",
            "aliases_path": "Experiment/datasets/string_v12/raw/9606.protein.aliases.v12.0.txt.gz",
            "info_path": "Experiment/datasets/string_v12/raw/9606.protein.info.v12.0.txt.gz",
            "download_page": "https://version-12-0.string-db.org/cgi/download",
        },
        "kegg": {
            "base_url": "https://rest.kegg.jp",
            "fetch_live_if_missing": True,
            "gene_catalog_path": "Experiment/datasets/kegg_hsa/raw/list_hsa.tsv",
            "pathway_list_path": "Experiment/datasets/kegg_hsa/raw/pathway_list_hsa.tsv",
            "pathway_links_path": "Experiment/datasets/kegg_hsa/raw/link_pathway_hsa.tsv",
            "pathway_catalog_path": "Experiment/datasets/kegg_hsa/raw/pathway_catalog_hsa.tsv",
            "include_top_level_classes": [
                "Cellular Processes",
                "Environmental Information Processing",
                "Genetic Information Processing",
                "Metabolism",
                "Organismal Systems",
            ],
            "exclude_pathway_names": [
                "Metabolic pathways",
                "Pathways in cancer",
                "MicroRNAs in cancer",
            ],
            "min_mapped_genes": 5,
            "max_mapped_genes": 300,
        },
        "output": {
            "root_dir": "outputs/priors/string_kegg_v1",
            "prefix": "string_kegg_v1",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build STRING/KEGG prior resources with unified gene mapping")
    parser.add_argument("--config", type=Path, help="Path to YAML config")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write the default config template and exit",
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def ensure_descendant(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Path escapes project root: {resolved}") from exc
    return resolved


def load_config(config_path: Path) -> dict[str, Any]:
    default_config = build_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_update(copy.deepcopy(default_config), user_config)
    config["_config_path"] = str(config_path.resolve())
    return config


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input file: {path}")
    return pd.read_csv(path, **kwargs)


def normalize_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", "", str(value).strip())
    return text.upper()


def normalize_ensembl_gene(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    if text.startswith("ENSG"):
        return text.split(".", 1)[0]
    return ""


def normalize_entrez_gene(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.fullmatch(r"(?:hsa:)?(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def join_values(values: set[str]) -> str:
    cleaned = sorted(str(value) for value in values if value and not (isinstance(value, float)))
    return "|".join(cleaned)


def pick_first(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def urlread(url: str) -> str:
    try:
        with urlopen(url) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Failed to fetch {url}: {exc}") from exc


def fetch_kegg_table(url: str, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = urlread(url)
    output_path.write_text(text, encoding="utf-8")
    return pd.read_csv(output_path, sep="\t", header=None, dtype=str)


def parse_kegg_pathway_get(text: str) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for block in text.split("///"):
        block = block.strip()
        if not block:
            continue
        entry_id = ""
        name = ""
        class_lines: list[str] = []
        current_key = ""
        for raw_line in block.splitlines():
            if not raw_line.strip():
                continue
            key = raw_line[:12].strip()
            value = raw_line[12:].strip()
            if key:
                current_key = key
            if current_key == "ENTRY" and not entry_id:
                entry_id = value.split()[0]
            elif current_key == "NAME" and not name:
                name = value.rstrip(";")
            elif current_key == "CLASS":
                class_lines.append(value.rstrip(";"))
        class_joined = "; ".join(class_lines)
        top_class = class_joined.split(";")[0].strip() if class_joined else ""
        subclass = "; ".join(part.strip() for part in class_joined.split(";")[1:]) if class_joined else ""
        records.append(
            {
                "pathway_id": f"path:{entry_id}" if entry_id and not entry_id.startswith("path:") else entry_id,
                "pathway_name": name,
                "kegg_class": class_joined,
                "pathway_top_class": top_class,
                "pathway_subclass": subclass,
            }
        )
    return pd.DataFrame.from_records(records)


def fetch_kegg_pathway_catalog(pathway_ids: list[str], base_url: str, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for start in range(0, len(pathway_ids), 10):
        chunk = pathway_ids[start : start + 10]
        query = "+".join(chunk)
        text = urlread(f"{base_url}/get/{query}")
        frame = parse_kegg_pathway_get(text)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged.to_csv(output_path, sep="\t", index=False)
    return merged


def ensure_kegg_raw_tables(config: dict[str, Any], project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_url = config["kegg"]["base_url"].rstrip("/")
    gene_catalog_path = ensure_descendant(resolve_path(project_root, config["kegg"]["gene_catalog_path"]), project_root)
    pathway_list_path = ensure_descendant(resolve_path(project_root, config["kegg"]["pathway_list_path"]), project_root)
    pathway_links_path = ensure_descendant(resolve_path(project_root, config["kegg"]["pathway_links_path"]), project_root)
    pathway_catalog_path = ensure_descendant(resolve_path(project_root, config["kegg"]["pathway_catalog_path"]), project_root)
    fetch_live = config["kegg"].get("fetch_live_if_missing", False)

    if gene_catalog_path is None or pathway_list_path is None or pathway_links_path is None or pathway_catalog_path is None:
        raise SystemExit("KEGG paths must be configured")

    if not gene_catalog_path.exists():
        if not fetch_live:
            raise SystemExit(f"Missing KEGG gene catalog: {gene_catalog_path}")
        fetch_kegg_table(f"{base_url}/list/hsa", gene_catalog_path)

    if not pathway_list_path.exists():
        if not fetch_live:
            raise SystemExit(f"Missing KEGG pathway list: {pathway_list_path}")
        fetch_kegg_table(f"{base_url}/list/pathway/hsa", pathway_list_path)

    if not pathway_links_path.exists():
        if not fetch_live:
            raise SystemExit(f"Missing KEGG pathway link table: {pathway_links_path}")
        fetch_kegg_table(f"{base_url}/link/pathway/hsa", pathway_links_path)

    pathway_list = pd.read_csv(pathway_list_path, sep="\t", header=None, names=["pathway_id", "pathway_name_raw"], dtype=str)

    refresh_pathway_catalog = not pathway_catalog_path.exists()
    if not refresh_pathway_catalog:
        existing_catalog = pd.read_csv(pathway_catalog_path, sep="\t", dtype=str)
        refresh_pathway_catalog = (
            "pathway_top_class" not in existing_catalog.columns
            or existing_catalog.get("pathway_top_class", pd.Series(dtype=str)).fillna("").eq("").all()
        )
    if refresh_pathway_catalog:
        if not fetch_live:
            raise SystemExit(f"Missing or invalid KEGG pathway catalog: {pathway_catalog_path}")
        fetch_kegg_pathway_catalog(pathway_list["pathway_id"].tolist(), base_url, pathway_catalog_path)

    gene_catalog = pd.read_csv(
        gene_catalog_path,
        sep="\t",
        header=None,
        names=["kegg_gene_id", "entry_type", "chromosome_count", "gene_definition"],
        usecols=[0, 1, 2, 3],
        dtype=str,
    )
    pathway_links = pd.read_csv(pathway_links_path, sep="\t", header=None, names=["kegg_gene_id", "pathway_id"], dtype=str)
    pathway_catalog = pd.read_csv(pathway_catalog_path, sep="\t", dtype=str)
    return gene_catalog, pathway_list, pathway_links, pathway_catalog


def load_gene_universe(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    path = resolve_path(project_root, config["mapping"].get("gene_universe_path"))
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "canonical_gene_symbol",
                "canonical_gene_symbol_norm",
                "gene_panel_gene_ids",
                "gene_panel_ensembl_gene_ids",
                "in_gene_universe",
            ]
        )
    path = ensure_descendant(path, project_root)

    frame = pd.read_csv(path, sep="\t", dtype=str)
    symbol_col = config["mapping"].get("gene_symbol_column", "gene_symbol")
    ensembl_col = config["mapping"].get("ensembl_gene_column", "gene_id")
    if symbol_col not in frame.columns:
        raise SystemExit(f"Configured gene symbol column {symbol_col!r} not found in {path}")
    if ensembl_col not in frame.columns:
        raise SystemExit(f"Configured Ensembl gene column {ensembl_col!r} not found in {path}")

    frame = frame.copy()
    frame["canonical_gene_symbol"] = frame[symbol_col].fillna("").astype(str)
    frame["canonical_gene_symbol_norm"] = frame["canonical_gene_symbol"].map(normalize_symbol)
    frame["gene_panel_ensembl_gene_id"] = frame[ensembl_col].map(normalize_ensembl_gene)
    frame["gene_panel_gene_id"] = frame[ensembl_col].fillna("").astype(str)
    frame = frame[frame["canonical_gene_symbol_norm"] != ""]
    grouped = (
        frame.groupby(["canonical_gene_symbol_norm", "canonical_gene_symbol"], as_index=False)
        .agg(
            gene_panel_gene_ids=("gene_panel_gene_id", lambda series: join_values(set(series.astype(str)))),
            gene_panel_ensembl_gene_ids=("gene_panel_ensembl_gene_id", lambda series: join_values(set(series.astype(str)))),
        )
        .sort_values(["canonical_gene_symbol_norm", "canonical_gene_symbol"])
        .drop_duplicates(subset=["canonical_gene_symbol_norm"], keep="first")
        .reset_index(drop=True)
    )
    grouped["in_gene_universe"] = True
    return grouped


def load_string_sources(config: dict[str, Any], project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links_path = ensure_descendant(resolve_path(project_root, config["string"]["links_path"]), project_root)
    aliases_path = ensure_descendant(resolve_path(project_root, config["string"]["aliases_path"]), project_root)
    info_path = ensure_descendant(resolve_path(project_root, config["string"]["info_path"]), project_root)
    if links_path is None or aliases_path is None or info_path is None:
        raise SystemExit("STRING links/aliases/info paths must be configured")

    links = read_table(
        links_path,
        sep=r"\s+",
        engine="python",
        compression="infer",
        dtype=str,
    )
    aliases = read_table(
        aliases_path,
        sep="\t",
        comment="#",
        names=["string_protein_id", "alias", "alias_source"],
        header=None,
        compression="infer",
        dtype=str,
    )
    info = read_table(
        info_path,
        sep="\t",
        comment="#",
        names=["string_protein_id", "preferred_name", "protein_size", "annotation"],
        header=None,
        compression="infer",
        dtype=str,
    )
    return links, aliases, info


def parse_kegg_gene_catalog(gene_catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for row in gene_catalog.itertuples(index=False):
        definition = getattr(row, "gene_definition", "") or ""
        symbol_part, _, description = definition.partition(";")
        symbol_tokens = [token.strip() for token in symbol_part.split(",") if token.strip()]
        primary_symbol = symbol_tokens[0] if symbol_tokens else ""
        alias_symbols = "|".join(symbol_tokens[1:]) if len(symbol_tokens) > 1 else ""
        kegg_gene_id = getattr(row, "kegg_gene_id", "")
        records.append(
            {
                "kegg_gene_id": kegg_gene_id,
                "entrez_gene_id": normalize_entrez_gene(kegg_gene_id),
                "kegg_primary_symbol": primary_symbol,
                "kegg_primary_symbol_norm": normalize_symbol(primary_symbol),
                "kegg_alias_symbols": alias_symbols,
                "gene_description": description.strip(),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_kegg_pathway_list(pathway_list: pd.DataFrame) -> pd.DataFrame:
    parsed = pathway_list.copy()
    parsed["pathway_id"] = parsed["pathway_id"].fillna("").astype(str).map(
        lambda value: value if value.startswith("path:") else f"path:{value}"
    )
    parsed["pathway_name"] = (
        parsed["pathway_name_raw"]
        .fillna("")
        .astype(str)
        .str.replace(r" - Homo sapiens \(human\)$", "", regex=True)
    )
    return parsed[["pathway_id", "pathway_name"]]


def build_string_protein_bridge(info: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for row in info.itertuples(index=False):
        protein_id = getattr(row, "string_protein_id")
        preferred_name = getattr(row, "preferred_name", "") or ""
        records[protein_id] = {
            "string_protein_id": protein_id,
            "string_preferred_name": preferred_name,
            "string_preferred_name_norm": normalize_symbol(preferred_name),
            "string_ensembl_gene_ids": set(),
            "string_entrez_gene_ids": set(),
            "string_alias_symbols": set(),
            "string_alias_symbols_norm": set(),
            "string_alias_sources": set(),
        }

    for row in aliases.itertuples(index=False):
        protein_id = getattr(row, "string_protein_id")
        alias = getattr(row, "alias", "") or ""
        source = getattr(row, "alias_source", "") or ""
        record = records.setdefault(
            protein_id,
            {
                "string_protein_id": protein_id,
                "string_preferred_name": "",
                "string_preferred_name_norm": "",
                "string_ensembl_gene_ids": set(),
                "string_entrez_gene_ids": set(),
                "string_alias_symbols": set(),
                "string_alias_symbols_norm": set(),
                "string_alias_sources": set(),
            },
        )
        record["string_alias_sources"].add(source)
        ensembl_gene = normalize_ensembl_gene(alias)
        entrez_gene = normalize_entrez_gene(alias)
        if ensembl_gene:
            record["string_ensembl_gene_ids"].add(ensembl_gene)
            continue
        if entrez_gene and ("Entrez" in source or "BioMart" in source or source == "BLAST_UniProt_AC"):
            record["string_entrez_gene_ids"].add(entrez_gene)
            continue
        alias_norm = normalize_symbol(alias)
        if alias_norm:
            record["string_alias_symbols"].add(str(alias).strip())
            record["string_alias_symbols_norm"].add(alias_norm)

    rows: list[dict[str, str]] = []
    for protein_id, record in sorted(records.items()):
        rows.append(
            {
                "string_protein_id": protein_id,
                "string_preferred_name": record["string_preferred_name"],
                "string_preferred_name_norm": record["string_preferred_name_norm"],
                "string_ensembl_gene_ids": join_values(record["string_ensembl_gene_ids"]),
                "string_entrez_gene_ids": join_values(record["string_entrez_gene_ids"]),
                "string_alias_symbols": join_values(record["string_alias_symbols"]),
                "string_alias_symbols_norm": join_values(record["string_alias_symbols_norm"]),
                "string_alias_sources": join_values(record["string_alias_sources"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_gene_master(
    gene_universe: pd.DataFrame,
    kegg_genes: pd.DataFrame,
    string_bridge: pd.DataFrame,
    allow_alias_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master: dict[str, dict[str, Any]] = {}
    ensembl_to_key: dict[str, str] = {}
    entrez_to_key: dict[str, str] = {}
    symbol_to_key: dict[str, str] = {}
    kegg_to_key: dict[str, str] = {}
    protein_to_key: dict[str, tuple[str, str]] = {}

    def ensure_record(key: str, symbol_hint: str) -> dict[str, Any]:
        record = master.get(key)
        if record is None:
            record = {
                "canonical_gene_symbol": symbol_hint or key,
                "canonical_gene_symbol_norm": key,
                "gene_panel_gene_ids": set(),
                "gene_panel_ensembl_gene_ids": set(),
                "string_protein_ids": set(),
                "string_preferred_names": set(),
                "string_ensembl_gene_ids": set(),
                "string_entrez_gene_ids": set(),
                "kegg_gene_ids": set(),
                "kegg_entrez_gene_ids": set(),
                "kegg_primary_symbols": set(),
                "alias_symbols": set(),
                "mapping_sources": set(),
                "in_gene_universe": False,
            }
            master[key] = record
        elif symbol_hint and not record["canonical_gene_symbol"]:
            record["canonical_gene_symbol"] = symbol_hint
        symbol_to_key[key] = key
        return record

    for row in gene_universe.itertuples(index=False):
        key = getattr(row, "canonical_gene_symbol_norm")
        symbol = getattr(row, "canonical_gene_symbol")
        record = ensure_record(key, symbol)
        record["gene_panel_gene_ids"].update(filter(None, str(getattr(row, "gene_panel_gene_ids", "")).split("|")))
        record["gene_panel_ensembl_gene_ids"].update(filter(None, str(getattr(row, "gene_panel_ensembl_gene_ids", "")).split("|")))
        record["mapping_sources"].add("gene_panel")
        record["in_gene_universe"] = True
        for ensembl in record["gene_panel_ensembl_gene_ids"]:
            ensembl_to_key[ensembl] = key

    for row in kegg_genes.itertuples(index=False):
        primary_norm = getattr(row, "kegg_primary_symbol_norm")
        primary_symbol = getattr(row, "kegg_primary_symbol")
        entrez = getattr(row, "entrez_gene_id")
        if entrez and entrez in entrez_to_key:
            key = entrez_to_key[entrez]
        elif primary_norm and primary_norm in symbol_to_key:
            key = symbol_to_key[primary_norm]
        else:
            key = primary_norm or normalize_symbol(primary_symbol)
        if not key:
            continue
        record = ensure_record(key, primary_symbol)
        record["kegg_gene_ids"].add(getattr(row, "kegg_gene_id"))
        if entrez:
            record["kegg_entrez_gene_ids"].add(entrez)
            entrez_to_key[entrez] = key
        if primary_symbol:
            record["kegg_primary_symbols"].add(primary_symbol)
            record["alias_symbols"].add(primary_symbol)
        alias_symbols = getattr(row, "kegg_alias_symbols", "")
        if alias_symbols:
            for alias in str(alias_symbols).split("|"):
                if alias:
                    record["alias_symbols"].add(alias)
                    alias_norm = normalize_symbol(alias)
                    if alias_norm and alias_norm not in symbol_to_key:
                        symbol_to_key[alias_norm] = key
        record["mapping_sources"].add("kegg")
        kegg_to_key[getattr(row, "kegg_gene_id")] = key

    def choose_mapping_key(row: pd.Series) -> tuple[str, str]:
        candidate_ensembl = [value for value in str(row["string_ensembl_gene_ids"]).split("|") if value]
        candidate_entrez = [value for value in str(row["string_entrez_gene_ids"]).split("|") if value]
        alias_symbols = [value for value in str(row["string_alias_symbols_norm"]).split("|") if value]
        preferred_norm = str(row["string_preferred_name_norm"])

        ensembl_hits = [ensembl_to_key[value] for value in candidate_ensembl if value in ensembl_to_key]
        if len(set(ensembl_hits)) == 1:
            return ensembl_hits[0], "string_ensembl"

        entrez_hits = [entrez_to_key[value] for value in candidate_entrez if value in entrez_to_key]
        if len(set(entrez_hits)) == 1:
            return entrez_hits[0], "string_entrez"

        if preferred_norm and preferred_norm in symbol_to_key:
            return symbol_to_key[preferred_norm], "preferred_symbol"

        if allow_alias_fallback:
            alias_hits = [symbol_to_key[value] for value in alias_symbols if value in symbol_to_key]
            if len(set(alias_hits)) == 1:
                return alias_hits[0], "alias_symbol"

        fallback_key = preferred_norm or pick_first(alias_symbols)
        return fallback_key, "string_preferred_fallback"

    string_bridge = string_bridge.copy()
    string_bridge["canonical_gene_symbol"] = ""
    string_bridge["canonical_gene_symbol_norm"] = ""
    string_bridge["mapping_rule"] = ""

    for idx, row in string_bridge.iterrows():
        key, mapping_rule = choose_mapping_key(row)
        if not key:
            continue
        symbol_hint = row["string_preferred_name"] or key
        record = ensure_record(key, symbol_hint)
        record["string_protein_ids"].add(row["string_protein_id"])
        record["string_preferred_names"].add(row["string_preferred_name"])
        record["mapping_sources"].add("string")
        for ensembl in filter(None, str(row["string_ensembl_gene_ids"]).split("|")):
            record["string_ensembl_gene_ids"].add(ensembl)
            if ensembl not in ensembl_to_key:
                ensembl_to_key[ensembl] = key
        for entrez in filter(None, str(row["string_entrez_gene_ids"]).split("|")):
            record["string_entrez_gene_ids"].add(entrez)
            if entrez not in entrez_to_key:
                entrez_to_key[entrez] = key
        for alias in filter(None, str(row["string_alias_symbols"]).split("|")):
            record["alias_symbols"].add(alias)
            alias_norm = normalize_symbol(alias)
            if alias_norm and alias_norm not in symbol_to_key:
                symbol_to_key[alias_norm] = key
        if row["string_preferred_name_norm"] and row["string_preferred_name_norm"] not in symbol_to_key:
            symbol_to_key[row["string_preferred_name_norm"]] = key
        string_bridge.at[idx, "canonical_gene_symbol"] = record["canonical_gene_symbol"]
        string_bridge.at[idx, "canonical_gene_symbol_norm"] = key
        string_bridge.at[idx, "mapping_rule"] = mapping_rule
        protein_to_key[row["string_protein_id"]] = (key, mapping_rule)

    rows: list[dict[str, Any]] = []
    for key, record in sorted(master.items()):
        rows.append(
            {
                "canonical_gene_symbol": record["canonical_gene_symbol"],
                "canonical_gene_symbol_norm": key,
                "gene_panel_gene_ids": join_values(record["gene_panel_gene_ids"]),
                "gene_panel_ensembl_gene_ids": join_values(record["gene_panel_ensembl_gene_ids"]),
                "string_protein_ids": join_values(record["string_protein_ids"]),
                "string_preferred_names": join_values(record["string_preferred_names"]),
                "string_ensembl_gene_ids": join_values(record["string_ensembl_gene_ids"]),
                "string_entrez_gene_ids": join_values(record["string_entrez_gene_ids"]),
                "kegg_gene_ids": join_values(record["kegg_gene_ids"]),
                "kegg_entrez_gene_ids": join_values(record["kegg_entrez_gene_ids"]),
                "kegg_primary_symbols": join_values(record["kegg_primary_symbols"]),
                "alias_symbols": join_values(record["alias_symbols"]),
                "mapping_sources": join_values(record["mapping_sources"]),
                "in_gene_universe": bool(record["in_gene_universe"]),
            }
        )
    return pd.DataFrame.from_records(rows), string_bridge


def build_gene_gene_edges(
    links: pd.DataFrame,
    string_bridge: pd.DataFrame,
    config: dict[str, Any],
    gene_master: pd.DataFrame,
) -> pd.DataFrame:
    bridge_lookup = (
        string_bridge[["string_protein_id", "canonical_gene_symbol", "canonical_gene_symbol_norm"]]
        .drop_duplicates()
        .set_index("string_protein_id")
        .to_dict(orient="index")
    )
    threshold = int(config["prior"]["gene_gene_score_threshold"])
    filter_to_universe = bool(config["prior"].get("filter_to_gene_universe", True))
    universe = set(
        gene_master.loc[gene_master["in_gene_universe"].astype(bool), "canonical_gene_symbol_norm"].astype(str)
    )

    score_column = "combined_score" if "combined_score" in links.columns else links.columns[-1]
    rows: list[dict[str, Any]] = []
    for row in links.itertuples(index=False):
        protein_a = getattr(row, links.columns[0])
        protein_b = getattr(row, links.columns[1])
        score_raw = getattr(row, score_column)
        if protein_a not in bridge_lookup or protein_b not in bridge_lookup:
            continue
        gene_a = bridge_lookup[protein_a]["canonical_gene_symbol_norm"]
        gene_b = bridge_lookup[protein_b]["canonical_gene_symbol_norm"]
        if not gene_a or not gene_b or gene_a == gene_b:
            continue
        if filter_to_universe and (gene_a not in universe or gene_b not in universe):
            continue
        source_symbol = bridge_lookup[protein_a]["canonical_gene_symbol"]
        target_symbol = bridge_lookup[protein_b]["canonical_gene_symbol"]
        pair = sorted(
            [
                (gene_a, source_symbol, protein_a),
                (gene_b, target_symbol, protein_b),
            ],
            key=lambda item: item[0],
        )
        score = int(str(score_raw))
        rows.append(
            {
                "source_gene_symbol": pair[0][1],
                "source_gene_symbol_norm": pair[0][0],
                "target_gene_symbol": pair[1][1],
                "target_gene_symbol_norm": pair[1][0],
                "source_string_protein_id": pair[0][2],
                "target_string_protein_id": pair[1][2],
                "combined_score": score,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_gene_symbol",
                "target_gene_symbol",
                "support_edge_count",
                "combined_score_max",
                "combined_score_mean",
                "edge_weight",
                "edge_weight_rule",
                "string_threshold_used",
                "retained_for_graph",
            ]
        )

    frame = pd.DataFrame.from_records(rows)
    aggregated = (
        frame.groupby(
            [
                "source_gene_symbol",
                "source_gene_symbol_norm",
                "target_gene_symbol",
                "target_gene_symbol_norm",
            ],
            as_index=False,
        )
        .agg(
            support_edge_count=("combined_score", "count"),
            combined_score_max=("combined_score", "max"),
            combined_score_mean=("combined_score", "mean"),
        )
        .sort_values(["combined_score_max", "support_edge_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    aggregated["edge_weight"] = aggregated["combined_score_max"] / 1000.0
    aggregated["edge_weight_rule"] = config["prior"]["gene_gene_weight_rule"]
    aggregated["string_threshold_used"] = threshold
    aggregated["retained_for_graph"] = aggregated["combined_score_max"] >= threshold
    aggregated = aggregated[aggregated["retained_for_graph"]].reset_index(drop=True)
    return aggregated


def build_gene_pathway_edges(
    pathway_links: pd.DataFrame,
    pathway_list: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
    gene_master: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kegg_lookup: dict[str, tuple[str, str, bool]] = {}
    for row in gene_master.itertuples(index=False):
        kegg_ids = [value for value in str(getattr(row, "kegg_gene_ids", "")).split("|") if value]
        for kegg_id in kegg_ids:
            kegg_lookup[kegg_id] = (
                getattr(row, "canonical_gene_symbol"),
                getattr(row, "canonical_gene_symbol_norm"),
                bool(getattr(row, "in_gene_universe")),
            )

    pathway_links = pathway_links.copy()
    pathway_links["pathway_id"] = pathway_links["pathway_id"].fillna("").astype(str).map(
        lambda value: value if value.startswith("path:") else f"path:{value}"
    )
    pathway_catalog = pathway_catalog.copy()
    pathway_catalog["pathway_id"] = pathway_catalog["pathway_id"].fillna("").astype(str).map(
        lambda value: value if value.startswith("path:") else f"path:{value}"
    )

    pathways = pathway_list.merge(pathway_catalog, on="pathway_id", how="left")
    if "pathway_name_x" in pathways.columns:
        pathways["pathway_name"] = pathways["pathway_name_x"].fillna(pathways["pathway_name_y"])
        pathways = pathways.drop(columns=[column for column in ["pathway_name_x", "pathway_name_y"] if column in pathways.columns])

    include_classes = set(config["kegg"]["include_top_level_classes"])
    exclude_names = set(config["kegg"]["exclude_pathway_names"])
    min_genes = int(config["kegg"]["min_mapped_genes"])
    max_genes = int(config["kegg"]["max_mapped_genes"])
    filter_to_universe = bool(config["prior"].get("filter_to_gene_universe", True))

    rows: list[dict[str, Any]] = []
    for row in pathway_links.itertuples(index=False):
        kegg_gene_id = getattr(row, "kegg_gene_id")
        pathway_id = getattr(row, "pathway_id")
        if kegg_gene_id not in kegg_lookup:
            continue
        gene_symbol, gene_norm, in_universe = kegg_lookup[kegg_gene_id]
        if filter_to_universe and not in_universe:
            continue
        rows.append(
            {
                "pathway_id": pathway_id,
                "kegg_gene_id": kegg_gene_id,
                "gene_symbol": gene_symbol,
                "gene_symbol_norm": gene_norm,
            }
        )

    edges = pd.DataFrame.from_records(rows)
    if edges.empty:
        pathways["total_mapped_genes"] = 0
        pathways["selected_for_graph"] = False
        pathways["selection_reason"] = "no_mapped_genes"
        return pathways, pd.DataFrame(
            columns=[
                "gene_symbol",
                "pathway_id",
                "pathway_name",
                "pathway_top_class",
                "pathway_subclass",
                "mapped_gene_count",
                "edge_weight",
                "edge_weight_rule",
            ]
        )

    mapped_counts = edges.groupby("pathway_id")["gene_symbol_norm"].nunique().rename("total_mapped_genes").reset_index()
    pathways = pathways.merge(mapped_counts, on="pathway_id", how="left")
    pathways["total_mapped_genes"] = pathways["total_mapped_genes"].fillna(0).astype(int)
    pathways["selected_for_graph"] = (
        pathways["pathway_top_class"].isin(include_classes)
        & ~pathways["pathway_name"].isin(exclude_names)
        & pathways["total_mapped_genes"].between(min_genes, max_genes)
    )

    selection_reason: list[str] = []
    for row in pathways.itertuples(index=False):
        reason = "selected"
        if getattr(row, "pathway_top_class", "") not in include_classes:
            reason = "excluded_class"
        elif getattr(row, "pathway_name", "") in exclude_names:
            reason = "excluded_name"
        elif getattr(row, "total_mapped_genes", 0) < min_genes:
            reason = "below_min_gene_count"
        elif getattr(row, "total_mapped_genes", 0) > max_genes:
            reason = "above_max_gene_count"
        selection_reason.append(reason)
    pathways["selection_reason"] = selection_reason

    selected_pathways = pathways.loc[pathways["selected_for_graph"], ["pathway_id", "pathway_name", "pathway_top_class", "pathway_subclass", "total_mapped_genes"]]
    edges = edges.merge(selected_pathways, on="pathway_id", how="inner")
    edges = edges.drop_duplicates(subset=["gene_symbol_norm", "pathway_id"]).reset_index(drop=True)
    edges["mapped_gene_count"] = edges["total_mapped_genes"]
    edges["edge_weight"] = edges["mapped_gene_count"].map(lambda value: round(1.0 / math.sqrt(value), 6) if value else 0.0)
    edges["edge_weight_rule"] = config["prior"]["gene_pathway_weight_rule"]
    edges = edges[
        [
            "gene_symbol",
            "gene_symbol_norm",
            "kegg_gene_id",
            "pathway_id",
            "pathway_name",
            "pathway_top_class",
            "pathway_subclass",
            "mapped_gene_count",
            "edge_weight",
            "edge_weight_rule",
        ]
    ].sort_values(["pathway_id", "gene_symbol_norm"]).reset_index(drop=True)
    return pathways, edges


def build_manifest(
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    gene_master: pd.DataFrame,
    gene_gene_edges: pd.DataFrame,
    gene_pathway_edges: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "builder": "build_prior_resources.py",
        "config_path": config.get("_config_path"),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "sources": {
            "string_version": config["string"]["version"],
            "string_download_page": config["string"]["download_page"],
            "kegg_base_url": config["kegg"]["base_url"],
        },
        "mapping_rules": [
            "Primary canonical ID is gene symbol.",
            "STRING proteins map to genes by Ensembl gene ID first, Entrez second, preferred symbol third, alias fallback last.",
            "KEGG genes map by Entrez and primary symbol; graph outputs optionally filter to the gene universe from the scRNA gene panel.",
        ],
        "thresholds": {
            "string_score_threshold": config["prior"]["gene_gene_score_threshold"],
            "string_score_sweep": config["prior"]["gene_gene_score_sweep"],
            "kegg_min_mapped_genes": config["kegg"]["min_mapped_genes"],
            "kegg_max_mapped_genes": config["kegg"]["max_mapped_genes"],
            "included_pathway_classes": config["kegg"]["include_top_level_classes"],
            "excluded_pathway_names": config["kegg"]["exclude_pathway_names"],
        },
        "counts": {
            "genes_in_master_table": int(len(gene_master)),
            "genes_in_universe": int(gene_master["in_gene_universe"].astype(bool).sum()) if not gene_master.empty else 0,
            "gene_gene_edges_retained": int(len(gene_gene_edges)),
            "gene_pathway_edges_retained": int(len(gene_pathway_edges)),
            "pathways_selected": int(pathway_catalog["selected_for_graph"].astype(bool).sum()) if "selected_for_graph" in pathway_catalog.columns else 0,
        },
        "edge_weight_rules": {
            "gene_gene": config["prior"]["gene_gene_weight_rule"],
            "gene_pathway": config["prior"]["gene_pathway_weight_rule"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        write_yaml(args.write_default_config, build_default_config())
        return

    if not args.config:
        raise SystemExit("Use --config or --write-default-config")

    config = load_config(args.config)
    project_root = discover_project_root(args.config)
    output_dir = resolve_path(project_root, config["output"]["root_dir"])
    if output_dir is None:
        raise SystemExit("output.root_dir must be set")
    output_dir = ensure_descendant(output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config["output"]["prefix"]

    gene_universe = load_gene_universe(config, project_root)
    links, aliases, info = load_string_sources(config, project_root)
    kegg_gene_catalog_raw, kegg_pathway_list_raw, kegg_pathway_links, kegg_pathway_catalog = ensure_kegg_raw_tables(config, project_root)

    kegg_genes = parse_kegg_gene_catalog(kegg_gene_catalog_raw)
    kegg_pathway_list = parse_kegg_pathway_list(kegg_pathway_list_raw)
    string_bridge = build_string_protein_bridge(info, aliases)
    gene_master, string_bridge = build_gene_master(
        gene_universe=gene_universe,
        kegg_genes=kegg_genes,
        string_bridge=string_bridge,
        allow_alias_fallback=bool(config["mapping"].get("allow_alias_fallback", True)),
    )
    pathway_catalog, gene_pathway_edges = build_gene_pathway_edges(
        pathway_links=kegg_pathway_links,
        pathway_list=kegg_pathway_list,
        pathway_catalog=kegg_pathway_catalog,
        gene_master=gene_master,
        config=config,
    )
    gene_gene_edges = build_gene_gene_edges(
        links=links,
        string_bridge=string_bridge,
        config=config,
        gene_master=gene_master,
    )

    manifest = build_manifest(
        config=config,
        project_root=project_root,
        output_dir=output_dir,
        gene_master=gene_master,
        gene_gene_edges=gene_gene_edges,
        gene_pathway_edges=gene_pathway_edges,
        pathway_catalog=pathway_catalog,
    )

    write_table(gene_universe, output_dir / f"{prefix}__step-01_gene_universe.tsv.gz")
    write_table(string_bridge, output_dir / f"{prefix}__step-02_string_protein_bridge.tsv.gz")
    write_table(pathway_catalog, output_dir / f"{prefix}__step-03_pathway_catalog.tsv.gz")
    write_table(gene_master, output_dir / f"{prefix}__step-04_gene_master.tsv.gz")
    write_table(gene_gene_edges, output_dir / f"{prefix}__step-05_gene_gene_edges.tsv.gz")
    write_table(gene_pathway_edges, output_dir / f"{prefix}__step-06_gene_pathway_edges.tsv.gz")
    write_json(output_dir / f"{prefix}__step-07_manifest.json", manifest)


if __name__ == "__main__":
    main()
