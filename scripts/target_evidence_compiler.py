#!/usr/bin/env python3
"""Candidate target biological evidence compiler and functional validation recommender.

Consumes the upstream interpretability outputs (gene_ranking.tsv, pathway_ranking.tsv,
pathway_enrichment.tsv, subtype-specific rankings) and compiles:

1. Biological evidence profiles for each candidate target, including:
   - Network role annotation (graph topology context)
   - Known immune function lookup
   - Subtype specificity assessment
   - Pathway membership and enrichment context
   - Literature support level classification

2. Functional validation pathway recommendations:
   - Stage 1: In silico re-validation (TISCH, TIDE, KM-plotter)
   - Stage 2: In vitro co-culture assays
   - Stage 3: In vivo syngeneic models

Outputs the final candidate_target_priority.tsv conforming to the project template.

Usage:
    python target_evidence_compiler.py \\
        --interpretability-dir outputs/results/EXP-M01-HGT/interpretability \\
        --output-dir outputs/results/EXP-M01-HGT/target_evidence \\
        --topk 50

    # With external pathway annotations:
    python target_evidence_compiler.py \\
        --interpretability-dir outputs/results/EXP-M01-HGT/interpretability \\
        --pathway-gmt resources/kegg_immune_pathways.gmt \\
        --topk 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known immune gene annotation database (built-in)
# ---------------------------------------------------------------------------

# Each entry: gene_symbol -> {category, function, literature_level, cell_types,
#                              time_relevance, ici_evidence}
IMMUNE_GENE_DATABASE: dict[str, dict[str, str]] = {
    # --- Immune checkpoints ---
    "CD274": {
        "category": "immune_checkpoint",
        "function": "PD-L1; primary ligand of PD-1; suppresses T-cell activation",
        "literature_level": "extensive",
        "cell_types": "tumor cells, macrophages, DCs",
        "time_relevance": "Hot: high expression correlates with inflamed phenotype; target of anti-PD-L1 therapy",
        "ici_evidence": "known predictive biomarker for anti-PD-1/PD-L1",
    },
    "PDCD1": {
        "category": "immune_checkpoint",
        "function": "PD-1; inhibitory receptor on exhausted T cells",
        "literature_level": "extensive",
        "cell_types": "exhausted CD8+ T cells, regulatory T cells",
        "time_relevance": "Hot: marks T-cell exhaustion; primary ICI target",
        "ici_evidence": "direct therapeutic target (nivolumab, pembrolizumab)",
    },
    "CTLA4": {
        "category": "immune_checkpoint",
        "function": "Competitive inhibitor of CD28 co-stimulation",
        "literature_level": "extensive",
        "cell_types": "activated T cells, regulatory T cells",
        "time_relevance": "Hot: regulates T-cell priming; synergizes with anti-PD-1",
        "ici_evidence": "direct therapeutic target (ipilimumab)",
    },
    "LAG3": {
        "category": "immune_checkpoint",
        "function": "Inhibitory receptor; binds MHC-II and FGL1",
        "literature_level": "strong",
        "cell_types": "exhausted T cells, NK cells",
        "time_relevance": "Hot/Excluded: co-expressed with PD-1 in exhaustion",
        "ici_evidence": "approved combination target (relatlimab + nivolumab)",
    },
    "HAVCR2": {
        "category": "immune_checkpoint",
        "function": "TIM-3; marks terminally exhausted T cells",
        "literature_level": "strong",
        "cell_types": "exhausted CD8+ T cells, DCs, macrophages",
        "time_relevance": "Hot: terminal exhaustion marker",
        "ici_evidence": "clinical trials ongoing (sabatolimab)",
    },
    "TIGIT": {
        "category": "immune_checkpoint",
        "function": "Inhibitory receptor competing with CD226 for PVR binding",
        "literature_level": "strong",
        "cell_types": "T cells, NK cells",
        "time_relevance": "Hot: co-inhibitory axis with PD-1",
        "ici_evidence": "clinical trials (tiragolumab); mixed phase III results",
    },
    "PDCD1LG2": {
        "category": "immune_checkpoint",
        "function": "PD-L2; second ligand of PD-1",
        "literature_level": "moderate",
        "cell_types": "DCs, macrophages, tumor cells",
        "time_relevance": "Hot: alternative PD-1 engagement pathway",
        "ici_evidence": "associated with anti-PD-1 response in some cohorts",
    },
    "VSIR": {
        "category": "immune_checkpoint",
        "function": "VISTA; suppresses early T-cell activation",
        "literature_level": "moderate",
        "cell_types": "myeloid cells, tumor cells",
        "time_relevance": "Excluded/Cold: suppresses immune initiation",
        "ici_evidence": "emerging target; phase I trials",
    },
    "CD80": {
        "category": "co_stimulation",
        "function": "B7-1; co-stimulatory ligand for CD28 and CTLA-4",
        "literature_level": "extensive",
        "cell_types": "DCs, macrophages, B cells",
        "time_relevance": "Hot: antigen presentation co-stimulation",
        "ici_evidence": "indirect relevance via CTLA-4 pathway",
    },
    "CD86": {
        "category": "co_stimulation",
        "function": "B7-2; primary co-stimulatory ligand for CD28",
        "literature_level": "extensive",
        "cell_types": "DCs, macrophages, B cells",
        "time_relevance": "Hot: essential for T-cell priming",
        "ici_evidence": "indirect relevance via CTLA-4 pathway",
    },
    "IDO1": {
        "category": "metabolic_checkpoint",
        "function": "Indoleamine 2,3-dioxygenase; tryptophan catabolism; immunosuppressive",
        "literature_level": "strong",
        "cell_types": "tumor cells, DCs, macrophages",
        "time_relevance": "Excluded: metabolic immune suppression",
        "ici_evidence": "failed as monotherapy (epacadostat); combination interest persists",
    },
    "SIGLEC15": {
        "category": "immune_checkpoint",
        "function": "Macrophage-expressed inhibitory receptor; suppresses T-cell activity",
        "literature_level": "moderate",
        "cell_types": "macrophages, tumor cells",
        "time_relevance": "Excluded/Cold: myeloid-mediated suppression",
        "ici_evidence": "phase I/II trials (NC318)",
    },
    # --- IFN-gamma pathway ---
    "IFNG": {
        "category": "ifn_gamma_pathway",
        "function": "Interferon-gamma; master cytokine for anti-tumor immunity",
        "literature_level": "extensive",
        "cell_types": "CD8+ T cells, NK cells, CD4+ Th1 cells",
        "time_relevance": "Hot: signature cytokine of inflamed TIME",
        "ici_evidence": "IFN-gamma signature predicts ICI response",
    },
    "STAT1": {
        "category": "ifn_gamma_pathway",
        "function": "Signal transducer for IFN-gamma signaling",
        "literature_level": "extensive",
        "cell_types": "broadly expressed; upregulated in immune-active tumors",
        "time_relevance": "Hot: downstream IFN-gamma effector",
        "ici_evidence": "component of IFN-gamma response signature",
    },
    "IRF1": {
        "category": "ifn_gamma_pathway",
        "function": "Interferon regulatory factor 1; transcription factor for PD-L1",
        "literature_level": "strong",
        "cell_types": "tumor cells, immune cells",
        "time_relevance": "Hot: links IFN-gamma to PD-L1 upregulation",
        "ici_evidence": "mediates adaptive immune resistance",
    },
    "CXCL9": {
        "category": "chemokine",
        "function": "T-cell chemoattractant via CXCR3; IFN-gamma-induced",
        "literature_level": "extensive",
        "cell_types": "macrophages, DCs, endothelial cells",
        "time_relevance": "Hot: recruits effector T cells into tumor",
        "ici_evidence": "positive correlation with ICI response",
    },
    "CXCL10": {
        "category": "chemokine",
        "function": "IP-10; CXCR3 ligand; T-cell and NK cell recruitment",
        "literature_level": "extensive",
        "cell_types": "macrophages, endothelial cells, fibroblasts",
        "time_relevance": "Hot: T-cell trafficking into tumor",
        "ici_evidence": "component of immunotherapy response signatures",
    },
    "CXCL11": {
        "category": "chemokine",
        "function": "I-TAC; CXCR3 ligand; T-cell chemoattractant",
        "literature_level": "moderate",
        "cell_types": "monocytes, endothelial cells",
        "time_relevance": "Hot: T-cell infiltration signal",
        "ici_evidence": "part of chemokine response axis",
    },
    "GBP1": {
        "category": "ifn_gamma_pathway",
        "function": "Guanylate-binding protein 1; IFN-gamma effector",
        "literature_level": "moderate",
        "cell_types": "macrophages, endothelial cells",
        "time_relevance": "Hot: IFN-gamma effector",
        "ici_evidence": "component of IFN response score",
    },
    # --- Antigen presentation ---
    "HLA-A": {
        "category": "antigen_presentation",
        "function": "MHC class I heavy chain; neoantigen presentation",
        "literature_level": "extensive",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot: essential for CD8+ T-cell recognition; loss → immune evasion",
        "ici_evidence": "HLA LOH predicts ICI resistance",
    },
    "HLA-B": {
        "category": "antigen_presentation",
        "function": "MHC class I heavy chain",
        "literature_level": "extensive",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot: neoantigen presentation",
        "ici_evidence": "HLA diversity associated with ICI benefit",
    },
    "HLA-C": {
        "category": "antigen_presentation",
        "function": "MHC class I heavy chain; NK cell ligand",
        "literature_level": "strong",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot: NK cell regulation + antigen presentation",
        "ici_evidence": "component of antigen presentation machinery",
    },
    "B2M": {
        "category": "antigen_presentation",
        "function": "Beta-2-microglobulin; MHC-I light chain; essential for surface display",
        "literature_level": "extensive",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot/Excluded: loss causes MHC-I downregulation → immune escape",
        "ici_evidence": "B2M loss = primary resistance to anti-PD-1",
    },
    "TAP1": {
        "category": "antigen_presentation",
        "function": "Transporter for antigen processing; peptide loading onto MHC-I",
        "literature_level": "strong",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot: deficiency impairs neoantigen presentation",
        "ici_evidence": "TAP deficiency associated with immune evasion",
    },
    "TAP2": {
        "category": "antigen_presentation",
        "function": "TAP complex subunit; peptide transport to ER",
        "literature_level": "strong",
        "cell_types": "all nucleated cells",
        "time_relevance": "Hot: essential for antigen processing pathway",
        "ici_evidence": "component of antigen presentation machinery",
    },
    "PSMB9": {
        "category": "antigen_presentation",
        "function": "Immunoproteasome subunit; enhances peptide generation for MHC-I",
        "literature_level": "moderate",
        "cell_types": "broadly expressed; IFN-gamma-induced",
        "time_relevance": "Hot: immunoproteasome component",
        "ici_evidence": "part of IFN-gamma response; loss impairs antigen presentation",
    },
    # --- T cell markers ---
    "CD3D": {
        "category": "t_cell_marker",
        "function": "T-cell receptor complex component",
        "literature_level": "extensive",
        "cell_types": "all T cells",
        "time_relevance": "Hot: pan-T-cell marker; abundance = infiltration level",
        "ici_evidence": "T-cell abundance predicts ICI response",
    },
    "CD3E": {
        "category": "t_cell_marker",
        "function": "T-cell receptor signaling chain",
        "literature_level": "extensive",
        "cell_types": "all T cells",
        "time_relevance": "Hot: T-cell infiltration marker",
        "ici_evidence": "component of immune infiltration scores",
    },
    "CD8A": {
        "category": "t_cell_marker",
        "function": "CD8 co-receptor alpha chain; marks cytotoxic T cells",
        "literature_level": "extensive",
        "cell_types": "CD8+ cytotoxic T cells",
        "time_relevance": "Hot: cytotoxic T-cell abundance",
        "ici_evidence": "CD8+ T-cell density predicts ICI response",
    },
    "CD8B": {
        "category": "t_cell_marker",
        "function": "CD8 co-receptor beta chain",
        "literature_level": "strong",
        "cell_types": "CD8+ cytotoxic T cells",
        "time_relevance": "Hot: cytotoxic T-cell marker",
        "ici_evidence": "component of cytotoxic T-cell signature",
    },
    "GZMA": {
        "category": "cytotoxicity",
        "function": "Granzyme A; serine protease for target cell killing",
        "literature_level": "extensive",
        "cell_types": "CD8+ T cells, NK cells",
        "time_relevance": "Hot: active cytotoxic effector function",
        "ici_evidence": "cytotoxicity signature predicts ICI benefit",
    },
    "GZMB": {
        "category": "cytotoxicity",
        "function": "Granzyme B; primary cytotoxic granule protease",
        "literature_level": "extensive",
        "cell_types": "CD8+ T cells, NK cells",
        "time_relevance": "Hot: direct tumor cell killing",
        "ici_evidence": "active cytotoxicity marker in responders",
    },
    "PRF1": {
        "category": "cytotoxicity",
        "function": "Perforin; pore-forming protein for granzyme delivery",
        "literature_level": "extensive",
        "cell_types": "CD8+ T cells, NK cells",
        "time_relevance": "Hot: essential for cytotoxic killing",
        "ici_evidence": "deficiency impairs anti-tumor response",
    },
    "ICOS": {
        "category": "co_stimulation",
        "function": "Inducible T-cell co-stimulator; enhances T-cell activation",
        "literature_level": "strong",
        "cell_types": "activated T cells",
        "time_relevance": "Hot: T-cell activation and Tfh differentiation",
        "ici_evidence": "ICOS+ T cells expand after anti-CTLA-4 therapy",
    },
    # --- Immune exclusion signals ---
    "TGFB1": {
        "category": "immune_exclusion",
        "function": "TGF-beta 1; master regulator of immune suppression and fibrosis",
        "literature_level": "extensive",
        "cell_types": "CAFs, regulatory T cells, macrophages, tumor cells",
        "time_relevance": "Excluded: drives stromal barrier, CAF activation, T-cell exclusion",
        "ici_evidence": "TGF-beta blockade enhances anti-PD-1 response (bintrafusp alfa)",
    },
    "TGFB2": {
        "category": "immune_exclusion",
        "function": "TGF-beta 2; immunosuppressive cytokine",
        "literature_level": "moderate",
        "cell_types": "tumor cells, stromal cells",
        "time_relevance": "Excluded: immune exclusion mediator",
        "ici_evidence": "component of TGF-beta axis",
    },
    "VEGFA": {
        "category": "angiogenesis",
        "function": "Vascular endothelial growth factor A; angiogenesis + immunosuppression",
        "literature_level": "extensive",
        "cell_types": "tumor cells, macrophages, stromal cells",
        "time_relevance": "Excluded/Cold: abnormal vasculature impedes T-cell trafficking",
        "ici_evidence": "anti-VEGF + anti-PD-L1 combinations (atezolizumab + bevacizumab)",
    },
    "WNT5A": {
        "category": "immune_exclusion",
        "function": "Non-canonical Wnt ligand; promotes immune exclusion",
        "literature_level": "moderate",
        "cell_types": "tumor cells, macrophages",
        "time_relevance": "Excluded: drives T-cell exclusion via beta-catenin signaling",
        "ici_evidence": "WNT/beta-catenin activation predicts ICI resistance",
    },
    "CTNNB1": {
        "category": "immune_exclusion",
        "function": "Beta-catenin; Wnt pathway effector; immune exclusion via CCL4 loss",
        "literature_level": "strong",
        "cell_types": "tumor cells",
        "time_relevance": "Excluded/Cold: activating mutations suppress DC recruitment",
        "ici_evidence": "CTNNB1-mutant tumors resist ICI",
    },
    # --- Chemokines ---
    "CCL2": {
        "category": "chemokine",
        "function": "MCP-1; recruits monocytes and macrophages",
        "literature_level": "extensive",
        "cell_types": "tumor cells, endothelial cells, fibroblasts",
        "time_relevance": "Hot/Excluded: recruits TAMs; dual role",
        "ici_evidence": "CCL2 blockade: context-dependent effects",
    },
    "CCL5": {
        "category": "chemokine",
        "function": "RANTES; recruits T cells, monocytes, eosinophils",
        "literature_level": "strong",
        "cell_types": "T cells, macrophages, tumor cells",
        "time_relevance": "Hot: T-cell and DC recruitment",
        "ici_evidence": "CCL5-CXCL9 axis predicts ICI response (Spranger signature)",
    },
    "CXCL12": {
        "category": "chemokine",
        "function": "SDF-1; CXCR4 ligand; retains immune cells in stroma",
        "literature_level": "strong",
        "cell_types": "CAFs, endothelial cells",
        "time_relevance": "Excluded: retains T cells in peritumoral stroma",
        "ici_evidence": "CXCR4 antagonists may overcome T-cell exclusion",
    },
    "CXCL13": {
        "category": "chemokine",
        "function": "B-cell chemoattractant; marks tertiary lymphoid structures",
        "literature_level": "strong",
        "cell_types": "Tfh cells, DCs",
        "time_relevance": "Hot: tertiary lymphoid structure formation",
        "ici_evidence": "CXCL13+ T cells predict ICI response",
    },
    # --- Myeloid markers ---
    "CD68": {
        "category": "myeloid_marker",
        "function": "Pan-macrophage marker",
        "literature_level": "extensive",
        "cell_types": "all macrophages",
        "time_relevance": "Hot/Excluded: macrophage abundance",
        "ici_evidence": "macrophage context determines ICI response",
    },
    "CD163": {
        "category": "myeloid_marker",
        "function": "M2 macrophage marker; scavenger receptor",
        "literature_level": "strong",
        "cell_types": "M2 macrophages, TAMs",
        "time_relevance": "Excluded/Cold: immunosuppressive macrophage polarization",
        "ici_evidence": "high CD163+ TAMs associated with ICI resistance",
    },
    "CSF1R": {
        "category": "myeloid_marker",
        "function": "CSF-1 receptor; macrophage differentiation and survival",
        "literature_level": "strong",
        "cell_types": "monocytes, macrophages",
        "time_relevance": "Excluded: TAM recruitment and survival",
        "ici_evidence": "CSF1R inhibitors in combination with ICI (clinical trials)",
    },
    "ARG1": {
        "category": "metabolic_checkpoint",
        "function": "Arginase 1; depletes arginine; suppresses T-cell proliferation",
        "literature_level": "strong",
        "cell_types": "M2 macrophages, MDSCs",
        "time_relevance": "Excluded/Cold: metabolic immune suppression",
        "ici_evidence": "arginase inhibitors in combination with ICI",
    },
    "MRC1": {
        "category": "myeloid_marker",
        "function": "CD206; mannose receptor; M2 macrophage marker",
        "literature_level": "moderate",
        "cell_types": "M2 macrophages",
        "time_relevance": "Excluded: immunosuppressive macrophage polarization",
        "ici_evidence": "high CD206 associated with poor ICI outcome",
    },
}


# ---------------------------------------------------------------------------
# Immune pathway categories for network role annotation
# ---------------------------------------------------------------------------

PATHWAY_CATEGORIES: dict[str, dict[str, str]] = {
    "chemokine_signaling": {
        "label": "Chemokine Signaling",
        "time_role": "T-cell recruitment and trafficking",
        "validation_focus": "In silico: TIDE response correlation",
    },
    "antigen_presentation": {
        "label": "Antigen Presentation / MHC-I",
        "time_role": "Neoantigen display; CD8+ T-cell recognition",
        "validation_focus": "In silico: HLA LOH analysis",
    },
    "ifn_gamma_response": {
        "label": "IFN-gamma Response",
        "time_role": "Master anti-tumor immunity axis",
        "validation_focus": "In silico: IFN-gamma signature score",
    },
    "immune_checkpoint": {
        "label": "Immune Checkpoint",
        "time_role": "T-cell exhaustion and inhibitory signaling",
        "validation_focus": "In silico: positive control (known ICI targets)",
    },
    "tgfb_exclusion": {
        "label": "TGF-beta / Immune Exclusion",
        "time_role": "Stromal barrier formation; T-cell exclusion",
        "validation_focus": "In vitro: CAF co-culture; assess T-cell penetration",
    },
    "angiogenesis": {
        "label": "Angiogenesis / Vascular",
        "time_role": "Abnormal vasculature impedes immune infiltration",
        "validation_focus": "In silico: VEGF pathway score + ICI response",
    },
    "wnt_beta_catenin": {
        "label": "Wnt/beta-catenin",
        "time_role": "Immune desert phenotype; DC exclusion",
        "validation_focus": "In silico: CTNNB1 mutation status + TIME label",
    },
    "metabolic_checkpoint": {
        "label": "Metabolic Immune Suppression",
        "time_role": "Tryptophan/arginine depletion; T-cell anergy",
        "validation_focus": "In vitro: metabolite assays + T-cell proliferation",
    },
    "cytotoxicity": {
        "label": "Cytotoxic Effector Function",
        "time_role": "Direct tumor cell killing by CD8+ T/NK cells",
        "validation_focus": "In silico: cytolytic activity score",
    },
}


# ---------------------------------------------------------------------------
# Validation pathway recommendation logic
# ---------------------------------------------------------------------------

VALIDATION_STAGES = {
    "stage_1_in_silico": {
        "label": "Stage 1: In Silico Re-validation",
        "databases": ["TISCH", "TIDE", "KM-plotter", "TCGA", "Visium (independent cohort)"],
        "description": "Database queries and expression analysis in public cohorts",
    },
    "stage_2_in_vitro": {
        "label": "Stage 2: In Vitro Validation",
        "assays": ["siRNA/CRISPR knockdown", "T-cell co-culture", "Flow cytometry"],
        "description": "Cell line perturbation + immune cell co-culture assays",
    },
    "stage_3_in_vivo": {
        "label": "Stage 3: In Vivo Validation",
        "models": ["4T1 syngeneic (breast)", "B16 syngeneic (melanoma)", "anti-PD-1 combination"],
        "description": "Animal models with TIL analysis and combination therapy",
    },
}


def recommend_validation(
    gene: str,
    tier: str,
    known_info: dict[str, str] | None,
    subtype_specificity: str,
    delta_prob: float,
) -> dict[str, Any]:
    """Recommend validation pathway based on target profile."""

    recommendation: dict[str, Any] = {
        "primary_stage": "",
        "specific_approach": "",
        "rationale": "",
        "databases": [],
        "assays": [],
    }

    if known_info is not None:
        lit_level = known_info.get("literature_level", "unknown")
    else:
        lit_level = "unknown"

    # Tier 1 + extensive literature → positive control, in silico only
    if tier == "Tier 1" and lit_level == "extensive":
        recommendation["primary_stage"] = "Stage 1: In Silico"
        recommendation["specific_approach"] = (
            f"Use as positive control. Verify {gene} expression in TISCH by cell type. "
            f"Check TIDE/KIM for ICI response correlation. "
            f"Validate spatial localization in independent Visium cohort."
        )
        recommendation["rationale"] = (
            "Known immune target with extensive literature; "
            "serves as model validation positive control."
        )
        recommendation["databases"] = ["TISCH", "TIDE", "KM-plotter"]
        recommendation["assays"] = []

    # Tier 1 + moderate/strong literature → in silico + optional in vitro
    elif tier == "Tier 1" and lit_level in ("strong", "moderate"):
        recommendation["primary_stage"] = "Stage 1: In Silico + Stage 2: In Vitro (optional)"
        recommendation["specific_approach"] = (
            f"Validate {gene} expression localization in TISCH. "
            f"Check survival stratification in KM-plotter. "
            f"If ICI cohort data available, test TIDE correlation. "
            f"Consider siRNA knockdown in relevant cell line + T-cell co-culture."
        )
        recommendation["rationale"] = (
            "Established immune function with moderate-to-strong literature; "
            "in silico validation required; in vitro strengthens mechanism."
        )
        recommendation["databases"] = ["TISCH", "KM-plotter", "TIDE"]
        recommendation["assays"] = ["siRNA knockdown", "T-cell co-culture (optional)"]

    # Tier 1 + no/limited literature → novel but stable → in silico + in vitro required
    elif tier == "Tier 1":
        recommendation["primary_stage"] = "Stage 1: In Silico + Stage 2: In Vitro"
        recommendation["specific_approach"] = (
            f"Verify {gene} expression and cell-type distribution in TISCH. "
            f"Assess spatial distribution in Visium data. "
            f"Perform siRNA/CRISPR knockdown in tumor or stromal cell line. "
            f"Co-culture with PBMCs/T-cells; measure IFN-gamma, Granzyme B (activation) "
            f"and PD-1, TIM-3 (exhaustion) by flow cytometry."
        )
        recommendation["rationale"] = (
            "Computationally stable and sensitive target with limited prior literature; "
            "represents potential novel finding requiring experimental validation."
        )
        recommendation["databases"] = ["TISCH", "TIDE", "Visium"]
        recommendation["assays"] = ["CRISPR knockdown", "T-cell co-culture", "Flow cytometry"]

    # Tier 2 with high perturbation → exploratory but impactful
    elif tier == "Tier 2" and delta_prob >= 0.02:
        recommendation["primary_stage"] = "Stage 1: In Silico + Stage 2: In Vitro (recommended)"
        if "excluded" in subtype_specificity.lower():
            recommendation["specific_approach"] = (
                f"Prioritize {gene} for Excluded→Hot conversion hypothesis. "
                f"Check CAF/stromal expression in TISCH. "
                f"In vitro: knockdown in CAF line + T-cell co-culture; "
                f"assess T-cell penetration in 3D spheroid model."
            )
        elif "cold" in subtype_specificity.lower():
            recommendation["specific_approach"] = (
                f"Investigate {gene} role in immune recruitment. "
                f"Check DC/macrophage expression in TISCH. "
                f"In vitro: overexpression in tumor cell line; "
                f"assess immune cell migration in transwell assay."
            )
        else:
            recommendation["specific_approach"] = (
                f"Verify {gene} cell-type expression in TISCH. "
                f"Perform knockdown in relevant cell line; "
                f"co-culture with T-cells to measure functional impact."
            )
        recommendation["rationale"] = (
            "Exploratory target with strong perturbation signal but insufficient "
            "cross-validation stability; requires independent experimental confirmation."
        )
        recommendation["databases"] = ["TISCH", "TIDE"]
        recommendation["assays"] = ["siRNA/CRISPR", "Co-culture", "Flow cytometry"]

    # Tier 2 low perturbation → in silico screening only
    else:
        recommendation["primary_stage"] = "Stage 1: In Silico (screening)"
        recommendation["specific_approach"] = (
            f"Screen {gene} expression pattern in TISCH. "
            f"Check survival association in KM-plotter. "
            f"If no signal, deprioritize. "
            f"If signal found, escalate to Stage 2."
        )
        recommendation["rationale"] = (
            "Model-nominated target lacking stability or perturbation support; "
            "requires in silico screening before experimental investment."
        )
        recommendation["databases"] = ["TISCH", "KM-plotter"]
        recommendation["assays"] = []

    return recommendation


# ---------------------------------------------------------------------------
# Subtype specificity assessment
# ---------------------------------------------------------------------------

def assess_subtype_specificity(
    gene_id: str,
    subtype_rankings: dict[str, list[dict[str, Any]]],
    topk: int = 30,
) -> str:
    """Determine which TIME subtype(s) a gene is most strongly associated with."""
    present_in: list[str] = []
    for subtype, ranking_rows in subtype_rankings.items():
        for row in ranking_rows[:topk]:
            if isinstance(row, dict):
                rid = row.get("gene_id") or row.get("Gene_ID") or row.get("id", "")
            elif isinstance(row, (list, tuple)) and len(row) >= 1:
                rid = str(row[0])
            else:
                continue
            if str(rid) == str(gene_id):
                present_in.append(subtype)
                break

    if not present_in:
        return "Non-specific"
    if len(present_in) == 1:
        return f"{present_in[0]}-specific"
    return f"Shared ({', '.join(sorted(present_in))})"


# ---------------------------------------------------------------------------
# TSV loading utilities
# ---------------------------------------------------------------------------

def load_gene_ranking_tsv(path: Path) -> list[dict[str, Any]]:
    """Load gene_ranking.tsv from interpretability output."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Normalize numeric fields
            for key in ("Rank", "Avg_Model_Score", "CV_Stability_Jaccard",
                        "Perturbation_Delta_Prob", "Perturbation_Delta_Pheno"):
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_pathway_ranking_tsv(path: Path) -> list[dict[str, Any]]:
    """Load pathway_ranking.tsv from interpretability output."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in ("Rank", "Avg_Model_Score", "CV_Stability",
                        "Perturbation_Delta_Prob", "Perturbation_Delta_Pheno"):
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_enrichment_tsv(path: Path) -> list[dict[str, Any]]:
    """Load pathway_enrichment.tsv."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in ("Overlap_Count", "Pathway_Size", "TopK_Size",
                        "Background_Size", "P_Value", "FDR"):
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_subtype_rankings(interp_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load subtype-specific gene rankings."""
    subtype_data: dict[str, list[dict[str, Any]]] = {}
    for tsv_path in sorted(interp_dir.glob("subtype_*_gene_ranking.tsv")):
        # Extract subtype name from filename: subtype_{name}_gene_ranking.tsv
        stem = tsv_path.stem
        parts = stem.split("_")
        # Find "gene" position and take everything between "subtype" and "gene"
        try:
            gene_idx = parts.index("gene")
            subtype_name = "_".join(parts[1:gene_idx])
        except ValueError:
            continue
        rows = []
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
        subtype_data[subtype_name] = rows
    return subtype_data


def load_interpretability_summary(interp_dir: Path) -> dict[str, Any]:
    """Load interpretability_summary.json."""
    path = interp_dir / "interpretability_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Network role annotation
# ---------------------------------------------------------------------------

def annotate_network_role(
    gene_id: str,
    known_info: dict[str, str] | None,
    pathway_memberships: list[str],
    subtype_spec: str,
    tier: str,
) -> str:
    """Generate a concise network role description for the gene."""
    parts: list[str] = []

    if known_info:
        category = known_info.get("category", "")
        cat_label = PATHWAY_CATEGORIES.get(category, {}).get("label", category)
        if cat_label:
            parts.append(f"{cat_label} component")

    if pathway_memberships:
        pw_str = ", ".join(pathway_memberships[:3])
        parts.append(f"member of {pw_str}")

    if "specific" in subtype_spec.lower():
        subtype = subtype_spec.replace("-specific", "")
        parts.append(f"preferentially weighted in {subtype} phenotype")
    elif "shared" in subtype_spec.lower():
        parts.append("cross-subtype hub")

    if tier == "Tier 2" and not known_info:
        parts.append("novel graph-topology-derived candidate")

    if not parts:
        parts.append("HGT-identified node")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Main evidence compilation
# ---------------------------------------------------------------------------

def compile_target_evidence(
    gene_rankings: list[dict[str, Any]],
    pathway_rankings: list[dict[str, Any]],
    enrichment_results: list[dict[str, Any]],
    subtype_rankings: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    topk: int = 50,
) -> list[dict[str, Any]]:
    """Compile biological evidence profile for each candidate target gene."""

    # Build gene→pathway mapping from enrichment
    gene_pathways: dict[str, list[str]] = defaultdict(list)
    for enr in enrichment_results:
        fdr = enr.get("FDR", 1.0)
        if isinstance(fdr, str):
            try:
                fdr = float(fdr)
            except ValueError:
                fdr = 1.0
        if fdr < 0.25:
            genes_str = enr.get("Overlapping_Genes", "")
            pathway_name = enr.get("Pathway", "")
            for g in genes_str.split("; "):
                g = g.strip()
                if g:
                    gene_pathways[g].append(pathway_name)

    evidence_list: list[dict[str, Any]] = []

    for row in gene_rankings[:topk]:
        gene_id = row.get("Gene_ID", row.get("Symbol", ""))
        tier = row.get("Tier", "Tier 2")
        score = row.get("Avg_Model_Score", 0.0)
        stability = row.get("CV_Stability_Jaccard", 0.0)
        delta_prob = row.get("Perturbation_Delta_Prob", 0.0)
        delta_pheno = row.get("Perturbation_Delta_Pheno", 0.0)
        known_match = str(row.get("Known_Target_Match", "False")).lower() == "true"
        assoc_pathways_str = row.get("Associated_Pathways", "")

        # Lookup in known immune database
        known_info = IMMUNE_GENE_DATABASE.get(gene_id)

        # Pathway memberships
        pw_from_enrichment = gene_pathways.get(gene_id, [])
        pw_from_ranking = [p.strip() for p in assoc_pathways_str.split(";") if p.strip()]
        all_pathways = list(set(pw_from_enrichment + pw_from_ranking))

        # Subtype specificity
        subtype_spec = assess_subtype_specificity(gene_id, subtype_rankings, topk=30)

        # Network role
        network_role = annotate_network_role(
            gene_id, known_info, all_pathways, subtype_spec, tier,
        )

        # Literature support level
        if known_info:
            lit_level = known_info["literature_level"]
            lit_desc = f"{lit_level.capitalize()} ({known_info['function'][:80]})"
        else:
            lit_desc = "Limited/Unknown (novel candidate from graph topology)"
            lit_level = "unknown"

        # Single-cell localization
        if known_info:
            cell_types = known_info["cell_types"]
        else:
            cell_types = "To be determined (TISCH query required)"

        # ICI cohort evidence
        if known_info:
            ici_evidence = known_info["ici_evidence"]
        else:
            ici_evidence = "To be tested (requires external validation)"

        # Validation recommendation
        validation = recommend_validation(
            gene_id, tier, known_info, subtype_spec,
            float(delta_prob) if isinstance(delta_prob, (int, float)) else 0.0,
        )

        evidence_list.append({
            "Gene_Symbol": gene_id,
            "Rank": row.get("Rank", 0),
            "Rank_Tier": tier.replace(" ", "_"),
            "Avg_Model_Score": score,
            "CV_Stability": stability,
            "Perturbation_Delta_Prob": delta_prob,
            "Perturbation_Delta_Pheno": delta_pheno,
            "Network_Role_in_HGT": network_role,
            "Literature_Support": lit_desc,
            "Single_Cell_Localization": cell_types,
            "Subtype_Specificity": subtype_spec,
            "ICI_Cohort_Validation_Status": ici_evidence,
            "Associated_Pathways": "; ".join(all_pathways) if all_pathways else "",
            "Suggested_Validation_Pathway": validation["primary_stage"],
            "Validation_Approach": validation["specific_approach"],
            "Validation_Rationale": validation["rationale"],
            "Suggested_Databases": ", ".join(validation["databases"]),
            "Suggested_Assays": ", ".join(validation["assays"]) if validation["assays"] else "N/A",
            "TIME_Relevance": known_info["time_relevance"] if known_info else "To be characterized",
        })

    return evidence_list


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_priority_tsv(evidence: list[dict[str, Any]], output_path: Path) -> None:
    """Write candidate_target_priority.tsv conforming to project template."""
    header = (
        "Gene_Symbol\tRank_Tier\tNetwork_Role_in_HGT\t"
        "Literature_Support\tSingle_Cell_Localization\t"
        "ICI_Cohort_Validation_Status\tSuggested_Validation_Pathway"
    )
    lines = [header]
    for entry in evidence:
        line = (
            f"{entry['Gene_Symbol']}\t"
            f"{entry['Rank_Tier']}\t"
            f"{entry['Network_Role_in_HGT']}\t"
            f"{entry['Literature_Support']}\t"
            f"{entry['Single_Cell_Localization']}\t"
            f"{entry['ICI_Cohort_Validation_Status']}\t"
            f"{entry['Suggested_Validation_Pathway']}"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Priority TSV: {output_path}")


def write_extended_evidence_tsv(evidence: list[dict[str, Any]], output_path: Path) -> None:
    """Write extended evidence TSV with all fields."""
    columns = [
        "Gene_Symbol", "Rank", "Rank_Tier", "Avg_Model_Score",
        "CV_Stability", "Perturbation_Delta_Prob", "Perturbation_Delta_Pheno",
        "Network_Role_in_HGT", "Literature_Support", "Single_Cell_Localization",
        "Subtype_Specificity", "ICI_Cohort_Validation_Status",
        "Associated_Pathways", "TIME_Relevance",
        "Suggested_Validation_Pathway", "Validation_Approach",
        "Validation_Rationale", "Suggested_Databases", "Suggested_Assays",
    ]
    lines = ["\t".join(columns)]
    for entry in evidence:
        values = []
        for col in columns:
            val = entry.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6f}"
            values.append(str(val))
        lines.append("\t".join(values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Extended evidence TSV: {output_path}")


def write_validation_roadmap_json(evidence: list[dict[str, Any]], output_path: Path) -> None:
    """Write structured validation roadmap JSON."""
    roadmap = {
        "validation_stages": VALIDATION_STAGES,
        "tier_1_targets": [],
        "tier_2_targets": [],
        "positive_controls": [],
        "summary": {
            "total_candidates": len(evidence),
            "tier_1_count": 0,
            "tier_2_count": 0,
            "known_target_count": 0,
            "novel_target_count": 0,
        },
    }

    for entry in evidence:
        target_entry = {
            "gene": entry["Gene_Symbol"],
            "rank": entry["Rank"],
            "score": entry["Avg_Model_Score"],
            "stability": entry["CV_Stability"],
            "delta_prob": entry["Perturbation_Delta_Prob"],
            "subtype": entry["Subtype_Specificity"],
            "validation_stage": entry["Suggested_Validation_Pathway"],
            "approach": entry["Validation_Approach"],
            "databases": entry["Suggested_Databases"],
            "assays": entry["Suggested_Assays"],
        }

        is_known = entry["Gene_Symbol"] in IMMUNE_GENE_DATABASE
        if is_known:
            roadmap["summary"]["known_target_count"] += 1
        else:
            roadmap["summary"]["novel_target_count"] += 1

        if entry["Rank_Tier"] == "Tier_1":
            roadmap["tier_1_targets"].append(target_entry)
            roadmap["summary"]["tier_1_count"] += 1
            if is_known and IMMUNE_GENE_DATABASE[entry["Gene_Symbol"]].get(
                "literature_level"
            ) == "extensive":
                roadmap["positive_controls"].append(entry["Gene_Symbol"])
        else:
            roadmap["tier_2_targets"].append(target_entry)
            roadmap["summary"]["tier_2_count"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(roadmap, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Validation roadmap JSON: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compile biological evidence for candidate targets"
    )
    parser.add_argument(
        "--interpretability-dir", type=str, required=True,
        help="Directory containing interpretability outputs (gene_ranking.tsv, etc.)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: {interpretability-dir}/../target_evidence)",
    )
    parser.add_argument("--topk", type=int, default=50, help="Top-k targets to process")
    args = parser.parse_args()

    interp_dir = Path(args.interpretability_dir)
    if not interp_dir.exists():
        raise SystemExit(f"Interpretability directory not found: {interp_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = interp_dir.parent / "target_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load upstream outputs
    # ========================================================================
    logger.info("=== Loading upstream interpretability outputs ===")

    gene_rankings = load_gene_ranking_tsv(interp_dir / "gene_ranking.tsv")
    logger.info(f"  Gene rankings: {len(gene_rankings)} entries")

    pathway_rankings = load_pathway_ranking_tsv(interp_dir / "pathway_ranking.tsv")
    logger.info(f"  Pathway rankings: {len(pathway_rankings)} entries")

    enrichment_results = load_enrichment_tsv(interp_dir / "pathway_enrichment.tsv")
    logger.info(f"  Enrichment results: {len(enrichment_results)} entries")

    subtype_rankings = load_subtype_rankings(interp_dir)
    logger.info(f"  Subtype rankings: {list(subtype_rankings.keys())}")

    summary = load_interpretability_summary(interp_dir)

    # ========================================================================
    # Compile evidence
    # ========================================================================
    logger.info("=== Compiling biological evidence ===")

    evidence = compile_target_evidence(
        gene_rankings=gene_rankings,
        pathway_rankings=pathway_rankings,
        enrichment_results=enrichment_results,
        subtype_rankings=subtype_rankings,
        summary=summary,
        topk=args.topk,
    )
    logger.info(f"  Compiled evidence for {len(evidence)} targets")

    n_tier1 = sum(1 for e in evidence if e["Rank_Tier"] == "Tier_1")
    n_tier2 = sum(1 for e in evidence if e["Rank_Tier"] == "Tier_2")
    n_known = sum(1 for e in evidence if e["Gene_Symbol"] in IMMUNE_GENE_DATABASE)
    n_novel = len(evidence) - n_known
    logger.info(f"  Tier 1: {n_tier1}, Tier 2: {n_tier2}")
    logger.info(f"  Known immune targets: {n_known}, Novel candidates: {n_novel}")

    # ========================================================================
    # Write outputs
    # ========================================================================
    logger.info("=== Writing outputs ===")

    # Project-template-conformant priority TSV
    write_priority_tsv(evidence, output_dir / "candidate_target_priority.tsv")

    # Extended evidence with all fields
    write_extended_evidence_tsv(evidence, output_dir / "candidate_target_extended_evidence.tsv")

    # Structured validation roadmap
    write_validation_roadmap_json(evidence, output_dir / "validation_roadmap.json")

    # Tier 1 only (for paper main text)
    tier1_evidence = [e for e in evidence if e["Rank_Tier"] == "Tier_1"]
    if tier1_evidence:
        write_extended_evidence_tsv(
            tier1_evidence, output_dir / "tier1_targets_for_paper.tsv",
        )
        logger.info(f"  Tier 1 targets for paper: {len(tier1_evidence)}")

    # Novel candidates only (for Discussion section)
    novel_evidence = [e for e in evidence if e["Gene_Symbol"] not in IMMUNE_GENE_DATABASE]
    if novel_evidence:
        write_extended_evidence_tsv(
            novel_evidence, output_dir / "novel_candidates.tsv",
        )
        logger.info(f"  Novel candidates: {len(novel_evidence)}")

    logger.info("=== Target evidence compilation complete ===")


if __name__ == "__main__":
    main()
