################################################################################
# PROJECT:
# AI-assisted literature review pipeline
#
# STRICT TOPIC:
# Determinants of public acceptability/support/opposition toward carbon taxes,
# fuel taxes, gasoline taxes, diesel taxes, fossil-fuel taxes, CO2 taxes, and
# TICPE-like fiscal instruments.
#
# IMPORTANT:
# This version is deliberately STRICT.
#
# It keeps only papers that directly analyze a carbon/fuel tax instrument.
# It excludes papers that only analyze:
#   - ETS / cap-and-trade / carbon markets / emission permits
#   - carbon pricing in general without a tax-specific analysis
#   - subsidies / standards / green spending
#   - pro-environmental behavior without tax acceptability
#   - general climate-policy support without a carbon/fuel tax item
#
# PIPELINE:
#   Step 0. Find and upload PDFs.
#   Step 1. Screening: full PDF -> decide whether paper contains direct
#           carbon-tax/fuel-tax/TICPE-like evidence.
#   Step 2. Extraction: full PDF -> extract only determinants directly associated
#           with acceptability/support/opposition/preferences toward carbon/fuel tax.
#   Step 3. Verification: full PDF -> verify and correct extraction.
#   Step 4. Build paper-level, long, audit, and harmonized tables.
#
# PARALLELIZATION:
#   - Papers are processed in parallel.
#   - Within each paper, stages remain sequential:
#       screening -> extraction -> verification -> final JSON.
#   - OpenAI calls are bounded by MAX_CONCURRENT_API_CALLS.
#   - File cache writes are protected by a lock.
#
# ROBUSTNESS FEATURES:
#   - Saves one status file per paper.
#   - Skips already completed included papers.
#   - Skips already completed excluded papers.
#   - Retries failed papers only.
#   - Reuses completed intermediate steps.
#   - Saves raw API responses when JSON parsing fails.
#   - Automatically increases max_output_tokens when output looks truncated.
#   - Moves corrupt JSON files aside instead of crashing.
#   - Writes JSON files atomically.
#
# REQUIREMENTS:
#   pip install openai pandas openpyxl pydantic tqdm python-dotenv
#
# ENVIRONMENT:
#   export OPENAI_API_KEY="your_api_key_here"
#
################################################################################


from __future__ import annotations

import json
import re
import time
import hashlib
import unicodedata
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, get_args
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from openai import OpenAI


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

import os

LITERATURE_DIR = Path(
    os.getenv("LITERATURE_DIR", Path(__file__).resolve().parents[1] / "data" / "papers")
)

OUTPUT_DIR = LITERATURE_DIR / "_ai_carbon_tax_strict_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_DIR = OUTPUT_DIR / "json_outputs"
SCREENING_DIR = JSON_DIR / "00_screening"
EXTRACTION_DIR = JSON_DIR / "01_extraction"
VERIFICATION_DIR = JSON_DIR / "02_verification"
FINAL_JSON_DIR = JSON_DIR / "03_final"
STATUS_DIR = JSON_DIR / "99_status"
LOG_DIR = OUTPUT_DIR / "logs"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [
    JSON_DIR,
    SCREENING_DIR,
    EXTRACTION_DIR,
    VERIFICATION_DIR,
    FINAL_JSON_DIR,
    STATUS_DIR,
    LOG_DIR,
    TABLE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------------------

MODEL_SCREEN = "gpt-5.4-mini"
MODEL_EXTRACT = "gpt-5.4-mini"
MODEL_VERIFY = "gpt-5.4-mini"
MODEL_HARMONIZE = "gpt-5.4-mini"

REASONING_EFFORT_SCREEN = "high"
REASONING_EFFORT_EXTRACT = "medium"
REASONING_EFFORT_VERIFY = "high"
REASONING_EFFORT_HARMONIZE = "high"


# ------------------------------------------------------------------------------
# Token budgets
# ------------------------------------------------------------------------------

MAX_OUTPUT_TOKENS_SCREEN = 10000
MAX_OUTPUT_TOKENS_EXTRACT = 45000
MAX_OUTPUT_TOKENS_VERIFY = 45000
MAX_OUTPUT_TOKENS_HARMONIZE = 120000

TOKEN_RETRY_MULTIPLIER = 1.5
MAX_TOKEN_RETRY_CEILING = 120000


# ------------------------------------------------------------------------------
# Run options
# ------------------------------------------------------------------------------

OVERWRITE_EXISTING = False
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3

TEST_MODE = False
N_TEST_FILES = 5

# Verification is now used as a lightweight audit step.
# The final JSON is the extraction JSON, not a full JSON re-written during verification.
USE_VERIFIED_FINAL_JSON = False
USE_FILE_CACHE = True

FILE_CACHE_PATH = OUTPUT_DIR / "openai_file_cache.json"

SAVE_SCHEMA_DEBUG = True

# If a paper was interrupted or failed, the pipeline retries it.
# Completed papers are skipped only if their expected output file is present and valid.
RETRY_FAILED_OR_INTERRUPTED = True

# Running statuses older than this threshold are considered stale and retried.
STALE_RUNNING_AFTER_HOURS = 2.0

# Rebuild harmonization after new papers are added or old outputs are cleaned.
OVERWRITE_HARMONIZATION = True


# ------------------------------------------------------------------------------
# Parallelization options
# ------------------------------------------------------------------------------

PARALLEL_PROCESSING = True

# Number of papers processed at the same time.
# Start with 2 or 3. Increase only if your API rate limits allow it.
MAX_WORKERS = 3

# Max simultaneous OpenAI calls across all threads.
# Usually keep equal to MAX_WORKERS or slightly lower.
MAX_CONCURRENT_API_CALLS = 3


# ==============================================================================
# 1. OPENAI CLIENT AND THREADING LOCKS
# ==============================================================================

client = OpenAI(timeout=1200.0)

FILE_CACHE_LOCK = threading.Lock()
API_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_API_CALLS)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    """
    Thread-safe print.
    """
    with PRINT_LOCK:
        print(message, flush=True)


# ==============================================================================
# 2. CLOSED VOCABULARIES
# ==============================================================================

ScreeningDecision = Literal[
    "include_direct_carbon_or_fuel_tax",
    "exclude_no_direct_carbon_or_fuel_tax",
    "uncertain_needs_manual_review",
]

TaxInstrumentType = Literal[
    "carbon_tax",
    "co2_tax",
    "fuel_tax",
    "gasoline_tax",
    "diesel_tax",
    "fossil_fuel_tax",
    "energy_tax_explicitly_carbon_or_fuel_based",
    "ticpe_like_excise_fuel_tax",
    "carbon_tax_with_revenue_recycling",
    "progressive_carbon_tax",
    "other_direct_carbon_or_fuel_tax",
    "unclear_tax_instrument",
]

ExcludedPolicyType = Literal[
    "ets_or_cap_and_trade",
    "carbon_pricing_general_without_tax_specific_analysis",
    "subsidy_or_incentive",
    "regulation_or_standard",
    "green_public_spending",
    "general_climate_policy_support",
    "environmental_policy_general",
    "pro_environmental_behavior_only",
    "willingness_to_contribute_or_donate_without_tax",
    "other_non_tax_policy",
    "not_applicable",
]

PaperType = Literal[
    "meta_analysis",
    "systematic_review",
    "narrative_review",
    "empirical_survey",
    "survey_experiment",
    "field_experiment",
    "lab_experiment",
    "referendum_or_voting_analysis",
    "quasi_experimental_policy_evaluation",
    "observational_empirical_analysis",
    "theoretical_or_conceptual",
    "commentary_or_policy_perspective",
    "working_paper",
    "other",
    "unclear",
]

DeterminantFamily = Literal[
    "policy_design",
    "revenue_recycling",
    "perceived_effectiveness",
    "fairness_and_distribution",
    "economic_self_interest_or_cost_exposure",
    "trust_and_institutions",
    "ideology_partisanship_and_values",
    "social_norms_and_perceived_support",
    "climate_beliefs_risk_perception_and_concern",
    "emotions_and_affect",
    "knowledge_information_and_misperceptions",
    "policy_experience_and_learning",
    "demographics_and_socioeconomic_status",
    "geography_and_contextual_exposure",
    "sectoral_or_occupational_exposure",
    "other",
    "unclear",
]

DeterminantRole = Literal[
    "substantive_mechanism",
    "policy_design_feature",
    "revenue_or_recycling_design",
    "experimental_treatment",
    "demographic_control",
    "socioeconomic_control",
    "political_control",
    "country_context",
    "methodological_or_study_design_covariate",
    "background_theoretical_concept",
    "unclear",
]

Direction = Literal["+", "-", "mixed", "n.a."]

EvidenceType = Literal[
    "statistical_association",
    "causal_estimate",
    "experimental_treatment_effect",
    "meta_analytic_estimate",
    "review_conclusion",
    "theoretical_argument",
    "descriptive_evidence",
    "qualitative_evidence",
    "not_applicable_or_unclear",
]

Confidence = Literal["high", "medium", "low"]

InclusionInMainTable = Literal[
    "yes_main_result",
    "yes_secondary_result",
    "no_control_or_methodological_only",
    "no_insufficiently_tax_specific",
    "no_unsupported_or_unclear",
]


SCREENING_DECISIONS = list(get_args(ScreeningDecision))
TAX_INSTRUMENT_TYPES = list(get_args(TaxInstrumentType))
EXCLUDED_POLICY_TYPES = list(get_args(ExcludedPolicyType))
PAPER_TYPES = list(get_args(PaperType))
DETERMINANT_FAMILIES = list(get_args(DeterminantFamily))
DETERMINANT_ROLES = list(get_args(DeterminantRole))
DIRECTIONS = list(get_args(Direction))
EVIDENCE_TYPES = list(get_args(EvidenceType))
CONFIDENCE_LEVELS = list(get_args(Confidence))
INCLUSION_IN_MAIN_TABLE_VALUES = list(get_args(InclusionInMainTable))


# ==============================================================================
# 3. PYDANTIC MODELS
# ==============================================================================

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaxMention(StrictBaseModel):
    tax_instrument_type: TaxInstrumentType
    exact_term_used_in_paper: str
    context: str
    page_or_section: str
    is_direct_acceptability_or_support_analysis: bool
    notes: Optional[str] = Field(description="Use null if none.")


class ExcludedPolicyMention(StrictBaseModel):
    excluded_policy_type: ExcludedPolicyType
    exact_term_used_in_paper: str
    reason_it_is_not_a_carbon_or_fuel_tax: str
    page_or_section: str


class ScreeningJSON(StrictBaseModel):
    paper_id: str
    title: Optional[str] = Field(description="Use null if unavailable.")
    authors: Optional[str] = Field(description="Use null if unavailable.")
    doi: Optional[str] = Field(description="Use null if unavailable.")
    publication_year: Optional[int] = Field(description="Use null if unavailable.")
    paper_type: PaperType

    screening_decision: ScreeningDecision
    contains_direct_carbon_or_fuel_tax: bool
    contains_tax_acceptability_or_support_analysis: bool

    tax_mentions: List[TaxMention]
    excluded_policy_mentions: List[ExcludedPolicyMention]

    inclusion_reason: str
    exclusion_reason: Optional[str] = Field(description="Use null if included.")
    confidence: Confidence


class PaperMetadataExtraction(StrictBaseModel):
    paper_id: str
    title: str
    authors: Optional[str] = Field(description="Use null if unavailable.")
    doi: Optional[str] = Field(description="Use null if unavailable.")
    publication_year: Optional[int] = Field(description="Use null if unavailable.")
    publication_date: Optional[str] = Field(description="Use null if unavailable.")
    journal_or_source: Optional[str] = Field(description="Use null if unavailable.")
    paper_type: PaperType
    geographic_scope: Optional[str] = Field(description="Use null if unavailable.")
    empirical_scope_or_sample: Optional[str] = Field(description="Use null if unavailable.")
    tax_instruments_analyzed: List[TaxInstrumentType]
    exact_tax_terms_used: List[str]
    carbon_tax_relevance_summary: str


class DeterminantExtraction(StrictBaseModel):
    raw_name_in_paper: str
    standardized_candidate_label: str
    determinant_family: DeterminantFamily
    determinant_role: DeterminantRole
    direction_on_tax_acceptability: Direction

    definition: str
    mechanism: str
    evidence_type: EvidenceType

    tax_instrument_type: TaxInstrumentType
    exact_tax_policy_context: str

    outcome_variable: str
    acceptability_dimension: Literal[
        "support",
        "opposition",
        "acceptability",
        "acceptance",
        "vote_choice",
        "willingness_to_pay",
        "policy_preference",
        "attitude",
        "other",
        "unclear",
    ]

    is_directly_about_carbon_or_fuel_tax: bool
    is_exclusively_about_ets_or_non_tax_policy: bool

    evidence_quote: str
    page_or_section: str
    statistical_or_result_detail: Optional[str] = Field(description="Use null if unavailable.")
    confidence: Confidence

    include_in_main_table: InclusionInMainTable
    exclusion_reason_if_not_main: Optional[str] = Field(description="Use null if included.")
    notes: Optional[str] = Field(description="Use null if none.")


class ExclusionOrUncertainty(StrictBaseModel):
    issue: str
    reason: str


class PaperExtractionJSON(StrictBaseModel):
    paper_metadata: PaperMetadataExtraction
    determinants: List[DeterminantExtraction]
    exclusions_or_uncertainties: List[ExclusionOrUncertainty]


class MetadataChecks(StrictBaseModel):
    title_correct: bool
    doi_correct: bool
    year_correct: bool
    paper_type_correct: bool
    tax_scope_correct: bool
    comments: Optional[str] = Field(description="Use null if none.")


class DeterminantCheck(StrictBaseModel):
    raw_name_in_paper: str
    status: Literal[
        "supported",
        "unsupported",
        "direction_error",
        "definition_error",
        "duplicate",
        "too_broad",
        "missing_evidence",
        "wrong_policy_context",
        "not_tax_specific",
        "control_not_substantive",
        "other",
    ]
    comment: str
    suggested_correction: Optional[str] = Field(description="Use null if no correction.")


class MissingTaxDeterminant(StrictBaseModel):
    raw_name_in_paper: str
    suggested_direction: Direction
    tax_instrument_type: TaxInstrumentType
    outcome_variable: str
    evidence_quote: str
    page_or_section: str
    comment: str


class PaperVerificationJSON(StrictBaseModel):
    """
    Lightweight verification schema.

    Important design choice:
    The verification step no longer reproduces a full corrected PaperExtractionJSON.
    Requiring the model to output both a full audit and a full corrected extraction
    caused long JSON outputs to be truncated. The final JSON is therefore the
    extraction JSON, while verification is stored as an audit layer.
    """
    paper_id: str
    overall_verdict: Literal["pass", "minor_revision", "major_revision", "fail"]
    should_remain_in_corpus: bool
    reason_for_exclusion_if_any: Optional[str] = Field(description="Use null if kept.")
    metadata_checks: MetadataChecks
    determinant_checks: List[DeterminantCheck]
    missing_tax_determinants: List[MissingTaxDeterminant]
    final_recommendation_summary: str


class HarmonizedDeterminant(StrictBaseModel):
    canonical_label: str
    determinant_family: DeterminantFamily
    definition: str
    raw_labels_grouped: List[str]
    recommended_main_table_label: str
    distinction_notes: Optional[str] = Field(description="Use null if none.")


class HarmonizationJSON(StrictBaseModel):
    harmonized_determinants: List[HarmonizedDeterminant]


# ==============================================================================
# 4. JSON SCHEMA SANITIZATION FOR OPENAI STRUCTURED OUTPUTS
# ==============================================================================

def sanitize_schema_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    def _sanitize(node: Any) -> Any:
        if isinstance(node, dict):
            node = dict(node)
            node.pop("default", None)

            if "$defs" in node and isinstance(node["$defs"], dict):
                node["$defs"] = {k: _sanitize(v) for k, v in node["$defs"].items()}

            for key in ["anyOf", "oneOf", "allOf"]:
                if key in node and isinstance(node[key], list):
                    node[key] = [_sanitize(v) for v in node[key]]

            if "items" in node:
                node["items"] = _sanitize(node["items"])

            if "properties" in node and isinstance(node["properties"], dict):
                node["properties"] = {
                    k: _sanitize(v)
                    for k, v in node["properties"].items()
                }
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False

            if node.get("type") == "object":
                node["additionalProperties"] = False
                if "properties" not in node:
                    node["properties"] = {}
                    node["required"] = []

            return node

        if isinstance(node, list):
            return [_sanitize(v) for v in node]

        return node

    return _sanitize(schema)


def pydantic_schema(model_class: Any) -> Dict[str, Any]:
    raw_schema = model_class.model_json_schema()
    clean_schema = sanitize_schema_for_openai(raw_schema)

    return {
        "type": "json_schema",
        "name": model_class.__name__,
        "schema": clean_schema,
        "strict": True,
    }


SCREENING_SCHEMA = pydantic_schema(ScreeningJSON)
EXTRACTION_SCHEMA = pydantic_schema(PaperExtractionJSON)
VERIFICATION_SCHEMA = pydantic_schema(PaperVerificationJSON)
HARMONIZATION_SCHEMA = pydantic_schema(HarmonizationJSON)


if SAVE_SCHEMA_DEBUG:
    with open(LOG_DIR / "debug_screening_schema.json", "w", encoding="utf-8") as f:
        json.dump(SCREENING_SCHEMA, f, ensure_ascii=False, indent=2)
    with open(LOG_DIR / "debug_extraction_schema.json", "w", encoding="utf-8") as f:
        json.dump(EXTRACTION_SCHEMA, f, ensure_ascii=False, indent=2)
    with open(LOG_DIR / "debug_verification_schema.json", "w", encoding="utf-8") as f:
        json.dump(VERIFICATION_SCHEMA, f, ensure_ascii=False, indent=2)
    with open(LOG_DIR / "debug_harmonization_schema.json", "w", encoding="utf-8") as f:
        json.dump(HARMONIZATION_SCHEMA, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================

def slugify(text: str, max_len: int = 90) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        corrupt_path = path.with_suffix(path.suffix + f".corrupt_{int(time.time())}")
        path.rename(corrupt_path)
        log(f"Warning: corrupt JSON moved to {corrupt_path}: {e}")
        return None


def save_json(obj: Any, path: Path) -> None:
    """
    Atomic JSON save to avoid partial files when running in parallel.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + f".tmp_{threading.get_ident()}")

    with open(tmp_path, "w", encoding="utf-8") as f:
        if isinstance(obj, BaseModel):
            json.dump(obj.model_dump(), f, ensure_ascii=False, indent=2)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def list_pdf_files(literature_dir: Path) -> List[Path]:
    if not literature_dir.exists():
        raise FileNotFoundError(f"Folder does not exist: {literature_dir}")

    return sorted([
        p for p in literature_dir.rglob("*.pdf")
        if p.is_file()
        and "_ai_carbon_tax_acceptability_review" not in str(p)
        and "_ai_carbon_tax_strict_review" not in str(p)
    ])


def load_file_cache_unlocked() -> Dict[str, Any]:
    data = load_json(FILE_CACHE_PATH)
    return data if isinstance(data, dict) else {}


def save_file_cache_unlocked(cache: Dict[str, Any]) -> None:
    save_json(cache, FILE_CACHE_PATH)


def upload_pdf_to_openai(pdf_path: Path) -> str:
    """
    Upload PDF to OpenAI and return file_id.

    The file cache is protected by a lock.
    Uploads are intentionally serialized through FILE_CACHE_LOCK to avoid
    duplicate uploads and cache corruption.
    """
    file_hash = file_sha256(pdf_path)
    cache_key = f"{pdf_path.name}::{file_hash}"

    if not USE_FILE_CACHE:
        log(f"Uploading PDF to OpenAI: {pdf_path.name}")
        with open(pdf_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
        return uploaded.id

    with FILE_CACHE_LOCK:
        cache = load_file_cache_unlocked()

        if cache_key in cache:
            cached_file_id = cache[cache_key].get("file_id")
            if cached_file_id:
                log(f"Using cached OpenAI file_id for: {pdf_path.name}")
                return cached_file_id

        log(f"Uploading PDF to OpenAI: {pdf_path.name}")

        with open(pdf_path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="user_data",
            )

        file_id = uploaded.id

        cache[cache_key] = {
            "file_id": file_id,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "sha256": file_hash,
            "uploaded_at_unix": time.time(),
        }

        save_file_cache_unlocked(cache)

        return file_id


def extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []

    try:
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content in item.content:
                    ctype = getattr(content, "type", None)

                    if ctype in ["output_text", "text"]:
                        text = getattr(content, "text", None)
                        if text:
                            chunks.append(text)
    except Exception:
        pass

    if chunks:
        return "\n".join(chunks)

    raise ValueError("Could not extract text from response.")


def save_raw_response_debug(
    *,
    response: Any,
    text: Optional[str],
    stage_label: str,
    attempt: int,
) -> None:
    debug_path = (
        LOG_DIR
        / f"debug_raw_response_{stage_label}_attempt_{attempt}_{int(time.time())}.json"
    )

    payload: Dict[str, Any] = {
        "stage_label": stage_label,
        "attempt": attempt,
        "text": text,
        "text_length": len(text) if text is not None else None,
        "response_repr": repr(response),
    }

    try:
        if hasattr(response, "model_dump"):
            payload["response_model_dump"] = response.model_dump()
    except Exception as e:
        payload["response_model_dump_error"] = repr(e)

    save_json(payload, debug_path)
    log(f"Saved raw response debug file: {debug_path}")


def is_likely_truncation_or_empty_output_error(error_text: str) -> bool:
    patterns = [
        "Unterminated string",
        "JSON parsing failed",
        "Could not extract text from response",
        "Expecting value",
        "Expecting ',' delimiter",
        "Expecting property name",
        "response incomplete",
        "OpenAI response incomplete",
        "max_output_tokens",
        "incomplete_details",
    ]

    return any(p in error_text for p in patterns)


def call_openai_with_pdf(
    *,
    pdf_file_id: str,
    prompt: str,
    model: str,
    json_schema: Dict[str, Any],
    reasoning_effort: str,
    max_output_tokens: int,
    stage_label: str,
) -> Dict[str, Any]:
    last_error = None
    current_max_output_tokens = max_output_tokens

    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        text = None

        try:
            log(
                f"Calling OpenAI [{stage_label}] "
                f"model={model}, reasoning={reasoning_effort}, "
                f"max_output_tokens={current_max_output_tokens}"
            )

            with API_SEMAPHORE:
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning_effort},
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": pdf_file_id,
                                },
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    text={"format": json_schema},
                    max_output_tokens=current_max_output_tokens,
                )

            if getattr(response, "status", None) == "incomplete":
                incomplete_details = getattr(response, "incomplete_details", None)
                reason = getattr(incomplete_details, "reason", None)

                save_raw_response_debug(
                    response=response,
                    text=None,
                    stage_label=stage_label,
                    attempt=attempt,
                )

                raise ValueError(
                    f"OpenAI response incomplete at stage={stage_label}, "
                    f"attempt={attempt}, reason={reason}"
                )

            text = extract_response_text(response)

            try:
                return json.loads(text)

            except json.JSONDecodeError as e:
                save_raw_response_debug(
                    response=response,
                    text=text,
                    stage_label=stage_label,
                    attempt=attempt,
                )

                preview = text[:1000] if text else ""
                tail = text[-1000:] if text else ""

                raise ValueError(
                    f"JSON parsing failed at stage={stage_label}, attempt={attempt}. "
                    f"Error={repr(e)}. "
                    f"Text length={len(text) if text else 0}. "
                    f"Preview={preview!r}. "
                    f"Tail={tail!r}."
                )

        except Exception as e:
            last_error = e
            error_text = repr(e)

            if (
                response is not None
                and text is None
                and "OpenAI response incomplete" not in error_text
            ):
                save_raw_response_debug(
                    response=response,
                    text=None,
                    stage_label=stage_label,
                    attempt=attempt,
                )

            if is_likely_truncation_or_empty_output_error(error_text):
                current_max_output_tokens = min(
                    int(current_max_output_tokens * TOKEN_RETRY_MULTIPLIER),
                    MAX_TOKEN_RETRY_CEILING,
                )
                log(
                    "Likely truncated or incomplete output. "
                    f"Increasing max_output_tokens to {current_max_output_tokens}."
                )

            wait = 2 ** attempt
            log(f"[Attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            log(f"Waiting {wait} seconds before retrying...")
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI call failed after {MAX_RETRIES} attempts "
        f"at stage={stage_label}: {last_error}"
    )


def call_openai_text_only(
    *,
    prompt: str,
    model: str,
    json_schema: Dict[str, Any],
    reasoning_effort: str,
    max_output_tokens: int,
    stage_label: str,
) -> Dict[str, Any]:
    last_error = None
    current_max_output_tokens = max_output_tokens

    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        text = None

        try:
            log(
                f"Calling OpenAI [{stage_label}] "
                f"model={model}, reasoning={reasoning_effort}, "
                f"max_output_tokens={current_max_output_tokens}"
            )

            with API_SEMAPHORE:
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning_effort},
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                    text={"format": json_schema},
                    max_output_tokens=current_max_output_tokens,
                )

            if getattr(response, "status", None) == "incomplete":
                incomplete_details = getattr(response, "incomplete_details", None)
                reason = getattr(incomplete_details, "reason", None)

                save_raw_response_debug(
                    response=response,
                    text=None,
                    stage_label=stage_label,
                    attempt=attempt,
                )

                raise ValueError(
                    f"OpenAI response incomplete at stage={stage_label}, "
                    f"attempt={attempt}, reason={reason}"
                )

            text = extract_response_text(response)

            try:
                return json.loads(text)

            except json.JSONDecodeError as e:
                save_raw_response_debug(
                    response=response,
                    text=text,
                    stage_label=stage_label,
                    attempt=attempt,
                )

                raise ValueError(
                    f"JSON parsing failed at stage={stage_label}, attempt={attempt}. "
                    f"Error={repr(e)}. Text length={len(text) if text else 0}."
                )

        except Exception as e:
            last_error = e
            error_text = repr(e)

            if (
                response is not None
                and text is None
                and "OpenAI response incomplete" not in error_text
            ):
                save_raw_response_debug(
                    response=response,
                    text=None,
                    stage_label=stage_label,
                    attempt=attempt,
                )

            if is_likely_truncation_or_empty_output_error(error_text):
                current_max_output_tokens = min(
                    int(current_max_output_tokens * TOKEN_RETRY_MULTIPLIER),
                    MAX_TOKEN_RETRY_CEILING,
                )
                log(
                    "Likely truncated or incomplete output. "
                    f"Increasing max_output_tokens to {current_max_output_tokens}."
                )

            wait = 2 ** attempt
            log(f"[Attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            log(f"Waiting {wait} seconds before retrying...")
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI text-only call failed after {MAX_RETRIES} attempts "
        f"at stage={stage_label}: {last_error}"
    )


def validate_or_save_error(
    data: Dict[str, Any],
    model_class: Any,
    error_path: Path,
) -> BaseModel:
    try:
        return model_class.model_validate(data)

    except ValidationError as e:
        save_json(
            {
                "validation_error": e.errors(),
                "raw_data": data,
            },
            error_path,
        )
        raise


def safe_bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(
        lambda x: True if x is True or str(x).lower() == "true" else False
    )


def join_unique(values: pd.Series, sep: str = " ; ") -> str:
    cleaned = sorted({
        str(v).strip()
        for v in values
        if pd.notna(v) and str(v).strip() != ""
    })
    return sep.join(cleaned)

def clean_excel_illegal_chars(value: Any) -> Any:
    """
    Remove characters that are illegal in Excel XML files.

    Excel .xlsx files are XML documents. They cannot contain NULL bytes,
    most ASCII control characters, unpaired surrogate characters, or Unicode
    code points outside the XML 1.0 valid character ranges.
    """
    if not isinstance(value, str):
        return value

    # Drop malformed surrogate sequences if they appear in model output.
    value = value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    cleaned_chars = []
    for ch in value:
        code = ord(ch)

        if (
            code == 0x09
            or code == 0x0A
            or code == 0x0D
            or 0x20 <= code <= 0xD7FF
            or 0xE000 <= code <= 0xFFFD
            or 0x10000 <= code <= 0x10FFFF
        ):
            cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)

    # Excel has a hard 32,767-character limit per cell.
    return cleaned[:32767]


def clean_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a DataFrame with Excel-incompatible characters removed.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    return out.map(clean_excel_illegal_chars)


def save_dataframe_csv_and_xlsx(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path,
) -> None:
    """
    Save a DataFrame to CSV and XLSX after removing XML-incompatible characters.
    """
    clean_df = clean_dataframe_for_excel(df)
    clean_df.to_csv(csv_path, index=False)
    clean_df.to_excel(xlsx_path, index=False)

# ==============================================================================
# 6. STATUS MANAGEMENT
# ==============================================================================

def status_path_for_paper(paper_id: str) -> Path:
    return STATUS_DIR / f"{paper_id}.json"


def load_status(paper_id: str) -> Optional[Dict[str, Any]]:
    return load_json(status_path_for_paper(paper_id))


def save_status(
    *,
    paper_id: str,
    pdf_path: Path,
    status: str,
    stage: str,
    reason: Optional[str] = None,
) -> None:
    save_json(
        {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "status": status,
            "stage": stage,
            "reason": reason,
            "updated_at_unix": time.time(),
        },
        status_path_for_paper(paper_id),
    )


# ---------------- New helper functions for robust status management ----------------

def status_age_hours(status: Optional[Dict[str, Any]]) -> Optional[float]:
    if not status:
        return None

    updated_at = status.get("updated_at_unix")
    if updated_at is None:
        return None

    try:
        return (time.time() - float(updated_at)) / 3600.0
    except Exception:
        return None


def is_stale_running_status(status: Optional[Dict[str, Any]]) -> bool:
    if not status:
        return False

    if status.get("status") != "running":
        return False

    age = status_age_hours(status)
    if age is None:
        return True

    return age >= STALE_RUNNING_AFTER_HOURS


def should_retry_status(status: Optional[Dict[str, Any]]) -> bool:
    if not status:
        return True

    status_value = status.get("status")

    if status_value in ["failed", "interrupted"]:
        return RETRY_FAILED_OR_INTERRUPTED

    if status_value == "running":
        return is_stale_running_status(status)

    return False


def is_valid_extraction_json(data: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(data, dict):
        return False

    if not isinstance(data.get("paper_metadata"), dict):
        return False

    if not isinstance(data.get("determinants"), list):
        return False

    return True


def final_json_is_valid(final_path: Path) -> bool:
    data = load_json(final_path)
    return is_valid_extraction_json(data)


def is_completed_status(status: Optional[Dict[str, Any]]) -> bool:
    if not status:
        return False

    return status.get("status") in [
        "completed_included",
        "completed_excluded_screening",
        "completed_excluded_verification",
    ]


# ==============================================================================
# 7. PROMPTS
# ==============================================================================

def build_screening_prompt(pdf_name: str, paper_id: str) -> str:
    return f"""
You are screening an academic paper for inclusion in a strict literature review.

The review is ONLY about determinants of public acceptability, acceptance,
support, opposition, vote choice, willingness to pay, or preferences toward
CARBON TAXES and FUEL TAXES.

The relevant policy object is a TAX instrument similar to the French TICPE or
a carbon component of fuel/energy taxation.

Do not include a paper only because it contains a generic “taxes” category.
The tax must be explicitly a carbon tax, CO2 tax, fossil-fuel tax, gasoline tax,
diesel tax, transport fuel tax, fuel excise tax, or a clearly equivalent domestic
carbon/fuel tax.

Carbon border adjustment, carbon border tax, import tariff, or border measure
should be excluded unless the paper also analyzes a domestic carbon/fuel tax.

If the policy item is only “taxes” without enough information to determine that
it is a carbon/fuel tax, use uncertain_needs_manual_review or exclude.

INCLUDE the paper if and only if it contains a direct analysis of public
acceptability/support/preferences for at least one of:
- carbon tax
- CO2 tax
- fuel tax
- fossil fuel tax
- gasoline/petrol/diesel tax
- transport fuel excise tax
- energy tax explicitly based on carbon or fossil fuels
- TICPE-like tax
- progressive carbon tax
- carbon tax with revenue recycling or lump-sum transfers
- fuel tax increase studied as a climate/carbon policy

EXCLUDE if it only studies:
- ETS, cap-and-trade, emission permits, carbon markets
- carbon pricing in general without separate tax-specific analysis
- subsidies, regulations, green spending
- willingness to donate/contribute without a tax instrument
- pro-environmental behavior without tax acceptability
- general climate policy support without a carbon/fuel tax item

If the paper discusses both carbon tax and ETS, INCLUDE it, but later only the
carbon/fuel-tax evidence will be extracted.

Output-length rules:
1. tax_mentions: maximum 3 items.
2. excluded_policy_mentions: maximum 3 items.
3. context: maximum 30 words.
4. notes: maximum 20 words or null.
5. reason_it_is_not_a_carbon_or_fuel_tax: maximum 25 words.
6. inclusion_reason: maximum 40 words.
7. exclusion_reason: maximum 40 words or null.
8. Do not write a long summary.
9. Return compact valid JSON only.

File name:
{pdf_name}

Use this paper_id:
{paper_id}

Allowed screening_decision values:
{SCREENING_DECISIONS}

Allowed tax_instrument_type values:
{TAX_INSTRUMENT_TYPES}

Allowed excluded_policy_type values:
{EXCLUDED_POLICY_TYPES}

Allowed paper_type values:
{PAPER_TYPES}
"""


def build_extraction_prompt(
    pdf_name: str,
    paper_id: str,
    screening_json: Dict[str, Any],
) -> str:
    screening_str = json.dumps(screening_json, ensure_ascii=False, indent=2)

    return f"""
You are an expert research assistant in environmental economics, public
economics, political economy, and carbon-tax acceptability.

You are analyzing one academic paper provided as a full PDF.

This paper passed screening as containing direct evidence on a carbon tax or
fuel tax.

Screening JSON:
{screening_str}

Your task:
Extract ONLY determinants directly associated with public acceptability,
acceptance, support, opposition, vote choice, willingness to pay, policy
preference, or attitudes toward a CARBON TAX or FUEL TAX.

Relevant instruments:
- carbon tax
- CO2 tax
- fuel tax
- gasoline/petrol/diesel tax
- fossil fuel tax
- energy tax explicitly based on carbon/fossil fuels
- TICPE-like excise fuel tax
- progressive carbon tax
- carbon tax with revenue recycling or dividends

Do NOT extract determinants only associated with:
- ETS, cap-and-trade, permits, carbon markets
- carbon pricing in general without a tax-specific result
- subsidies, regulations, green spending
- donation/contribution without a tax
- pro-environmental behavior without tax support
- general climate policy support unless there is a separate carbon/fuel tax item

Rules:
1. Extract at most 20 determinants.
2. Prioritize substantive determinants and tax-policy design features.
3. Include demographic or political controls only if explicitly reported for a
   carbon/fuel-tax outcome.
4. Do not extract methodological variables.
5. Do not invent signs.
6. Use direction_on_tax_acceptability:
   - "+" if determinant increases tax support/acceptability;
   - "-" if it decreases it;
   - "mixed" if heterogeneous or specification-dependent;
   - "n.a." if no clear direction or non-significant.
7. Keep fields concise:
   - definition: max 35 words;
   - mechanism: max 40 words;
   - evidence_quote: max 35 words;
   - statistical_or_result_detail: max 35 words or null;
   - notes: max 35 words or null.
8. Use null for unavailable nullable fields.
9. Return compact valid JSON only.

File name:
{pdf_name}

paper_id:
{paper_id}

Allowed tax_instrument_type values:
{TAX_INSTRUMENT_TYPES}

Allowed paper_type values:
{PAPER_TYPES}

Allowed determinant_family values:
{DETERMINANT_FAMILIES}

Allowed determinant_role values:
{DETERMINANT_ROLES}

Allowed evidence_type values:
{EVIDENCE_TYPES}

Allowed inclusion values:
{INCLUSION_IN_MAIN_TABLE_VALUES}
"""


def build_verification_prompt(
    pdf_name: str,
    paper_id: str,
    screening_json: Dict[str, Any],
    extraction_json: Dict[str, Any],
) -> str:
    screening_str = json.dumps(screening_json, ensure_ascii=False, indent=2)
    extraction_str = json.dumps(extraction_json, ensure_ascii=False, indent=2)

    return f"""
You are a critical reviewer in environmental economics and political economy.

You are given:
1. The full PDF.
2. The screening JSON.
3. The extraction JSON.

Your task is to verify whether the extraction includes ONLY determinants directly
associated with carbon-tax or fuel-tax acceptability/support/opposition.

Screening JSON:
{screening_str}

Extraction JSON:
{extraction_str}

Verification rules:
1. Remove every determinant not directly linked to a carbon/fuel-tax outcome.
2. Remove determinants only concerning ETS, cap-and-trade, subsidies,
   regulations, green spending, general climate action, or private behavior.
3. Keep determinants for carbon/fuel-tax outcomes even if the paper also studies
   other policies.
4. Check whether the paper should remain in the strict tax corpus.
5. Check title, DOI, year, paper type, and tax scope.
6. Check signs and directions.
7. Identify missing carbon/fuel-tax determinants.
8. Mark demographic/political controls as controls.
9. Do NOT reproduce the full extraction JSON. Verification is an audit step only.
10. Provide only determinant checks, missing determinants, and a concise final_recommendation_summary.
11. Keep comments concise:
    - determinant check comment: max 35 words;
    - suggested_correction: max 35 words or null;
    - metadata comments: max 50 words or null.
12. Use null for unavailable nullable fields.
13. Return compact valid JSON only.

File name:
{pdf_name}

paper_id:
{paper_id}

Allowed tax_instrument_type values:
{TAX_INSTRUMENT_TYPES}

Allowed determinant_family values:
{DETERMINANT_FAMILIES}

Allowed determinant_role values:
{DETERMINANT_ROLES}

Allowed evidence_type values:
{EVIDENCE_TYPES}

Allowed inclusion values:
{INCLUSION_IN_MAIN_TABLE_VALUES}
"""


def build_harmonization_prompt(long_table_records: List[Dict[str, Any]]) -> str:
    records_str = json.dumps(long_table_records, ensure_ascii=False, indent=2)

    return f"""
You are harmonizing determinant labels for a strict literature review on
carbon-tax and fuel-tax acceptability.

The determinants are already restricted to tax outcomes.

Rules:
1. Preserve meaningful conceptual distinctions.
2. Keep perceived effectiveness distinct from trust.
3. Keep fairness distinct from economic self-interest.
4. Keep revenue recycling distinct from earmarking if justified.
5. Keep ideology/partisanship distinct from trust if possible.
6. Keep demographic controls identifiable.
7. Do not merge experimental treatments with general psychological constructs if
   that would obscure the evidence.
8. Keep definitions concise, max 35 words.
9. Return compact valid JSON only.

Allowed determinant_family values:
{DETERMINANT_FAMILIES}

Raw records:
{records_str}
"""


# ==============================================================================
# 8. PROCESS ONE PAPER
# ==============================================================================

# ==============================================================================
# 8. PROCESS ONE PAPER
# ==============================================================================

def process_one_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    paper_id = slugify(pdf_path.stem)

    screening_path = SCREENING_DIR / f"{paper_id}.json"
    extraction_path = EXTRACTION_DIR / f"{paper_id}.json"
    verification_path = VERIFICATION_DIR / f"{paper_id}.json"
    final_path = FINAL_JSON_DIR / f"{paper_id}.json"

    existing_status = load_status(paper_id)

    # ------------------------------------------------------------------
    # Robust resume logic
    # ------------------------------------------------------------------
    if is_completed_status(existing_status) and not OVERWRITE_EXISTING:
        status_value = existing_status.get("status")

        if status_value == "completed_included":
            if final_json_is_valid(final_path):
                log(f"Skipping already completed paper: {paper_id} status={status_value}")
                return load_json(final_path)

            log(
                f"Completed status found but final JSON is missing/corrupt/invalid. "
                f"Retrying paper: {paper_id}"
            )

        elif status_value in ["completed_excluded_screening", "completed_excluded_verification"]:
            log(f"Skipping already completed paper: {paper_id} status={status_value}")
            return None

    elif existing_status is not None and not should_retry_status(existing_status) and not OVERWRITE_EXISTING:
        status_value = existing_status.get("status")
        log(
            f"Status is not completed but not retryable yet: {paper_id} "
            f"status={status_value}. Skipping for this run."
        )
        return None

    # If final JSON exists and is valid, it takes precedence even if the status file
    # is missing or stale.
    if final_path.exists() and not OVERWRITE_EXISTING:
        final_data = load_json(final_path)

        if is_valid_extraction_json(final_data):
            log(f"Skipping existing valid final JSON: {paper_id}")
            save_status(
                paper_id=paper_id,
                pdf_path=pdf_path,
                status="completed_included",
                stage="final",
                reason=None,
            )
            return final_data

        log(f"Existing final JSON is invalid or corrupt. Retrying paper: {paper_id}")

    log("\n" + "-" * 80)
    log(f"Processing: {pdf_path.name}")
    log(f"paper_id: {paper_id}")

    save_status(
        paper_id=paper_id,
        pdf_path=pdf_path,
        status="running",
        stage="upload",
        reason=None,
    )

    file_id = upload_pdf_to_openai(pdf_path)

    # ----------------------------------------------------------------------
    # STEP 1: SCREENING
    # ----------------------------------------------------------------------
    save_status(
        paper_id=paper_id,
        pdf_path=pdf_path,
        status="running",
        stage="screening",
        reason=None,
    )

    screening_data = load_json(screening_path) if not OVERWRITE_EXISTING else None

    if screening_data is not None:
        try:
            ScreeningJSON.model_validate(screening_data)
            log(f"Using existing screening JSON: {paper_id}")
        except ValidationError:
            log(f"Existing screening JSON is invalid. Re-running screening: {paper_id}")
            screening_data = None

    if screening_data is None:
        prompt = build_screening_prompt(
            pdf_name=pdf_path.name,
            paper_id=paper_id,
        )

        raw = call_openai_with_pdf(
            pdf_file_id=file_id,
            prompt=prompt,
            model=MODEL_SCREEN,
            json_schema=SCREENING_SCHEMA,
            reasoning_effort=REASONING_EFFORT_SCREEN,
            max_output_tokens=MAX_OUTPUT_TOKENS_SCREEN,
            stage_label=f"{paper_id}_screening",
        )

        model = validate_or_save_error(
            raw,
            ScreeningJSON,
            LOG_DIR / f"{paper_id}_screening_validation_error.json",
        )

        screening_data = model.model_dump()
        save_json(screening_data, screening_path)

    time.sleep(SLEEP_BETWEEN_CALLS)

    keep = (
        screening_data.get("screening_decision") == "include_direct_carbon_or_fuel_tax"
        and screening_data.get("contains_direct_carbon_or_fuel_tax") is True
        and screening_data.get("contains_tax_acceptability_or_support_analysis") is True
    )

    if not keep:
        reason = screening_data.get("exclusion_reason")
        log(f"Excluded after screening: {paper_id}")
        log(f"Reason: {reason}")

        save_status(
            paper_id=paper_id,
            pdf_path=pdf_path,
            status="completed_excluded_screening",
            stage="screening",
            reason=reason,
        )

        return None

    # ----------------------------------------------------------------------
    # STEP 2: EXTRACTION
    # ----------------------------------------------------------------------
    save_status(
        paper_id=paper_id,
        pdf_path=pdf_path,
        status="running",
        stage="extraction",
        reason=None,
    )

    extraction_data = load_json(extraction_path) if not OVERWRITE_EXISTING else None

    if extraction_data is not None:
        try:
            PaperExtractionJSON.model_validate(extraction_data)
            log(f"Using existing extraction JSON: {paper_id}")
        except ValidationError:
            log(f"Existing extraction JSON is invalid. Re-running extraction: {paper_id}")
            extraction_data = None

    if extraction_data is None:
        prompt = build_extraction_prompt(
            pdf_name=pdf_path.name,
            paper_id=paper_id,
            screening_json=screening_data,
        )

        raw = call_openai_with_pdf(
            pdf_file_id=file_id,
            prompt=prompt,
            model=MODEL_EXTRACT,
            json_schema=EXTRACTION_SCHEMA,
            reasoning_effort=REASONING_EFFORT_EXTRACT,
            max_output_tokens=MAX_OUTPUT_TOKENS_EXTRACT,
            stage_label=f"{paper_id}_extraction",
        )

        model = validate_or_save_error(
            raw,
            PaperExtractionJSON,
            LOG_DIR / f"{paper_id}_extraction_validation_error.json",
        )

        extraction_data = model.model_dump()
        save_json(extraction_data, extraction_path)

    time.sleep(SLEEP_BETWEEN_CALLS)

    # ----------------------------------------------------------------------
    # STEP 3: VERIFICATION, lightweight audit only
    # ----------------------------------------------------------------------
    save_status(
        paper_id=paper_id,
        pdf_path=pdf_path,
        status="running",
        stage="verification",
        reason=None,
    )

    verification_data = load_json(verification_path) if not OVERWRITE_EXISTING else None

    if verification_data is not None:
        try:
            PaperVerificationJSON.model_validate(verification_data)
            log(f"Using existing verification JSON: {paper_id}")
        except ValidationError:
            log(f"Existing verification JSON is invalid or uses the old heavy schema. Re-running verification: {paper_id}")
            verification_data = None

    if verification_data is None:
        prompt = build_verification_prompt(
            pdf_name=pdf_path.name,
            paper_id=paper_id,
            screening_json=screening_data,
            extraction_json=extraction_data,
        )

        raw = call_openai_with_pdf(
            pdf_file_id=file_id,
            prompt=prompt,
            model=MODEL_VERIFY,
            json_schema=VERIFICATION_SCHEMA,
            reasoning_effort=REASONING_EFFORT_VERIFY,
            max_output_tokens=MAX_OUTPUT_TOKENS_VERIFY,
            stage_label=f"{paper_id}_verification",
        )

        model = validate_or_save_error(
            raw,
            PaperVerificationJSON,
            LOG_DIR / f"{paper_id}_verification_validation_error.json",
        )

        verification_data = model.model_dump()
        save_json(verification_data, verification_path)

    if not verification_data.get("should_remain_in_corpus", True):
        reason = verification_data.get("reason_for_exclusion_if_any")
        log(f"Excluded after verification: {paper_id}")
        log(f"Reason: {reason}")

        save_status(
            paper_id=paper_id,
            pdf_path=pdf_path,
            status="completed_excluded_verification",
            stage="verification",
            reason=reason,
        )

        return None

    # ----------------------------------------------------------------------
    # STEP 4: FINAL JSON
    # ----------------------------------------------------------------------
    # The final JSON is intentionally the extraction JSON.
    # Verification is preserved as a separate audit file and no longer rewrites
    # the full extraction, which avoids long truncated JSON responses.
    final_data = extraction_data

    save_json(final_data, final_path)

    save_status(
        paper_id=paper_id,
        pdf_path=pdf_path,
        status="completed_included",
        stage="final",
        reason=None,
    )

    return final_data


# ==============================================================================
# 9. BUILD TABLES
# ==============================================================================

def load_all_final_jsons() -> List[Dict[str, Any]]:
    final_jsons = []

    for path in sorted(FINAL_JSON_DIR.glob("*.json")):
        data = load_json(path)
        if data is not None:
            final_jsons.append(data)

    return final_jsons


def load_all_screening_jsons() -> List[Dict[str, Any]]:
    out = []

    for path in sorted(SCREENING_DIR.glob("*.json")):
        data = load_json(path)
        if data is not None:
            out.append(data)

    return out


def load_all_status_jsons() -> List[Dict[str, Any]]:
    out = []

    for path in sorted(STATUS_DIR.glob("*.json")):
        data = load_json(path)
        if data is not None:
            out.append(data)

    return out


def build_status_table(status_jsons: List[Dict[str, Any]]) -> pd.DataFrame:
    if not status_jsons:
        return pd.DataFrame(
            columns=[
                "paper_id",
                "pdf_path",
                "status",
                "stage",
                "reason",
                "updated_at_unix",
            ]
        )

    return pd.DataFrame(status_jsons)


def build_screening_table(screening_jsons: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for s in screening_jsons:
        tax_terms = []
        for t in s.get("tax_mentions", []):
            tax_terms.append(
                f"{t.get('exact_term_used_in_paper')} "
                f"[{t.get('tax_instrument_type')}; "
                f"direct_analysis={t.get('is_direct_acceptability_or_support_analysis')}]"
            )

        excluded_terms = []
        for e in s.get("excluded_policy_mentions", []):
            excluded_terms.append(
                f"{e.get('exact_term_used_in_paper')} "
                f"[{e.get('excluded_policy_type')}]"
            )

        rows.append({
            "paper_id": s.get("paper_id"),
            "title": s.get("title"),
            "authors": s.get("authors"),
            "doi": s.get("doi"),
            "publication_year": s.get("publication_year"),
            "paper_type": s.get("paper_type"),
            "screening_decision": s.get("screening_decision"),
            "contains_direct_carbon_or_fuel_tax": s.get("contains_direct_carbon_or_fuel_tax"),
            "contains_tax_acceptability_or_support_analysis": s.get("contains_tax_acceptability_or_support_analysis"),
            "confidence": s.get("confidence"),
            "inclusion_reason": s.get("inclusion_reason"),
            "exclusion_reason": s.get("exclusion_reason"),
            "tax_mentions": " ; ".join(tax_terms),
            "excluded_policy_mentions": " ; ".join(excluded_terms),
        })

    return pd.DataFrame(rows)


def build_paper_level_table(final_jsons: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for data in final_jsons:
        meta = data.get("paper_metadata", {})
        determinants = data.get("determinants", [])

        main_dets = [
            d for d in determinants
            if d.get("include_in_main_table") in ["yes_main_result", "yes_secondary_result"]
        ]

        det_strings = []
        for d in determinants:
            det_strings.append(
                f"{d.get('raw_name_in_paper')} "
                f"({d.get('direction_on_tax_acceptability')}) "
                f"[{d.get('tax_instrument_type')}; "
                f"role={d.get('determinant_role')}; "
                f"main={d.get('include_in_main_table')}]"
            )

        rows.append({
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "doi": meta.get("doi"),
            "publication_year": meta.get("publication_year"),
            "publication_date": meta.get("publication_date"),
            "journal_or_source": meta.get("journal_or_source"),
            "paper_type": meta.get("paper_type"),
            "geographic_scope": meta.get("geographic_scope"),
            "empirical_scope_or_sample": meta.get("empirical_scope_or_sample"),
            "tax_instruments_analyzed": "; ".join(meta.get("tax_instruments_analyzed", [])),
            "exact_tax_terms_used": "; ".join(meta.get("exact_tax_terms_used", [])),
            "carbon_tax_relevance_summary": meta.get("carbon_tax_relevance_summary"),
            "n_tax_determinants": len(determinants),
            "n_main_tax_determinants": len(main_dets),
            "determinants": " ; ".join(det_strings),
        })

    return pd.DataFrame(rows)


def build_long_determinant_table(final_jsons: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for data in final_jsons:
        meta = data.get("paper_metadata", {})
        determinants = data.get("determinants", [])

        for i, d in enumerate(determinants, start=1):
            rows.append({
                "paper_id": meta.get("paper_id"),
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "doi": meta.get("doi"),
                "publication_year": meta.get("publication_year"),
                "paper_type": meta.get("paper_type"),
                "geographic_scope": meta.get("geographic_scope"),
                "determinant_index": i,
                "raw_determinant": d.get("raw_name_in_paper"),
                "standardized_candidate_label": d.get("standardized_candidate_label"),
                "determinant_family": d.get("determinant_family"),
                "determinant_role": d.get("determinant_role"),
                "direction": d.get("direction_on_tax_acceptability"),
                "definition": d.get("definition"),
                "mechanism": d.get("mechanism"),
                "evidence_type": d.get("evidence_type"),
                "tax_instrument_type": d.get("tax_instrument_type"),
                "exact_tax_policy_context": d.get("exact_tax_policy_context"),
                "outcome_variable": d.get("outcome_variable"),
                "acceptability_dimension": d.get("acceptability_dimension"),
                "is_directly_about_carbon_or_fuel_tax": d.get("is_directly_about_carbon_or_fuel_tax"),
                "is_exclusively_about_ets_or_non_tax_policy": d.get("is_exclusively_about_ets_or_non_tax_policy"),
                "evidence_quote": d.get("evidence_quote"),
                "page_or_section": d.get("page_or_section"),
                "statistical_or_result_detail": d.get("statistical_or_result_detail"),
                "confidence": d.get("confidence"),
                "include_in_main_table": d.get("include_in_main_table"),
                "exclusion_reason_if_not_main": d.get("exclusion_reason_if_not_main"),
                "notes": d.get("notes"),
            })

    return pd.DataFrame(rows)


def build_verification_audit_table() -> pd.DataFrame:
    rows = []

    for path in sorted(VERIFICATION_DIR.glob("*.json")):
        v = load_json(path)
        if not v:
            continue

        checks = v.get("determinant_checks", [])
        status_counts = {}

        for c in checks:
            status = c.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        rows.append({
            "paper_id": v.get("paper_id"),
            "overall_verdict": v.get("overall_verdict"),
            "should_remain_in_corpus": v.get("should_remain_in_corpus"),
            "reason_for_exclusion_if_any": v.get("reason_for_exclusion_if_any"),
            "n_determinant_checks": len(checks),
            "n_missing_tax_determinants": len(v.get("missing_tax_determinants", [])),
            "status_counts": json.dumps(status_counts, ensure_ascii=False),
            "metadata_comments": v.get("metadata_checks", {}).get("comments"),
            "final_recommendation_summary": v.get("final_recommendation_summary"),
        })

    return pd.DataFrame(rows)


def save_core_tables(
    screening_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    long_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> None:
    save_dataframe_csv_and_xlsx(
    paper_df,
    TABLE_DIR / "table_1_paper_level_tax_only.csv",
    TABLE_DIR / "table_1_paper_level_tax_only.xlsx",
    )

    save_dataframe_csv_and_xlsx(
    screening_df,
    TABLE_DIR / "screening_decisions.csv",
    TABLE_DIR / "screening_decisions.xlsx",
    )

    save_dataframe_csv_and_xlsx(
        long_df,
        TABLE_DIR / "determinants_long_format_tax_only.csv",
        TABLE_DIR / "determinants_long_format_tax_only.xlsx",
    )

    save_dataframe_csv_and_xlsx(
        audit_df,
        TABLE_DIR / "verification_audit_report.csv",
        TABLE_DIR / "verification_audit_report.xlsx",
    )

    save_dataframe_csv_and_xlsx(
        status_df,
        TABLE_DIR / "pipeline_status.csv",
        TABLE_DIR / "pipeline_status.xlsx",
    )

    log(f"Saved core tables in: {TABLE_DIR}")


# ==============================================================================
# 10. HARMONIZATION
# ==============================================================================

def prepare_records_for_harmonization(long_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "raw_determinant",
        "standardized_candidate_label",
        "determinant_family",
        "determinant_role",
        "tax_instrument_type",
        "direction",
        "include_in_main_table",
    ]

    available_cols = [c for c in cols if c in long_df.columns]

    records = (
        long_df[available_cols]
        .drop_duplicates()
        .fillna("")
        .to_dict(orient="records")
    )

    return records


def harmonize_determinants_with_llm(long_df: pd.DataFrame) -> Dict[str, Any]:
    harmonization_path = JSON_DIR / "04_harmonization_dictionary.json"

    if harmonization_path.exists() and not OVERWRITE_EXISTING and not OVERWRITE_HARMONIZATION:
        data = load_json(harmonization_path)
        if data is not None:
            return data

    records = prepare_records_for_harmonization(long_df)
    prompt = build_harmonization_prompt(records)

    raw = call_openai_text_only(
        prompt=prompt,
        model=MODEL_HARMONIZE,
        json_schema=HARMONIZATION_SCHEMA,
        reasoning_effort=REASONING_EFFORT_HARMONIZE,
        max_output_tokens=MAX_OUTPUT_TOKENS_HARMONIZE,
        stage_label="harmonization_text_only",
    )

    model = validate_or_save_error(
        raw,
        HarmonizationJSON,
        LOG_DIR / "harmonization_validation_error.json",
    )

    data = model.model_dump()
    save_json(data, harmonization_path)
    return data


def build_harmonization_dataframe(harmonization_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    for item in harmonization_json.get("harmonized_determinants", []):
        for raw_label in item.get("raw_labels_grouped", []):
            rows.append({
                "raw_determinant": raw_label,
                "canonical_determinant": item.get("canonical_label"),
                "canonical_family": item.get("determinant_family"),
                "canonical_definition": item.get("definition"),
                "recommended_main_table_label": item.get("recommended_main_table_label"),
                "distinction_notes": item.get("distinction_notes"),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "raw_determinant",
                "canonical_determinant",
                "canonical_family",
                "canonical_definition",
                "recommended_main_table_label",
                "distinction_notes",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates()


def apply_harmonization(
    long_df: pd.DataFrame,
    harmonization_df: pd.DataFrame,
) -> pd.DataFrame:
    df = long_df.copy()

    df["raw_determinant_clean"] = (
        df["raw_determinant"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if harmonization_df.empty:
        df["canonical_determinant"] = df["standardized_candidate_label"]
        df["canonical_family"] = df["determinant_family"]
        df["canonical_definition"] = df["definition"]
        df["recommended_main_table_label"] = df["standardized_candidate_label"]
        df["distinction_notes"] = ""
        return df.drop(columns=["raw_determinant_clean"])

    harm = harmonization_df.copy()
    harm["raw_determinant_clean"] = (
        harm["raw_determinant"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df.merge(
        harm.drop(columns=["raw_determinant"]),
        on="raw_determinant_clean",
        how="left",
    )

    df["canonical_determinant"] = df["canonical_determinant"].fillna(
        df["standardized_candidate_label"]
    )
    df["canonical_family"] = df["canonical_family"].fillna(
        df["determinant_family"]
    )
    df["canonical_definition"] = df["canonical_definition"].fillna(
        df["definition"]
    )
    df["recommended_main_table_label"] = df["recommended_main_table_label"].fillna(
        df["standardized_candidate_label"]
    )

    return df.drop(columns=["raw_determinant_clean"])


def build_determinant_level_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    main_df = df[
        df["include_in_main_table"].isin(["yes_main_result", "yes_secondary_result"])
    ].copy()

    if main_df.empty:
        return pd.DataFrame()

    grouped = (
        main_df.groupby("canonical_determinant", dropna=False)
        .agg(
            recommended_main_table_label=("recommended_main_table_label", join_unique),
            determinant_family=("canonical_family", join_unique),
            canonical_definition=("canonical_definition", join_unique),
            determinant_roles=("determinant_role", join_unique),
            n_papers=("title", lambda x: len(set([v for v in x if pd.notna(v)]))),
            papers=("title", join_unique),
            dois=("doi", join_unique),
            raw_terms=("raw_determinant", join_unique),
            tax_instruments=("tax_instrument_type", join_unique),
            directions_observed=("direction", join_unique),
            evidence_types=("evidence_type", join_unique),
            outcome_variables=("outcome_variable", join_unique),
        )
        .reset_index()
        .sort_values(["n_papers", "canonical_determinant"], ascending=[False, True])
    )

    return grouped


def save_harmonized_outputs(
    harmonization_df: pd.DataFrame,
    harmonized_long_df: pd.DataFrame,
    determinant_table: pd.DataFrame,
) -> None:
    save_dataframe_csv_and_xlsx(
        harmonization_df,
        TABLE_DIR / "harmonization_dictionary_tax_only.csv",
        TABLE_DIR / "harmonization_dictionary_tax_only.xlsx",
    )

    save_dataframe_csv_and_xlsx(
        harmonized_long_df,
        TABLE_DIR / "determinants_long_format_tax_only_harmonized.csv",
        TABLE_DIR / "determinants_long_format_tax_only_harmonized.xlsx",
    )

    save_dataframe_csv_and_xlsx(
        determinant_table,
        TABLE_DIR / "table_2_determinant_level_tax_only.csv",
        TABLE_DIR / "table_2_determinant_level_tax_only.xlsx",
    )

    log(f"Saved harmonized outputs in: {TABLE_DIR}")


# ==============================================================================
# 11. MAIN PIPELINE
# ==============================================================================

def run_one_pdf_safe(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    Wrapper used by the parallel executor.
    It catches fatal errors and saves a failed status.
    """
    paper_id = slugify(pdf_path.stem)

    try:
        return process_one_pdf(pdf_path)

    except Exception as e:
        error_path = LOG_DIR / f"{paper_id}_fatal_error.json"

        save_json(
            {
                "pdf_path": str(pdf_path),
                "paper_id": paper_id,
                "error": repr(e),
                "updated_at_unix": time.time(),
            },
            error_path,
        )

        previous_status = load_status(paper_id)
        previous_stage = previous_status.get("stage") if previous_status else "unknown"

        save_status(
            paper_id=paper_id,
            pdf_path=pdf_path,
            status="failed",
            stage=previous_stage or "unknown",
            reason=repr(e),
        )

        log(f"Fatal error for {pdf_path.name}: {e}")
        return None


def process_pdfs_parallel(pdf_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Process papers in parallel.

    Each paper remains sequential internally:
    screening -> extraction -> verification.

    KeyboardInterrupt is handled so pending futures are cancelled and
    currently running papers are marked as interrupted.
    """
    final_jsons: List[Dict[str, Any]] = []

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    future_to_pdf = {}

    try:
        future_to_pdf = {
            executor.submit(run_one_pdf_safe, pdf_path): pdf_path
            for pdf_path in pdf_files
        }

        for future in tqdm(
            as_completed(future_to_pdf),
            total=len(future_to_pdf),
            desc=f"Processing PDFs in parallel ({MAX_WORKERS} workers)",
        ):
            pdf_path = future_to_pdf[future]

            try:
                result = future.result()

                if result is not None:
                    final_jsons.append(result)

            except Exception as e:
                paper_id = slugify(pdf_path.stem)

                save_status(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    status="failed",
                    stage="parallel_future",
                    reason=repr(e),
                )

                log(f"Unexpected future error for {pdf_path.name}: {e}")

    except KeyboardInterrupt:
        log("\nKeyboardInterrupt received. Cancelling pending futures...")

        for future, pdf_path in future_to_pdf.items():
            paper_id = slugify(pdf_path.stem)

            if not future.done():
                future.cancel()

                previous_status = load_status(paper_id)
                previous_stage = previous_status.get("stage") if previous_status else "parallel_processing"

                save_status(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    status="interrupted",
                    stage=previous_stage or "parallel_processing",
                    reason="Interrupted by user with KeyboardInterrupt.",
                )

        executor.shutdown(wait=False, cancel_futures=True)

        log("Pending futures cancelled. You can relaunch the pipeline; interrupted papers will be retried.")
        return final_jsons

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return final_jsons


def process_pdfs_sequential(pdf_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Sequential fallback.
    """
    final_jsons: List[Dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs sequentially"):
        result = run_one_pdf_safe(pdf_path)

        if result is not None:
            final_jsons.append(result)

    return final_jsons


def run_pipeline() -> None:
    print("=" * 80)
    print("STRICT AI-assisted carbon/fuel tax acceptability review pipeline")
    print("=" * 80)

    if PARALLEL_PROCESSING:
        print(f"Parallel processing enabled: MAX_WORKERS={MAX_WORKERS}")
        print(f"Max concurrent OpenAI calls: MAX_CONCURRENT_API_CALLS={MAX_CONCURRENT_API_CALLS}")
    else:
        print("Parallel processing disabled.")

    pdf_files = list_pdf_files(LITERATURE_DIR)

    if TEST_MODE:
        pdf_files = pdf_files[:N_TEST_FILES]

    print(f"Number of PDFs found: {len(pdf_files)}")
    for p in pdf_files:
        print(f" - {p.name}")

    if len(pdf_files) == 0:
        print("No PDF files found. Stopping.")
        return

    if PARALLEL_PROCESSING:
        _ = process_pdfs_parallel(pdf_files)
    else:
        _ = process_pdfs_sequential(pdf_files)

    # Always reload everything from disk after processing.
    # This is safer in parallel mode and preserves previously completed papers.
    screening_jsons = load_all_screening_jsons()
    final_jsons = load_all_final_jsons()
    status_jsons = load_all_status_jsons()

    screening_df = build_screening_table(screening_jsons)
    audit_df = build_verification_audit_table()
    status_df = build_status_table(status_jsons)

    if len(final_jsons) == 0:
        print("No final included JSON files found.")
        print("Saving screening, audit, and status tables only.")

        empty_paper_df = pd.DataFrame()
        empty_long_df = pd.DataFrame()

        save_core_tables(
            screening_df=screening_df,
            paper_df=empty_paper_df,
            long_df=empty_long_df,
            audit_df=audit_df,
            status_df=status_df,
        )
        return

    paper_df = build_paper_level_table(final_jsons)
    long_df = build_long_determinant_table(final_jsons)

    save_core_tables(
        screening_df=screening_df,
        paper_df=paper_df,
        long_df=long_df,
        audit_df=audit_df,
        status_df=status_df,
    )

    if long_df.empty:
        print("Long determinant table is empty. Stopping before harmonization.")
        return

    print("Starting harmonization step.")
    print("This step is run once after all parallel paper-level jobs are completed.")

    harmonization_json = harmonize_determinants_with_llm(long_df)
    harmonization_df = build_harmonization_dataframe(harmonization_json)

    harmonized_long_df = apply_harmonization(long_df, harmonization_df)
    determinant_table = build_determinant_level_table(harmonized_long_df)

    save_harmonized_outputs(
        harmonization_df=harmonization_df,
        harmonized_long_df=harmonized_long_df,
        determinant_table=determinant_table,
    )

    print("=" * 80)
    print("Pipeline completed.")
    status_counts = status_df["status"].value_counts(dropna=False).to_dict() if not status_df.empty else {}
    print(f"Status summary: {status_counts}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()