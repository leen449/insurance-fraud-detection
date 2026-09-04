"""
app.py — Gradio interface for the Insurance Claim Fraud Detection & Action
Recommendation System, ready to deploy as a Hugging Face Space.

Visual design is ported 1:1 from the project's original Flask/HTML/CSS
frontend (`style.css`'s "deep-audit navy palette" design system: dark navy
background, teal/amber/red/green signal colors, Fraunces serif headers,
Inter sans body, IBM Plex Mono for numbers, a circular probability gauge,
pill-shaped chips, and diverging SHAP bar rows) — same look, pure Python.

To run locally:
    pip install -r requirements.txt
    python app.py

To deploy to Hugging Face Spaces:
  1. Create a new Space at huggingface.co/new-space, SDK = Gradio.
  2. Push app.py, model_pipeline.py, requirements.txt, README.md to the
     Space's repo root.
"""

import base64
import math
from pathlib import Path

import gradio as gr
import pandas as pd

import model_pipeline as mp

_LOGO_PATH = Path(__file__).parent / "assets" / "innovexa_header_mark.png"
_LOGO_B64 = (
    base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    if _LOGO_PATH.exists() else ""
)
_LOGO_SRC = f"data:image/png;base64,{_LOGO_B64}" if _LOGO_B64 else ""

print("=" * 70)
print("Loading finalized, rubric-passed fraud-detection results")
print("(XGBoost + isotonic calibration; SHAP drivers; Qwen2.5 briefs).")
print("Precomputed — no live training.")
print("=" * 70)
mp.PIPELINE.fit()

CLAIM_IDS = mp.PIPELINE.list_claim_ids()

# ---------------------------------------------------------------------------
# Design tokens — copied verbatim from the original frontend's :root vars
# in style.css, so this is the same palette, not a re-interpretation.
# ---------------------------------------------------------------------------

BG = "#0A0E17"
PANEL = "#10161F"
PANEL_RAISED = "#141B27"
BORDER = "#232C3B"
BORDER_SOFT = "#1A2130"
TEXT = "#E8ECF3"
TEXT_MUTED = "#8996AC"
TEXT_FAINT = "#5C6880"

TEAL = "#2FD3C4"
TEAL_DIM = "rgba(47, 211, 196, 0.14)"
AMBER = "#F2B03D"
AMBER_DIM = "rgba(242, 176, 61, 0.14)"
RED = "#F0645A"
RED_DIM = "rgba(240, 100, 90, 0.14)"
GREEN = "#4ADE9A"
GREEN_DIM = "rgba(74, 222, 154, 0.14)"

TIER_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED}
TIER_DIM = {"Low": GREEN_DIM, "Medium": AMBER_DIM, "High": RED_DIM}

FONT_IMPORT = (
    '@import url(\'https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;'
    '9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:'
    'wght@400;500;600&display=swap\');'
)

CUSTOM_CSS = f"""
{FONT_IMPORT}

.gradio-container {{
  max-width: 1360px !important; margin: auto !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background:
    radial-gradient(circle at 12% 0%, rgba(47, 211, 196, 0.06), transparent 40%),
    radial-gradient(circle at 90% 8%, rgba(242, 176, 61, 0.04), transparent 35%),
    {BG} !important;
}}
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container label {{
  color: {TEXT} !important;
}}
h1, h2, h3, .prose h1, .prose h2, .prose h3 {{
  font-family: 'Fraunces', Georgia, serif !important; font-weight: 500 !important;
  color: {TEXT} !important;
}}
code, .mono {{
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  color: inherit !important; padding: 0 !important;
}}

/* Panels / blocks */
.gr-block, .block {{ background: transparent; }}
.panel-block {{
  background: {PANEL} !important; border: 1px solid {BORDER} !important;
  border-radius: 10px !important;
}}

/* Accordions -> field groups */
.field-group-accordion {{
  background: {PANEL_RAISED} !important; border: 1px solid {BORDER_SOFT} !important;
  border-radius: 6px !important; margin-bottom: 10px !important;
}}
.field-group-accordion .label-wrap span {{ font-weight: 600 !important; font-size: 13.5px !important; }}

/* Inputs */
input, select, textarea, .gr-box {{
  background: {BG} !important; border: 1px solid {BORDER} !important;
  color: {TEXT} !important; border-radius: 6px !important;
}}
input:focus, select:focus {{ border-color: {TEAL} !important; }}
::placeholder {{ color: {TEXT_FAINT} !important; }}

/* Buttons */
#analyze-btn button, .btn-primary-custom button {{
  background: {TEAL} !important; color: #06231F !important; font-weight: 700 !important;
  border: none !important;
}}
#sample-btn button, .btn-secondary-custom button {{
  background: transparent !important; color: {TEXT_MUTED} !important;
  border: 1px solid {BORDER} !important; font-weight: 600 !important;
}}

/* Human-review actions: retain clear colour-coded affordances without the
   default Gradio white button rectangles. */
#approve-btn button, #review-btn button, #investigate-btn button {{
  background: transparent !important;
  border: 1px solid {BORDER} !important;
  box-shadow: none !important;
  font-weight: 650 !important;
}}
#approve-btn button {{ color: {GREEN} !important; border-color: rgba(74, 222, 154, .45) !important; }}
#review-btn button {{ color: {AMBER} !important; border-color: rgba(242, 176, 61, .45) !important; }}
#investigate-btn button {{ color: {RED} !important; border-color: rgba(240, 100, 90, .45) !important; }}
#approve-btn button:hover {{ background: rgba(74, 222, 154, .12) !important; }}
#review-btn button:hover {{ background: rgba(242, 176, 61, .12) !important; }}
#investigate-btn button:hover {{ background: rgba(240, 100, 90, .12) !important; }}

/* Tabs */
.tab-nav button {{ color: {TEXT_MUTED} !important; font-weight: 600 !important; }}
.tab-nav button.selected {{ color: {TEAL} !important; border-color: {TEAL} !important; }}

/* Dataframe */
table, .table-wrap {{ background: {PANEL} !important; color: {TEXT} !important; }}
thead th {{ background: {PANEL_RAISED} !important; color: {TEXT_MUTED} !important; }}

footer {{ display: none !important; }}

/* Rubric breakdown disclosure - native <details>, no JS, styled to match the panel
   system. Closed by default so the compact quality pill stays the primary view;
   the per-criterion grid reflows on its own (auto-fit) rather than needing a
   media query, so it stays usable at any column width. */
.rubric-details {{ margin-top: 10px; }}
.rubric-details summary {{
  cursor: pointer; list-style: none; font-size: 12px; font-weight: 600;
  color: {TEXT_MUTED}; padding: 6px 2px; display: flex; align-items: center;
  gap: 6px; user-select: none; border-radius: 4px;
}}
.rubric-details summary::-webkit-details-marker {{ display: none; }}
.rubric-details summary::before {{
  content: "▸"; display: inline-block; color: {TEXT_FAINT}; font-size: 10px;
  transition: transform 0.15s ease;
}}
.rubric-details[open] summary::before {{ transform: rotate(90deg); }}
.rubric-details summary:hover {{ color: {TEAL}; }}
.rubric-details summary:focus-visible {{ outline: 1px solid {TEAL}; outline-offset: 2px; }}
"""


# ---------------------------------------------------------------------------
# Result panel — ported 1:1 from the original .prob-gauge / .chip /
# .shap-bar-row / .brief-text markup and CSS, rendered as inline-styled HTML
# (Gradio's theme system can't reach into Python-generated HTML, so this
# result block carries its own styles matching the tokens above exactly).
# ---------------------------------------------------------------------------

def _empty_state_html() -> str:
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; text-align:center;
                color:{TEXT_MUTED}; padding:60px 18px;">
      <svg viewBox="0 0 64 64" width="46" height="46" style="color:{TEXT_FAINT}; margin-bottom:16px;">
        <circle cx="32" cy="32" r="29" fill="none" stroke="currentColor" stroke-width="2"/>
        <path d="M22 33 L29 40 L43 24" fill="none" stroke="currentColor" stroke-width="2.5"
              stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <h3 style="color:{TEXT}; font-size:17px; margin-bottom:8px; font-family:'Fraunces',serif;">
        No claim selected yet
      </h3>
      <p style="font-size:13.5px; max-width:34ch; margin:0;">
        Pick a claim from the dropdown to view its finalized, rubric-passed result from the
        calibrated XGBoost model, TreeSHAP explainer and action-recommendation engine.
      </p>
    </div>
    """


RUBRIC_CRITERIA_LABELS = {
    "factual_consistency": "Factual consistency",
    "faithfulness_to_shap": "Faithfulness to SHAP",
    "actionability": "Actionability",
    "no_fabrication": "No fabrication",
}


def _rubric_score_color(score) -> str:
    if score == 2:
        return GREEN
    if score == 1:
        return AMBER
    return RED  # 0, or a missing/non-numeric score - treated as worst-case, never hidden


def _rubric_detail_html(rubric: dict) -> str:
    """Per-criterion breakdown behind the compact quality pill - closed by default
    (progressive disclosure), same real data the pill's total is computed from, not
    a separate claim. Answers 'which criterion, and why' for a claim that failed,
    instead of leaving a bare FAIL with no reasoning attached."""
    cells = []
    for key, label in RUBRIC_CRITERIA_LABELS.items():
        score = rubric.get(key)
        color = _rubric_score_color(score)
        cells.append(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;
                    padding:8px 10px; background:{PANEL}; border:1px solid {BORDER_SOFT}; border-radius:6px;">
          <span style="font-size:11.5px; color:{TEXT_MUTED};">{label}</span>
          <span style="display:inline-flex; align-items:center; justify-content:center; min-width:22px;
                      height:22px; padding:0 6px; border-radius:999px; font-family:'IBM Plex Mono',monospace;
                      font-size:11px; font-weight:700; background:{color}22; color:{color}; border:1px solid {color}55;">
            {score if score is not None else '?'}
          </span>
        </div>
        """)
    criteria_grid = (
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); '
        'gap:8px; margin-top:8px;">' + "".join(cells) + "</div>"
    )

    note = rubric.get("reviewer_notes", "")
    note_html = (
        f'<p style="margin:10px 0 0; padding:10px 12px; background:{PANEL}; border:1px solid {BORDER_SOFT}; '
        f'border-radius:6px; font-size:12px; line-height:1.6; color:{TEXT_FAINT}; font-style:italic;">'
        f'"{note}"</p>'
    ) if note else ""

    return f"""
    <details class="rubric-details">
      <summary>View rubric breakdown ({rubric['total']}/{rubric['max_total']}, 0-2 per criterion)</summary>
      {criteria_grid}
      {note_html}
    </details>
    """


def _result_html(result: dict) -> str:
    tier = result["risk_tier"]
    tier_color = TIER_COLOR[tier]
    tier_dim = TIER_DIM[tier]
    prob_pct = result["fraud_probability"] * 100
    flagged = result["prediction"] == 1
    pred_color = RED if flagged else GREEN
    pred_dim = RED_DIM if flagged else GREEN_DIM

    # Circular gauge: circumference = 2*pi*60 (matches the original r=60 gauge)
    circumference = 2 * math.pi * 60
    offset = circumference * (1 - min(prob_pct, 100) / 100)
    gauge_color = tier_color

    # --- SHAP bar rows: diverging bars centered at 0, red=up teal=down,
    #     ported from .shap-bar-row / .shap-bar-track / .shap-bar-fill ---
    drivers = result["top_shap_drivers"]
    max_abs = max((abs(d["shap_contribution"]) for d in drivers), default=1) or 1
    bar_rows = []
    for d in drivers:
        pct = min(abs(d["shap_contribution"]) / max_abs * 50, 50)
        is_up = d["direction"] == "up"
        fill_color = RED if is_up else TEAL
        fill_style = f"left:50%; width:{pct:.1f}%;" if is_up else f"right:50%; width:{pct:.1f}%;"
        label = f"{d['feature']}"
        val = f"{d['value']}"
        bar_rows.append(f"""
        <div style="display:grid; grid-template-columns:148px 1fr 60px; align-items:center; gap:10px; font-size:12px;">
          <div style="color:{TEXT_MUTED}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            {label} <span style="color:{TEXT_FAINT};">({val})</span>
          </div>
          <div style="position:relative; height:8px; background:{BORDER_SOFT}; border-radius:4px; overflow:hidden;">
            <div style="position:absolute; top:0; bottom:0; {fill_style} background:{fill_color}; border-radius:4px;"></div>
            <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:{TEXT_FAINT}; opacity:0.4;"></div>
          </div>
          <div style="font-family:'IBM Plex Mono',monospace; text-align:right; color:{TEXT_FAINT}; font-size:11.5px;">
            {d['shap_contribution']:+.3f}
          </div>
        </div>
        """)
    shap_bars_html = '<div style="display:flex; flex-direction:column; gap:9px;">' + "".join(bar_rows) + "</div>"

    brief_mode_label = ("live Qwen2.5" if result.get("brief_source") == "live"
                         else "fallback demo mode — no live Qwen2.5 endpoint configured")
    rubric = result.get("brief_rubric_score", {"total": 0, "max_total": 8, "verdict": "N/A"})
    rubric_pass = rubric.get("verdict") == "PASS"
    rubric_color = GREEN if rubric_pass else AMBER
    rubric_dim = GREEN_DIM if rubric_pass else AMBER_DIM
    rubric_pill_style = (f"display:inline-flex; align-items:center; padding:5px 11px; border-radius:999px; "
                          f"font-size:11.5px; font-weight:700; border:1px solid {rubric_color}55; "
                          f"background:{rubric_dim}; color:{rubric_color};")
    n_fields_sent = len(result.get("privacy", {}).get("fields_sent_to_qwen", []))

    return f"""
    <div style="display:flex; flex-direction:column; gap:26px; font-family:'Inter',sans-serif;">

      <div style="display:flex; align-items:center; gap:22px; flex-wrap:wrap;">
        <div style="position:relative; width:140px; height:140px; flex-shrink:0;">
          <svg viewBox="0 0 140 140" width="140" height="140" style="transform:rotate(-90deg);">
            <circle cx="70" cy="70" r="60" fill="none" stroke="{BORDER_SOFT}" stroke-width="10"/>
            <circle cx="70" cy="70" r="60" fill="none" stroke="{gauge_color}" stroke-width="10"
                    stroke-linecap="round" stroke-dasharray="{circumference:.1f}"
                    stroke-dashoffset="{offset:.1f}"/>
          </svg>
          <div style="position:absolute; inset:0; display:flex; flex-direction:column;
                      align-items:center; justify-content:center;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:{TEXT};">
              {prob_pct:.0f}%
            </span>
            <span style="font-size:10.5px; color:{TEXT_FAINT}; text-transform:uppercase; letter-spacing:.05em; margin-top:2px;">
              fraud probability
            </span>
          </div>
        </div>

        <div style="flex:1; min-width:200px; display:flex; flex-direction:column; gap:12px;">
          <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <span style="display:inline-flex; align-items:center; padding:6px 13px; border-radius:999px;
                        font-size:12.5px; font-weight:600; border:1px solid {pred_color}55;
                        background:{pred_dim}; color:{pred_color};">
              {result['prediction_label']}
            </span>
            <span style="display:inline-flex; align-items:center; padding:6px 13px; border-radius:999px;
                        font-size:12.5px; font-weight:600; border:1px solid {tier_color}55;
                        background:{tier_dim}; color:{tier_color};">
              {tier} risk
            </span>
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;
                      background:{PANEL_RAISED}; border:1px solid {BORDER_SOFT}; border-radius:6px; padding:12px 14px;">
            <span style="font-size:12px; color:{TEXT_MUTED};">Recommended action</span>
            <span style="font-weight:700; font-size:14px; color:{TEXT};">{result['recommended_action']}</span>
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; gap:18px;
                      font-size:12px; color:{TEXT_FAINT};">
            <span>Operating threshold: <strong style="color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace;
                  font-weight:600;">{result['operating_threshold']:.0%}</strong></span>
            <span>Raw model score: <strong style="color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace;
                  font-weight:600;">{result['raw_model_score']:.1%}</strong></span>
          </div>
        </div>
      </div>

      <div>
        <h3 style="font-size:14.5px; font-family:'Inter',sans-serif; font-weight:700; margin-bottom:12px; color:{TEXT};">
          Key risk factors
          <span style="font-weight:400; color:{TEXT_FAINT}; font-size:12px;"> (TreeSHAP, this claim only)</span>
        </h3>
        {shap_bars_html}
      </div>

      <div>
        <h3 style="font-size:14.5px; font-family:'Inter',sans-serif; font-weight:700; margin-bottom:12px; color:{TEXT};">
          Investigation brief
          <span style="font-weight:400; color:{TEXT_FAINT}; font-size:12px;">
            ({brief_mode_label})
          </span>
        </h3>
        <p style="background:{PANEL_RAISED}; border:1px solid {BORDER_SOFT}; border-left:3px solid {TEAL};
                  border-radius:6px; padding:16px 18px; font-size:13.5px; line-height:1.65; color:{TEXT}; margin:0;">
          {result['investigation_brief']}
        </p>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
          <span style="{rubric_pill_style}">
            Brief quality: {rubric['total']}/{rubric['max_total']} — {rubric['verdict']}
          </span>
          <span style="display:inline-flex; align-items:center; padding:5px 11px; border-radius:999px;
                      font-size:11.5px; font-weight:600; border:1px solid {BORDER}; color:{TEXT_MUTED};">
            Qwen data shared: {n_fields_sent} fields (Sex/Age withheld unless a top risk driver)
          </span>
        </div>
        {_rubric_detail_html(rubric)}
      </div>

      <div>
        <h3 style="font-size:14.5px; font-family:'Inter',sans-serif; font-weight:700; margin-bottom:12px; color:{TEXT};">
          Human review safeguard
        </h3>
        <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;
                    background:{PANEL_RAISED}; border:1px solid {BORDER_SOFT}; border-radius:6px; padding:12px 14px;">
          <div style="font-size:12.5px; color:{TEXT_MUTED};">
            ML sets the <strong style="color:{TEXT};">probability &amp; risk tier</strong>. Qwen only drafts the
            <strong style="color:{TEXT};">explanation &amp; suggested action</strong>. A human adjuster makes the
            <strong style="color:{TEXT};">final operational decision</strong> below.
          </div>
          <span style="display:inline-flex; align-items:center; padding:6px 13px; border-radius:999px;
                      font-size:12.5px; font-weight:700; border:1px solid {AMBER}55; background:{AMBER_DIM}; color:{AMBER};">
            {result['human_review_status']}
          </span>
        </div>
      </div>
    </div>
    """


def _claim_details_df(claim_details: dict) -> pd.DataFrame:
    rows = [{"Field": k, "Value": v} for k, v in claim_details.items()]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def analyze_claim(claim_id: str):
    if not claim_id:
        return (_empty_state_html(), None, _adjuster_status_html(None), _claim_details_df({}))
    try:
        result = mp.PIPELINE.predict({"claim_id": claim_id})
    except Exception as exc:  # noqa: BLE001
        error_html = (f"<div style='border:1px solid {RED}55; background:{RED_DIM}; border-radius:6px; "
                       f"padding:14px 18px; color:{RED}; font-weight:600;'>Error: {exc}</div>")
        return error_html, None, _adjuster_status_html(None), _claim_details_df({})

    tier = result["risk_tier"]
    result_html = _result_html(result)
    claim_details = result.get("claim_details", {})

    # Context handed to the adjuster-decision buttons below — every new
    # analysis resets the decision to unresolved, since a fresh ML/Qwen
    # output requires a fresh human sign-off (a stale approval from a
    # previous claim must never carry over).
    decision_context = {
        "risk_tier": tier, "recommended_action": result["recommended_action"],
        "fraud_probability": result["fraud_probability"],
    }
    return (result_html, decision_context, _adjuster_status_html(None), _claim_details_df(claim_details))


def _adjuster_status_html(decision: str | None, context: dict | None = None) -> str:
    """Renders the human-adjuster sign-off status. `decision` is None until
    a reviewer actually clicks one of the three decision buttons below —
    the system-suggested action is never auto-promoted to a final decision."""
    if decision is None:
        return f"""
        <div style="display:flex; align-items:center; gap:8px; padding:9px 14px; border-radius:6px;
                    border:1px solid {AMBER}55; background:{AMBER_DIM}; color:{AMBER}; font-weight:700;
                    font-size:12.5px; margin-top:10px;">
          Pending Adjuster Review — analyze a claim, then a human must record a decision below before any
          operational action is considered final.
        </div>
        """
    color = {"Approved": GREEN, "Manual Review": AMBER, "Fraud Investigation": RED}.get(decision, TEAL)
    dim = {"Approved": GREEN_DIM, "Manual Review": AMBER_DIM, "Fraud Investigation": RED_DIM}.get(decision, TEAL_DIM)
    suggested = f" (system suggested: {context['recommended_action']})" if context else ""
    return f"""
    <div style="display:flex; align-items:center; gap:8px; padding:9px 14px; border-radius:6px;
                border:1px solid {color}55; background:{dim}; color:{color}; font-weight:700;
                font-size:12.5px; margin-top:10px;">
      Final decision: {decision} — recorded by human adjuster{suggested}. ML and Qwen provided input only;
      this decision is the human's, not the system's.
    </div>
    """


def record_adjuster_decision(decision: str, context: dict | None):
    if context is None:
        return _adjuster_status_html(None)
    return _adjuster_status_html(decision, context)


# ---------------------------------------------------------------------------
# Model strip (header stat cards) + Model Performance tab
# ---------------------------------------------------------------------------

def _stat_card(label: str, value) -> str:
    return f"""
    <div style="min-width:118px; padding:8px 14px; border:1px solid {BORDER_SOFT}; border-radius:6px; background:{PANEL};">
      <span style="display:block; font-size:10.5px; text-transform:uppercase; letter-spacing:.06em;
                   color:{TEXT_FAINT}; margin-bottom:4px;">{label}</span>
      <span style="font-family:'IBM Plex Mono',monospace; font-size:17px; font-weight:600; color:{TEAL};">{value}</span>
    </div>
    """


def _header_html() -> str:
    m = mp.PIPELINE.metrics
    stats = "".join([
        _stat_card("PR-AUC", m["pr_auc_test"]),
        _stat_card("Beats RF baseline", m["pr_auc_status"]),
        _stat_card("Recall", m["recall_test"]),
        _stat_card("Precision", m["precision_test"]),
        _stat_card("Brier", m["brier_score_test"]),
        _stat_card("Qwen2.5", m["qwen_status"]["mode"]),
    ])
    return f"""
    <div style="display:flex; align-items:center; justify-content:space-between; gap:24px;
                padding:28px 0 22px; border-bottom:1px solid {BORDER_SOFT}; flex-wrap:wrap;">
      <div style="display:flex; align-items:center; gap:18px;">
        <img src="{_LOGO_SRC}" alt="Innovexa"
             style="height:100px; width:144px; object-fit:contain; flex-shrink:0; border-radius:12px;
                    image-rendering:auto; filter:drop-shadow(0 5px 14px rgba(47, 211, 196, .22));"/>
        <div>
          <h1 style="font-size:26px; letter-spacing:.2px; margin:0; font-family:'Fraunces',serif; color:{TEXT};">Innovexa</h1>
          <p style="margin:3px 0 0; color:{TEXT_MUTED}; font-size:13px;">
            Claim Fraud Intelligence Console — Team Innovexa, SIC AI Capstone (Gradio build)
          </p>
        </div>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">{stats}</div>
    </div>
    {_data_source_banner_html()}
    """


def _data_source_banner_html() -> str:
    m = mp.PIPELINE.metrics
    if m.get("is_real_data"):
        return ""
    return f"""
    <div style="margin-top:14px; padding:12px 16px; border:1px solid {AMBER}55; background:{AMBER_DIM};
                border-radius:8px; font-size:12.5px; color:{AMBER}; line-height:1.6;">
      <strong>⚠ SYNTHETIC FALLBACK DATA.</strong> The real Kaggle dataset was not found, so this app trained on a
      statistically-matched synthetic stand-in — metrics below are NOT the Capstone's reported results.
      {m.get('dataset_placement_help', '')}
    </div>
    """


def _footer_html() -> str:
    return f"""
    <div style="margin-top:30px; padding-top:18px; border-top:1px solid {BORDER_SOFT};">
      <p style="color:{TEXT_FAINT}; font-size:11.5px; max-width:90ch; margin:0;">
        Proof-of-concept trained on a historical, statistically-matched benchmark. Not a production
        deployment decision system — see project scope note. Model: XGBoost + isotonic calibration ·
        Explainability: TreeSHAP · Data: Vehicle Claim Fraud Detection (CC0).
      </p>
    </div>
    """


def _metrics_body_html() -> str:
    m = mp.PIPELINE.metrics
    cm = m["confusion_matrix_test"]
    imbalance = m["imbalance_comparison"]
    imbalance_line = (
        f"(class-weight PR-AUC={imbalance['class_weight']['mean_pr_auc']} vs. "
        f"SMOTENC PR-AUC={imbalance['smotenc']['mean_pr_auc']}, mean over CV folds)"
        if imbalance.get("available")
        else f"— {imbalance.get('note', 'comparison not available')}"
    )
    cards = "".join([
        _stat_card("PR-AUC (test)", m["pr_auc_test"]),
        _stat_card("ROC-AUC", m["roc_auc_test"]),
        _stat_card("Recall", m["recall_test"]),
        _stat_card("Precision", m["precision_test"]),
        _stat_card("Macro-F1", m["macro_f1_test"]),
        _stat_card("MCC", m["mcc_test"]),
        _stat_card("Brier score", m["brier_score_test"]),
    ])
    pass_color = GREEN if m["pr_auc_pass"] else RED
    pass_dim = GREEN_DIM if m["pr_auc_pass"] else RED_DIM
    return f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 20px;">{cards}</div>
    <div style="display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:8px;
                border:1px solid {pass_color}55; background:{pass_dim}; color:{pass_color}; font-weight:700;
                font-size:13.5px; margin-bottom:16px;">
      Beats Balanced RF baseline (validation, the real bar per Notebook 04): {m['pr_auc_status']} &nbsp;
      <span style="font-weight:400; color:{TEXT_MUTED}; font-size:12px;">
        (candidate val PR-AUC = {m['pr_auc_candidate_validation']} vs. RF baseline = {m['pr_auc_rf_baseline']};
        test PR-AUC = {m['pr_auc_test']}, un-manipulated; published reference ~{m['pr_auc_published_reference']}
        is context only, not the gate)
      </span>
    </div>
    <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:18px 20px;
                font-size:13.5px; color:{TEXT_MUTED}; line-height:1.8;">
      <b style="color:{TEXT};">Data source:</b> {m['data_source']} (Kaggle CC0 1.0 / public domain; label-quality
      checked — no duplicate rows, no missing fraud labels, target consistently 0/1)<br>
      <b style="color:{TEXT};">Total claims:</b> {m['n_total']:,} spanning 1994–1996 (fraud rate: ~{m['fraud_rate_pct']}%)
      &nbsp; <span style="color:{TEXT_FAINT};">— train / val / test split: {m['n_train']} / {m['n_val']} / {m['n_test']}</span><br>
      <b style="color:{TEXT};">Chosen imbalance strategy:</b>
      <code style="color:{TEAL};">{m['imbalance_strategy_chosen']}</code>
      {imbalance_line}<br>
      <b style="color:{TEXT};">Confusion matrix</b> @ threshold {m['operating_threshold']}:
      TP={cm['tp']} &nbsp; FP={cm['fp']} &nbsp; FN={cm['fn']} &nbsp; TN={cm['tn']}<br>
      <b style="color:{TEXT};">Risk tiers</b> (bounds chosen on the VALIDATION split only, never test):
      <span style="color:{GREEN}; font-weight:700;">Low</span> &lt; {m['risk_tiers']['low_max']:.1%}
      (<span style="color:{TEXT_FAINT};">{m['risk_tiers']['low_share_pct']}% of claims, {m['risk_tiers']['low_fraud_rate_pct']}% fraud rate</span>) ≤
      <span style="color:{AMBER}; font-weight:700;">Medium</span> &lt; {m['risk_tiers']['high_min']:.0%}
      (<span style="color:{TEXT_FAINT};">{m['risk_tiers']['medium_share_pct']}% of claims, {m['risk_tiers']['medium_fraud_rate_pct']}% fraud rate</span>) ≤
      <span style="color:{RED}; font-weight:700;">High</span>
      (<span style="color:{TEXT_FAINT};">{m['risk_tiers']['high_share_pct']}% of claims, {m['risk_tiers']['high_fraud_rate_pct']}% fraud rate</span>)<br>
    </div>
    """


def _temporal_body_html() -> str:
    t = mp.PIPELINE.metrics.get("temporal_validation", {})
    if not t.get("available"):
        return f"""
        <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:18px 20px;
                    color:{TEXT_MUTED}; font-size:13.5px;">
          Temporal validation unavailable: {t.get('reason', 'unknown reason')}
        </div>
        """
    # This experiment was deliberately never calibrated or thresholded (it's a
    # distribution-shift sanity check, not the production pipeline) - so there is no
    # real confusion matrix, ECE, or pass/fail badge to show for it. Showing only
    # what genuinely exists here, rather than fabricating those to match the main
    # test-set panel's layout.
    cards = "".join([
        _stat_card("PR-AUC", t["pr_auc"]),
        _stat_card("ROC-AUC", t["roc_auc"]),
        _stat_card("Recall", t["recall"]),
        _stat_card("Precision", t["precision"]),
        _stat_card("Macro-F1", t["macro_f1"]),
        _stat_card("MCC", t["mcc"]),
        _stat_card("Brier score", t["brier_score"]),
    ])
    return f"""
    <p style="color:{TEXT_MUTED}; font-size:13px; max-width:80ch;">
      Separate temporal-generalisation experiment — a fresh model (same recipe as the
      production model) trained only on policy years
      <strong style="color:{TEXT};">{t['train_period']}</strong> and evaluated once on unseen year
      <strong style="color:{TEXT};">{t['test_period']}</strong>. This model was
      <strong style="color:{TEXT};">not calibrated or thresholded</strong> — it's a distribution-shift sanity
      check, not a second primary evaluation, and its numbers aren't directly comparable to the main
      test-set metrics above.
    </p>
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 16px;">{cards}</div>
    <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:18px 20px;
                font-size:13.5px; color:{TEXT_MUTED}; line-height:1.8;">
      <b style="color:{TEXT};">Train fraud rate ({t['train_period']}):</b> {t['train_fraud_rate_pct']}%
      &nbsp; <b style="color:{TEXT};">Test fraud rate ({t['test_period']}):</b> {t['test_fraud_rate_pct']}%<br>
      <span style="color:{TEXT_FAINT};">{t['description']}</span>
    </div>
    """


def _fairness_df() -> pd.DataFrame:
    fairness = mp.PIPELINE.metrics.get("fairness", {})
    rows = []
    for attribute, groups in fairness.get("groups_by_attribute", {}).items():
        for g in groups:
            rows.append({
                "Attribute": attribute, "Group": g["group"], "N": g["n"],
                "% of test set": g["pct_of_test"], "Flag rate %": g["flag_rate_pct"],
                "False positive rate %": g["fpr_pct"], "False negative rate %": g["fnr_pct"],
                "Fraud base rate %": g["fraud_base_rate_pct"],
            })
    return pd.DataFrame(rows)


def _fairness_note_html() -> str:
    fairness = mp.PIPELINE.metrics.get("fairness", {})
    description = fairness.get("description", "")
    note = fairness.get("note", "")
    note_html = f'<br><br><b style="color:{TEXT};">Note:</b> {note}' if note else ""
    return f"""
    <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px;
                font-size:12.5px; color:{TEXT_MUTED}; line-height:1.7; margin-top:10px;">
      {description}{note_html}
    </div>
    """


def _global_shap_df() -> pd.DataFrame:
    rows = mp.PIPELINE.metrics.get("global_shap", [])
    return pd.DataFrame(rows).rename(columns={"feature": "Feature", "mean_abs_shap": "Mean |SHAP value|"})


def _qwen_status_html() -> str:
    status = mp.PIPELINE.metrics.get("qwen_status", {})
    is_live = status.get("mode") == "live"
    color = GREEN if is_live else AMBER
    dim = GREEN_DIM if is_live else AMBER_DIM
    extra = f" &nbsp; endpoint: <code style='color:{TEAL};'>{status.get('api_base')}</code>" if is_live else ""
    return f"""
    <div style="display:inline-flex; align-items:center; gap:8px; padding:10px 16px; border-radius:8px;
                border:1px solid {color}55; background:{dim}; color:{color}; font-weight:700; font-size:13.5px;
                margin-bottom:14px;">
      Qwen2.5 status: {status.get('mode', 'unknown').upper()}
      <span style="font-weight:400; color:{TEXT_MUTED}; font-size:12px;">
        &nbsp;— model: {status.get('model')} · {status.get('reason', '')}{extra}
      </span>
    </div>
    <p style="color:{TEXT_FAINT}; font-size:12px; max-width:80ch; margin:0 0 14px;">
      Configure via environment variables: <code>QWEN_ENABLED=true</code>, <code>QWEN_API_KEY=...</code>,
      <code>QWEN_MODEL=Qwen2.5-...</code>, <code>QWEN_API_BASE=https://your-endpoint/v1</code>. The API key is
      read server-side only and is never sent to the frontend.
    </p>
    """


def _baseline_comparison_df() -> pd.DataFrame:
    comp = mp.PIPELINE.metrics["baseline_comparison"]
    rows = []
    for model_name, vals in comp.items():
        if model_name == "production_model" or "note" in vals:
            continue
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "PR-AUC (val)": vals["pr_auc"],
            "Recall @ 0.5 (val)": vals["recall_at_050"],
            "Precision @ 0.5 (val)": vals["precision_at_050"],
        })
    return pd.DataFrame(rows).sort_values("PR-AUC (val)", ascending=False)


def _reliability_df() -> pd.DataFrame:
    curve = mp.PIPELINE.metrics.get("reliability_curve_test", [])
    return pd.DataFrame(curve).rename(columns={
        "mean_predicted": "Mean predicted probability",
        "observed_frequency": "Observed fraud frequency",
    })


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app
# ---------------------------------------------------------------------------

THEME = gr.themes.Base().set(
    body_background_fill=BG, body_background_fill_dark=BG,
    body_text_color=TEXT, body_text_color_dark=TEXT,
    background_fill_primary=PANEL, background_fill_primary_dark=PANEL,
    background_fill_secondary=PANEL_RAISED, background_fill_secondary_dark=PANEL_RAISED,
    border_color_primary=BORDER, border_color_primary_dark=BORDER,
    block_background_fill=PANEL, block_background_fill_dark=PANEL,
    block_border_color=BORDER, block_border_color_dark=BORDER,
    block_title_text_color=TEXT, block_title_text_color_dark=TEXT,
    block_label_text_color=TEXT_MUTED, block_label_text_color_dark=TEXT_MUTED,
    panel_background_fill=PANEL_RAISED, panel_background_fill_dark=PANEL_RAISED,
    panel_border_color=BORDER_SOFT, panel_border_color_dark=BORDER_SOFT,
    input_background_fill=BG, input_background_fill_dark=BG,
    input_border_color=BORDER, input_border_color_dark=BORDER,
    button_primary_background_fill=TEAL, button_primary_background_fill_dark=TEAL,
    button_primary_background_fill_hover=TEAL, button_primary_text_color="#06231F",
    button_primary_text_color_dark="#06231F",
    button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
    button_secondary_border_color=BORDER, button_secondary_border_color_dark=BORDER,
    button_secondary_text_color=TEXT_MUTED, button_secondary_text_color_dark=TEXT_MUTED,
)

with gr.Blocks(title="Innovexa — Claim Fraud Intelligence Console") as demo:
    gr.HTML(_header_html())

    with gr.Tab("Claim Analyzer"):
        with gr.Row():
            with gr.Column(scale=135, elem_classes=["panel-block"]):
                gr.Markdown(
                    f"### Claim Selection\n"
                    f"Choose one of the {len(CLAIM_IDS)} rubric-passed claims (spanning Low/Medium/High risk "
                    "tiers) to view its finalized, precomputed result — no live model runs here, these are "
                    "the project's real reported outputs."
                )
                claim_dropdown = gr.Dropdown(
                    choices=CLAIM_IDS, value=CLAIM_IDS[0] if CLAIM_IDS else None,
                    label="Claim ID", elem_id="claim-selector",
                )
                with gr.Accordion("Claim details", open=True, elem_classes=["field-group-accordion"]):
                    claim_details_table = gr.Dataframe(
                        value=_claim_details_df({}), headers=["Field", "Value"],
                        interactive=False, wrap=True,
                    )

            with gr.Column(scale=100, elem_classes=["panel-block"]):
                result_html = gr.HTML(_empty_state_html())
                decision_state = gr.State(value=None)
                adjuster_status_html = gr.HTML(_adjuster_status_html(None))
                with gr.Row():
                    approve_btn = gr.Button("✓ Approve", elem_id="approve-btn", size="sm")
                    review_btn = gr.Button("Send to Manual Review", elem_id="review-btn", size="sm")
                    investigate_btn = gr.Button("Escalate to Fraud Investigation", elem_id="investigate-btn", size="sm")

        analyze_outputs = [result_html, decision_state, adjuster_status_html, claim_details_table]
        claim_dropdown.change(fn=analyze_claim, inputs=claim_dropdown, outputs=analyze_outputs)
        demo.load(fn=analyze_claim, inputs=claim_dropdown, outputs=analyze_outputs)
        approve_btn.click(fn=lambda ctx: record_adjuster_decision("Approved", ctx),
                           inputs=decision_state, outputs=adjuster_status_html)
        review_btn.click(fn=lambda ctx: record_adjuster_decision("Manual Review", ctx),
                          inputs=decision_state, outputs=adjuster_status_html)
        investigate_btn.click(fn=lambda ctx: record_adjuster_decision("Fraud Investigation", ctx),
                               inputs=decision_state, outputs=adjuster_status_html)

    with gr.Tab("Model Performance"):
        gr.HTML(_metrics_body_html())
        gr.Markdown("#### Baseline comparison (validation set) — confirms the added model complexity is justified")
        gr.Dataframe(value=_baseline_comparison_df(), interactive=False)
        gr.Markdown("#### Precision / recall trade-off across candidate thresholds (validation set)")
        threshold_sweep = mp.PIPELINE.metrics.get("threshold_sweep", [])
        if threshold_sweep:
            gr.Dataframe(value=pd.DataFrame(threshold_sweep), interactive=False)
        else:
            gr.Markdown("n/a — the per-threshold sweep table isn't part of this precomputed release.")
       

    with gr.Tab("Temporal Validation"):
        gr.Markdown(
            "The full dataset spans **1994–1996**, so it may not reflect current fraud patterns — vehicle "
            "values, policy structures, and fraud tactics have all changed since then. This project validates "
            "the methodology on a historical benchmark rather than claiming present-day deployment readiness; "
            "real-world use would require retraining on recent claims data. The split below is exactly how "
            "that limitation is tested within the available data: everything the model has ever seen is "
            "capped at 1995, and it's scored purely on unseen 1996 claims."
        )
        gr.HTML(_temporal_body_html())

    with gr.Tab("Explainability & Fairness"):
        gr.Markdown("#### Global SHAP feature importance\nMean |SHAP value| across a sample of held-out test "
                     "claims, computed against the exact same calibrated model used for live predictions.")
        if mp.PIPELINE.metrics.get("global_shap"):
            gr.Dataframe(value=_global_shap_df(), interactive=False)
        else:
            gr.Markdown("n/a — global SHAP importances aren't part of this precomputed release. Per-claim "
                        "SHAP drivers are still available in the Claim Analyzer tab.")
        gr.Markdown("#### Fairness evaluation (held-out test set, at the production operating threshold)")
        fairness_df = _fairness_df()
        if not fairness_df.empty:
            gr.Dataframe(value=fairness_df, interactive=False)
        gr.HTML(_fairness_note_html())

    with gr.Tab("Qwen2.5 & Human Review"):
        gr.HTML(_qwen_status_html())
        gr.Markdown(
            "#### Investigation-brief quality rubric\n"
            "Every generated brief is automatically scored 0/1/2 on each "
            "criterion below; a brief is accepted only if it scores ≥6/8 overall AND scores 2 on both "
            "*factual consistency* and *no fabrication* — otherwise it's marked FAILED / NEEDS REVIEW.\n\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in mp.QWEN_BRIEF_RUBRIC.items()
                         if k not in ("scale", "acceptance_rule", "note_vs_action_plan"))
            + f"\n\n*Note on method:* {mp.QWEN_BRIEF_RUBRIC['note_vs_action_plan']}"
        )
        gr.Markdown(
            "#### Human-in-the-loop safeguard\n"
            "For every claim: the **ML classifier** is the sole source of fraud probability and risk tier; "
            "**Qwen2.5** may only draft an explanation and a suggested next action from those facts, and can "
            "never override them; a **human adjuster** must review the Claim Analyzer output before any "
            "operational action (approve / manual review / fraud investigation) is actually taken. Qwen never "
            "silently becomes the final decision-maker."
        )
        gr.Markdown(
            "#### Privacy / data-minimisation layer\n"
            "Before any claim data is sent to Qwen2.5, a privacy layer strips sensitive demographic fields "
            "(Sex, Age, Age bracket, Marital Status) unless that specific field is itself one of the claim's "
            "top SHAP risk drivers — in which case it's kept because the brief needs to reference the evidence "
            "it's explaining. `QWEN_ENABLED` must be explicitly set to `true` for any claim data to leave the "
            "process at all; the Claim Analyzer result panel always shows exactly how many fields were sent."
        )

    gr.HTML(_footer_html())

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
