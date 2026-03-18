import os
import joblib
import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Transfer Decision Engine", layout="wide")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# OPENAI
# =========================
OPENAI_ENABLED = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

client = None
if OPENAI_ENABLED:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None
        OPENAI_ENABLED = False

# =========================
# CLUB LOGOS
# =========================
CLUB_LOGOS = {
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Barcelona": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg",
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg",
    "Bayern Munich": "https://upload.wikimedia.org/wikipedia/en/1/1f/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg",
    "PSG": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg",
    "Paris Saint-Germain": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg",
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Manchester United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "Newcastle United": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Juventus": "https://upload.wikimedia.org/wikipedia/commons/1/15/Juventus_FC_2017_logo.svg",
    "Inter": "https://upload.wikimedia.org/wikipedia/commons/0/05/FC_Internazionale_Milano_2021.svg",
    "AC Milan": "https://upload.wikimedia.org/wikipedia/commons/d/d0/Logo_of_AC_Milan.svg",
    "Napoli": "https://upload.wikimedia.org/wikipedia/commons/2/2d/SSC_Neapel.svg",
    "Atletico Madrid": "https://upload.wikimedia.org/wikipedia/en/f/f4/Atletico_Madrid_2017_logo.svg",
    "Borussia Dortmund": "https://upload.wikimedia.org/wikipedia/commons/6/67/Borussia_Dortmund_logo.svg",
    "RB Leipzig": "https://upload.wikimedia.org/wikipedia/en/0/04/RB_Leipzig_2014_logo.svg",
}

PLAYER_PLACEHOLDER = "https://cdn-icons-png.flaticon.com/512/149/149071.png"

# =========================
# HEADER
# =========================
st.title("⚽ AI Transfer Decision Engine")
st.caption("Select a player + target club to predict success probability, injury risk, tactical fit, and ROI.")

# =========================
# GLOBAL STYLE
# =========================
st.markdown(
    """
    <style>
    .ai-mini-strip {
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.14);
    }
    .ai-mini-strip .label {
        font-size: 12px;
        opacity: 0.72;
        margin-bottom: 6px;
        letter-spacing: 0.2px;
        font-weight: 600;
    }
    .ai-mini-strip .verdict {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .ai-mini-strip .reason {
        font-size: 14px;
        opacity: 0.92;
        line-height: 1.55;
    }

    .ai-metric-chip {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
        border-radius: 16px;
        padding: 12px 14px;
        min-height: 78px;
    }
    .ai-metric-chip .k {
        font-size: 11px;
        opacity: 0.62;
        text-transform: uppercase;
        letter-spacing: 0.35px;
        margin-bottom: 5px;
        font-weight: 700;
    }
    .ai-metric-chip .v {
        font-size: 17px;
        font-weight: 700;
        line-height: 1.25;
    }

    .ai-report-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 210px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.14);
    }
    .ai-report-card-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.45px;
        opacity: 0.62;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .ai-report-card-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
        line-height: 1.35;
    }
    .ai-report-card-text {
        font-size: 15px;
        line-height: 1.8;
        opacity: 0.96;
    }

    .ai-footnote {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.07);
        font-size: 12px;
        opacity: 0.74;
        line-height: 1.6;
    }

    .sidebar-llm-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-radius: 16px;
        padding: 14px 16px;
        line-height: 1.7;
        font-size: 14px;
        margin-top: 10px;
    }

    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[role="radiogroup"] {
        display: flex !important;
        gap: 10px !important;
        flex-wrap: nowrap !important;
        margin-bottom: 12px !important;
    }
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 999px !important;
        padding: 10px 16px !important;
        min-height: auto !important;
    }
    div[role="radiogroup"] label p {
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 999px !important;
        padding: 0.55rem 1rem !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.02) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: rgba(255,255,255,0.22) !important;
        background: rgba(255,255,255,0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD DATA + MODEL
# =========================
players = pd.read_csv(os.path.join(BASE_DIR, "players_app.csv"))
teams = pd.read_csv(os.path.join(BASE_DIR, "teams.csv"))
leagues = pd.read_csv(os.path.join(BASE_DIR, "leagues.csv"))

def train_fallback_model():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "transfers_train.csv"))

    y = train_df["transfer_success"]

    drop_cols = ["transfer_success", "target_team", "target_league"]
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    model_local = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model_local.fit(X, y)
    joblib.dump(model_local, MODEL_PATH)
    return model_local

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = train_fallback_model()

league_diff = dict(zip(leagues["league"], leagues["difficulty"]))
league_intensity = dict(zip(leagues["league"], leagues["intensity"]))
league_mm = dict(zip(leagues["league"], leagues["market_multiplier"]))

SYSTEM_ROLE_NEEDS = {
    "4-3-3": {"wing_play": 0.8, "press_intensity": 0.7, "build_up": 0.7, "directness": 0.5, "duel_focus": 0.5},
    "4-2-3-1": {"wing_play": 0.7, "press_intensity": 0.6, "build_up": 0.75, "directness": 0.55, "duel_focus": 0.55},
    "3-5-2": {"wing_play": 0.55, "press_intensity": 0.65, "build_up": 0.6, "directness": 0.65, "duel_focus": 0.7},
}

# =========================
# HELPERS
# =========================
def euro(x: float) -> str:
    return f"€{x:,.0f}"

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def clamp01(x: float) -> float:
    return float(np.clip(x, 0, 1))

def clamp100(x: float) -> float:
    return float(np.clip(x, 0, 100))

def label_from_score(x: float):
    if x < 0.20:
        return "Low", "🟢", "good"
    if x < 0.35:
        return "Moderate", "🟠", "warning"
    return "High", "🔴", "bad"

def roi_to_score(roi_value: float) -> float:
    return clamp100((roi_value + 0.50) * 100)

def adaptation_score(league_jump: float, age: int, injury_risk_value: float) -> float:
    jump_penalty = abs(league_jump - 1.0) * 35
    age_penalty = max(0, age - 27) * 3.5
    injury_penalty = injury_risk_value * 25
    score = 100 - jump_penalty - age_penalty - injury_penalty
    return clamp100(score)

def verdict_badge(verdict: str) -> str:
    if verdict in ["Buy", "Approve", "Strong Buy"]:
        return f"<span style='padding:4px 10px;border-radius:999px;background:rgba(60,180,75,.18);border:1px solid rgba(60,180,75,.35);font-weight:600;'>{verdict}</span>"
    if verdict in ["Monitor", "Cautious", "Negotiate", "Approve with Conditions"]:
        return f"<span style='padding:4px 10px;border-radius:999px;background:rgba(255,165,0,.16);border:1px solid rgba(255,165,0,.35);font-weight:600;'>{verdict}</span>"
    return f"<span style='padding:4px 10px;border-radius:999px;background:rgba(220,20,60,.16);border:1px solid rgba(220,20,60,.35);font-weight:600;'>{verdict}</span>"

def make_outlook_chart(df: pd.DataFrame):
    fig = go.Figure()
    series = [
        "Sporting Score",
        "Tactical Fit Score",
        "Availability Score",
        "Financial Value Score",
    ]
    for col in series:
        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df[col],
                mode="lines+markers",
                name=col.replace(" Score", ""),
                hovertemplate="Year %{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Contract Year", tickmode="linear", dtick=1, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Score", range=[0, 110], gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig

STYLE_DIMENSIONS = [
    ("press_intensity", "Pressing"),
    ("build_up", "Build-up"),
    ("wing_play", "Wing Play"),
    ("directness", "Directness"),
    ("duel_focus", "Duels"),
]

def player_style_vector(p):
    return {
        "press_intensity": float(p["pressing"]),
        "build_up": float(p["creativity"]) * 0.6 + float(p["ball_carrying"]) * 0.4,
        "wing_play": float(p["pace"]) * 0.6 + float(p["ball_carrying"]) * 0.4,
        "directness": float(p["finishing"]) * 0.7 + float(p["pace"]) * 0.3,
        "duel_focus": float(p["duel_strength"]),
    }

def system_needs(t):
    return SYSTEM_ROLE_NEEDS.get(t["default_system"], SYSTEM_ROLE_NEEDS["4-3-3"])

def tactical_fit(p, t):
    ps = player_style_vector(p)
    needs = system_needs(t)
    dist, wsum = 0.0, 0.0
    for k, w in needs.items():
        dist += w * abs(ps[k] - float(t[k]))
        wsum += w
    fit = 1.0 - (dist / max(wsum, 1e-9))
    return clamp01(fit)

def fit_breakdown(p, t):
    ps = player_style_vector(p)
    out = []
    for key, label in STYLE_DIMENSIONS:
        team_val = float(t[key])
        player_val = float(ps[key])
        gap = abs(player_val - team_val)
        score = clamp01(1.0 - gap)
        out.append({
            "Dimension": label,
            "Player": player_val,
            "Team": team_val,
            "Fit": score,
            "Gap": gap
        })
    return pd.DataFrame(out)

def role_suggestion(position, sys):
    pos = str(position).upper()
    if sys == "4-3-3":
        if pos in ["RW", "LW"]:
            return "Inverted Winger / Wide Forward"
        if pos in ["ST", "CF"]:
            return "Pressing Forward / False 9"
        if pos in ["CM", "CAM"]:
            return "8 (Box-to-box) / Advanced Playmaker"
        if pos in ["CDM", "DM"]:
            return "6 (Anchor) / Deep-lying Playmaker"
        if pos in ["RB", "LB", "RWB", "LWB"]:
            return "Overlapping Fullback"
        if pos == "CB":
            return "Ball-playing CB"
    if sys == "4-2-3-1":
        if pos in ["CAM", "CM"]:
            return "10 (Creator) / 8-10 Hybrid"
        if pos in ["RW", "LW"]:
            return "Inside Forward"
        if pos in ["ST", "CF"]:
            return "Target / Pressing 9"
        if pos in ["CDM", "DM"]:
            return "Double Pivot (6/8)"
    if sys == "3-5-2":
        if pos in ["RWB", "LWB", "RB", "LB"]:
            return "Wingback (high engine)"
        if pos == "CB":
            return "Wide CB / Sweeper"
        if pos in ["ST", "CF"]:
            return "Strike Partner (runner/finisher)"
        if pos in ["CM", "CDM", "DM"]:
            return "Central Controller"
    return "Role depends on coach preferences"

def pros_and_risks(p, t):
    ps = player_style_vector(p)
    pros = []
    risks = []

    if ps["press_intensity"] > float(t["press_intensity"]) + 0.12:
        pros.append("Strong presser — can raise team intensity off the ball.")
    elif ps["press_intensity"] < float(t["press_intensity"]) - 0.12:
        risks.append("Pressing intensity may be below team requirements.")

    if ps["build_up"] > float(t["build_up"]) + 0.12:
        pros.append("Good in build-up — helps possession and progression.")
    elif ps["build_up"] < float(t["build_up"]) - 0.12:
        risks.append("Build-up contribution might be limited for this system.")

    if ps["wing_play"] > float(t["wing_play"]) + 0.12:
        pros.append("Strong wide threat — suits wing-play demands.")
    elif ps["wing_play"] < float(t["wing_play"]) - 0.12:
        risks.append("May not provide enough width/ball-carrying for the wings.")

    if ps["directness"] > float(t["directness"]) + 0.12:
        pros.append("Direct goal threat — adds end product.")
    elif ps["directness"] < float(t["directness"]) - 0.12:
        risks.append("May lack direct goal contribution in this role.")

    if ps["duel_focus"] > float(t["duel_focus"]) + 0.12:
        pros.append("Strong in duels — useful in physical matches and set pieces.")
    elif ps["duel_focus"] < float(t["duel_focus"]) - 0.12:
        risks.append("Duel strength could be a weakness in this environment.")

    age = int(p["age"])
    inj = float(p["injury_prone"])
    if age >= 30:
        risks.append("Age profile suggests reduced upside / resale potential.")
    if inj >= 0.35:
        risks.append("High injury-prone index — medical risk needs mitigation.")

    pros = pros[:3] if pros else ["No major tactical advantages detected vs club profile."]
    risks = risks[:3] if risks else ["No major tactical risks detected vs club profile."]
    return pros, risks

# =========================
# PREDICTION + RISK + ROI
# =========================
def predict_success(p, t, transfer_fee, weekly_wage, years):
    curr = p["current_league"]
    tgt = t["league"]

    league_jump = league_diff[tgt] / league_diff[curr]
    fit = tactical_fit(p, t)

    market_value = float(p["market_value_eur"])
    total_cost = transfer_fee + weekly_wage * 52 * years
    cost_pressure = total_cost / max(market_value, 1.0)

    X = pd.DataFrame([{
        "age": p["age"],
        "overall": p["overall"],
        "potential": p["potential"],
        "xg_p90": p["xg_p90"],
        "xa_p90": p["xa_p90"],
        "tackles_p90": p["tackles_p90"],
        "interceptions_p90": p["interceptions_p90"],
        "aerial_wins_p90": p["aerial_wins_p90"],
        "pace": p["pace"],
        "pressing": p["pressing"],
        "creativity": p["creativity"],
        "ball_carrying": p["ball_carrying"],
        "finishing": p["finishing"],
        "duel_strength": p["duel_strength"],
        "injury_prone": p["injury_prone"],
        "league_jump": league_jump,
        "tactical_fit": fit,
        "transfer_fee": transfer_fee,
        "weekly_wage": weekly_wage,
        "contract_years": years,
        "cost_pressure": cost_pressure,
    }])

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    base_proba = float(model.predict_proba(X)[0][1])

    wage_pressure = weekly_wage / 250_000
    years_pressure = max(0, years - 3) * 0.10

    adj = 0.0
    adj += -0.10 * (cost_pressure - 1.4)
    adj += -0.04 * (wage_pressure - 1.0)
    adj += -0.03 * years_pressure

    proba = float(np.clip(base_proba + adj, 0.02, 0.98))
    return proba, fit, league_jump, cost_pressure, base_proba, adj

def injury_components(p, tgt_league, weekly_wage):
    base = float(p["injury_prone"])
    age = int(p["age"])
    age_factor = max(0, (age - 26) * 0.015)
    intensity = float(league_intensity[tgt_league])
    intensity_factor = (intensity - 1.0) * 0.25
    wage_factor = min(0.08, weekly_wage / 1_000_000)
    raw = base + age_factor + intensity_factor + wage_factor
    return raw, base, age_factor, intensity_factor, wage_factor

def injury_risk(p, tgt_league, weekly_wage):
    raw, *_ = injury_components(p, tgt_league, weekly_wage)
    return float(np.clip(raw, 0, 1))

def expected_games_missed(inj_prob, season_games=50):
    missed = season_games * (inj_prob ** 1.3) * 0.6
    return int(np.clip(round(missed), 0, season_games))

def injury_recommendations(inj_label, years):
    rec = []
    if inj_label == "High":
        rec += [
            "Require full medical screening + recent workload data.",
            "Prefer performance bonuses over fixed wage.",
            "Avoid long guaranteed contracts (consider 1–2 year + option)."
        ]
    elif inj_label == "Moderate":
        rec += [
            "Add rotation plan and minutes management.",
            "Include appearance-based bonuses and fitness clauses."
        ]
    else:
        rec += [
            "Standard medical screening; risk profile is acceptable."
        ]
    if years >= 4:
        rec.append("Long contract: consider club option year to reduce downside.")
    return rec[:4]

def roi_projection(p, tgt_league, success_prob, injury_prob, transfer_fee, weekly_wage, years):
    total_cost = transfer_fee + weekly_wage * 52 * years
    base_value = float(p["market_value_eur"]) * float(league_mm[tgt_league])
    expected_resale = base_value * (0.6 + 0.8 * success_prob) * (1.0 - 0.35 * injury_prob)
    roi = (expected_resale - total_cost) / max(total_cost, 1.0)
    break_even_prob = float(np.clip(success_prob * (1.0 - 0.5 * injury_prob), 0, 1))
    return total_cost, expected_resale, roi, break_even_prob

def simulate_value_curve(start_value, years, success_prob, injury_prob, age, seed=42):
    rng = np.random.default_rng(seed)
    values = [float(start_value)]
    for _ in range(1, years + 1):
        age_drag = max(0, (age - 27)) * 0.015
        drift = (success_prob - 0.5) * 0.10 - injury_prob * 0.10 - age_drag
        shock = rng.normal(0, 0.08)
        growth = drift + shock
        next_val = max(1_000_000, values[-1] * (1.0 + growth))
        values.append(float(next_val))
    return pd.DataFrame({"Year": list(range(0, years + 1)), "Value (€)": values})

def simulate_committee_outlook(years, success_prob, tactical_fit_score, roi_value, availability_pct, league_jump, age):
    rows = []
    for year in range(1, years + 1):
        sporting_score = clamp100(success_prob * 100 - (year - 1) * 2.0 - max(0, age - 29) * 0.8)
        tactical_score = clamp100(tactical_fit_score * 100 - (year - 1) * 1.0)
        availability_score = clamp100(availability_pct - (year - 1) * 3.0 - max(0, age - 29) * 1.5)
        financial_score = clamp100(
            roi_to_score(roi_value)
            + (year - 1) * 4.0
            - abs(league_jump - 1.0) * 8.0
            - max(0, age - 29) * 1.2
        )
        rows.append({
            "Year": year,
            "Sporting Score": sporting_score,
            "Tactical Fit Score": tactical_score,
            "Availability Score": availability_score,
            "Financial Value Score": financial_score,
        })
    return pd.DataFrame(rows)

# =========================
# AI DECISION ROOM HELPERS
# =========================
def summarize_case(data):
    sp = float(data["success_probability"])
    tf = float(data["tactical_fit"])
    ir = float(data["injury_risk"])
    roi_val = float(data["roi"])

    if sp >= 0.70 and tf >= 0.70:
        sporting = "Strong"
    elif sp >= 0.60 or tf >= 0.60:
        sporting = "Mixed"
    else:
        sporting = "Weak"

    if ir < 0.20:
        medical = "Manageable"
    elif ir < 0.32:
        medical = "Cautious"
    else:
        medical = "High Risk"

    if roi_val >= 0.05:
        financial = "Efficient"
    elif roi_val >= -0.05:
        financial = "Borderline"
    else:
        financial = "Weak"

    return sporting, medical, financial

def ai_final_verdict(data):
    sporting, medical, financial = summarize_case(data)
    if sporting == "Strong" and medical == "Manageable" and financial == "Efficient":
        return "Approve"
    if sporting == "Strong" and financial in ["Borderline", "Weak"]:
        return "Approve with Conditions"
    if sporting == "Mixed" and financial == "Efficient" and medical != "High Risk":
        return "Monitor"
    if medical == "High Risk" and financial == "Weak":
        return "Reject"
    if sporting == "Weak":
        return "Reject"
    return "Monitor"

def ai_key_reason(data):
    sporting, medical, financial = summarize_case(data)
    if sporting == "Strong" and financial == "Weak":
        return "The sporting case is attractive, but the current deal structure does not justify the investment."
    if sporting == "Strong" and medical == "Cautious":
        return "The player fits the sporting model, but medical exposure lowers certainty around impact."
    if sporting == "Mixed" and financial == "Efficient":
        return "The numbers are acceptable, but the player is not yet a fully convincing strategic fit."
    if medical == "High Risk":
        return "Availability risk is too high for the current level of guaranteed commitment."
    if sporting == "Weak":
        return "The player does not currently project as a strong enough fit for this move."
    return "The overall case is not yet strong enough to move beyond a cautious position."

def ai_recommended_action(data):
    verdict = ai_final_verdict(data)
    roi_val = float(data["roi"])
    ir = float(data["injury_risk"])
    cp = float(data["cost_pressure"])

    if verdict == "Approve":
        return "Proceed, but keep the final structure disciplined and avoid unnecessary escalation."
    if verdict == "Approve with Conditions":
        return "Proceed only if the club can reduce guaranteed cost and protect downside through structure."
    if verdict == "Monitor":
        if roi_val < 0 or cp > 1.5:
            return "Do not advance at the current price; revisit only if valuation softens."
        if ir >= 0.28:
            return "Keep the player on the shortlist, but require stronger medical comfort before advancing."
        return "Continue monitoring rather than committing immediately."
    return "Do not advance unless the core risk profile changes materially."

def ai_boundary_thresholds(data):
    transfer_fee = float(data["transfer_fee"])
    weekly_wage = float(data["weekly_wage"])
    success_prob = float(data["success_probability"])
    injury_risk = float(data["injury_risk"])

    max_fee = transfer_fee
    max_wage = weekly_wage

    roi_val = float(data["roi"])
    cp = float(data["cost_pressure"])
    age = int(data["age"])

    if roi_val < 0:
        max_fee *= 0.88
        max_wage *= 0.90
    elif roi_val < 0.05:
        max_fee *= 0.95
        max_wage *= 0.96
    else:
        max_fee *= 1.03
        max_wage *= 1.03

    if injury_risk >= 0.28:
        max_wage *= 0.92
    if cp >= 1.75:
        max_fee *= 0.94
    if age >= 29:
        max_fee *= 0.96

    min_success = max(0.62, success_prob - 0.04)
    max_injury = min(0.30, max(0.22, injury_risk + 0.02))
    return max_fee, max_wage, min_success, max_injury

def ai_answer_final_recommendation(data):
    sporting, medical, financial = summarize_case(data)
    return {
        "title": "Final Recommendation",
        "headline": ai_final_verdict(data),
        "summary": f"The transfer case currently reads as Sporting: {sporting}, Medical: {medical}, Financial: {financial}.",
        "meaning": ai_key_reason(data),
        "action": ai_recommended_action(data),
    }

def ai_answer_biggest_risk(data):
    sporting, medical, financial = summarize_case(data)
    cp = float(data["cost_pressure"])
    ir = float(data["injury_risk"])
    tf = float(data["tactical_fit"])

    if financial == "Weak" or cp >= 1.60:
        headline = "Deal efficiency is the main risk."
        summary = "The player may still improve the team, but the current fee and wage structure weaken the value case."
        meaning = "This is less a player-quality problem and more a pricing problem. The sporting appeal is stronger than the economic structure."
        action = "The club should stay disciplined and avoid progressing unless the guaranteed cost comes down."
    elif medical == "High Risk" or ir >= 0.28:
        headline = "Availability is the main risk."
        summary = "The player can still fit the system, but missed minutes would hurt both impact and long-term value."
        meaning = "Even a strong profile becomes difficult to defend if the availability outlook is unstable over the contract period."
        action = "Protect downside with structure and avoid overcommitting guaranteed years or salary."
    elif sporting == "Mixed" or tf < 0.60:
        headline = "Tactical translation is the main risk."
        summary = "The player may be good individually, but the move still depends on role clarity and quick adaptation."
        meaning = "Without a clearly defined tactical role, the signing risks delivering less than the raw profile suggests."
        action = "Only proceed if the coaching view on role and usage is fully aligned."
    else:
        headline = "Execution is the main risk."
        summary = "The underlying player case is workable, but the structure still has to match the risk profile."
        meaning = "A reasonable target can still become a weak transfer if the negotiation is handled without discipline."
        action = "Keep the negotiation controlled and avoid unnecessary guarantees."

    return {
        "title": "Biggest Risk",
        "headline": headline,
        "summary": summary,
        "meaning": meaning,
        "action": action,
    }

def ai_answer_what_needs_to_change(data):
    verdict = ai_final_verdict(data)
    max_fee, max_wage, min_success, max_injury = ai_boundary_thresholds(data)

    if verdict == "Approve":
        headline = "The deal is already close to approval level."
        summary = "The main task is to preserve efficiency rather than rebuild the deal."
    elif verdict == "Approve with Conditions":
        headline = "The structure needs tightening."
        summary = "The player can be approved, but only if the club improves the guaranteed terms."
    elif verdict == "Monitor":
        headline = "Several conditions need to improve."
        summary = "The club should not move from monitoring to approval unless the structure becomes cleaner and more efficient."
    else:
        headline = "The deal would need material improvement."
        summary = "At its current level, the case is not strong enough to support progression."

    meaning = (
        f"A more acceptable structure requires four changes: lower the fee toward {euro(max_fee)}, "
        f"keep wages at or below {euro(max_wage)} per week, maintain projected success at {pct(min_success)} or better, "
        f"and keep injury risk at {pct(max_injury)} or lower."
    )

    action = "If those conditions are met, the move becomes much easier to defend from both a sporting and financial standpoint."

    return {
        "title": "What Needs to Change?",
        "headline": headline,
        "summary": summary,
        "meaning": meaning,
        "action": action,
    }

def ai_answer_negotiation_advice(data):
    max_fee, max_wage, _, _ = ai_boundary_thresholds(data)
    ir = float(data["injury_risk"])
    age = int(data["age"])

    headline = "Negotiate from discipline, not urgency."
    summary = f"The club should treat {euro(max_fee)} as the upper fee boundary and keep fixed wages around {euro(max_wage)} per week or lower."

    if ir >= 0.28 and age >= 29:
        meaning = (
            "Given the combination of medical exposure and age profile, this deal should lean heavily on protection: "
            "appearance-based incentives, controlled guarantees, and no unnecessarily long commitment."
        )
    elif ir >= 0.28:
        meaning = (
            "Medical risk means the structure should protect downside. More value should sit in performance "
            "and availability-based triggers rather than fully guaranteed compensation."
        )
    elif age >= 29:
        meaning = "Because long-term upside is more limited, flexibility matters more than a long fully guaranteed structure."
    else:
        meaning = (
            "The club can be moderately aggressive on structure, but only if guaranteed spend remains controlled "
            "and the deal does not drift upward without justification."
        )

    action = "The negotiation objective should be to preserve the sporting upside without paying a premium for certainty that the profile does not fully offer."

    return {
        "title": "Negotiation Advice",
        "headline": headline,
        "summary": summary,
        "meaning": meaning,
        "action": action,
    }

def generate_ai_card(mode_label, data):
    if mode_label == "Final Recommendation":
        return ai_answer_final_recommendation(data)
    if mode_label == "Biggest Risk":
        return ai_answer_biggest_risk(data)
    if mode_label == "What Needs to Change?":
        return ai_answer_what_needs_to_change(data)
    if mode_label == "Negotiation Advice":
        return ai_answer_negotiation_advice(data)
    return ai_answer_final_recommendation(data)

def render_ai_mini_strip(verdict, reason):
    st.markdown(
        f"""
        <div class="ai-mini-strip">
            <div class="label">AI Verdict</div>
            <div class="verdict">{verdict}</div>
            <div class="reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_ai_response_card(card, data):
    verdict = ai_final_verdict(data)
    sporting, medical, financial = summarize_case(data)

    st.markdown(f"### {card['headline']}")
    st.markdown(f"**Executive view:** {card['summary']}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="ai-metric-chip"><div class="k">Sporting</div><div class="v">{sporting}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="ai-metric-chip"><div class="k">Medical</div><div class="v">{medical}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="ai-metric-chip"><div class="k">Financial</div><div class="v">{financial}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="ai-metric-chip"><div class="k">Verdict</div><div class="v">{verdict}</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="ai-report-card">
                <div class="ai-report-card-label">What it means</div>
                <div class="ai-report-card-title">Committee interpretation</div>
                <div class="ai-report-card-text">{card["meaning"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="ai-report-card">
                <div class="ai-report-card-label">Recommended action</div>
                <div class="ai-report-card-title">Next club step</div>
                <div class="ai-report-card-text">{card["action"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ai-footnote">
            Built from success probability, tactical fit, injury risk, ROI, and deal structure already shown elsewhere in the dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# LLM HELPER
# =========================
def build_deal_context(data, board_verdict_value):
    sporting, medical, financial = summarize_case(data)
    return f"""
Current transfer deal context

Player: {data['player_name']}
Position: {data['position']}
Age: {data['age']}
Current Team: {data['current_team']}
Target Team: {data['target_team']}
Target System: {data['target_system']}
Suggested Role: {data['suggested_role']}

Sporting Assessment:
- Success Probability: {pct(data['success_probability'])}
- Tactical Fit: {pct(data['tactical_fit'])}
- Sporting Label: {sporting}

Medical Assessment:
- Injury Risk: {pct(data['injury_risk'])}
- Injury Label: {data['injury_label']}
- Expected Games Missed: {data['expected_games_missed']}
- Availability: {data['availability_percent']}%
- Medical Label: {medical}

Financial Assessment:
- Transfer Fee: {euro(data['transfer_fee'])}
- Weekly Wage: {euro(data['weekly_wage'])}
- Contract Length: {data['contract_years']} years
- Total Cost: {euro(data['total_cost'])}
- Expected Resale: {euro(data['expected_resale'])}
- ROI: {data['roi']*100:.1f}%
- Cost Pressure: {data['cost_pressure']:.2f}
- Financial Label: {financial}

Decision Layer:
- Board Verdict: {board_verdict_value}
- AI Verdict: {ai_final_verdict(data)}

Pros:
- {" | ".join(data['pros'])}

Risks:
- {" | ".join(data['risks'])}
""".strip()

def ask_transfer_ai(question, context):
    if not OPENAI_ENABLED or client is None:
        return "LLM assistant is not enabled in this session."

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional football transfer decision assistant. "
                        "Answer only using the provided deal context. "
                        "Do not invent outside scouting history, rumors, or facts not present in the context. "
                        "Be concise, professional, and realistic, like an internal club memo. "
                        "Use short paragraphs. "
                        "If the answer cannot be fully determined from the provided data, say that clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Deal context:\n\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response.output_text.strip()
    except Exception as e:
        return f"LLM error: {e}"

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("Transfer Setup")
    player_name = st.selectbox("Player", players["name"].tolist())
    target_team = st.selectbox("Target club", teams["team"].tolist())
    transfer_fee = st.number_input("Transfer fee (€)", min_value=0, value=40_000_000, step=1_000_000)
    weekly_wage = st.number_input("Weekly wage (€)", min_value=0, value=150_000, step=5_000)
    years = st.slider("Contract length (years)", 1, 5, 4)

# =========================
# SELECTED DATA
# =========================
p = players[players["name"] == player_name].iloc[0]
t = teams[teams["team"] == target_team].iloc[0]
tgt_league = t["league"]

success_prob, fit, league_jump, cost_pressure, base_ml, adj = predict_success(p, t, transfer_fee, weekly_wage, years)
inj_prob = injury_risk(p, tgt_league, weekly_wage)
total_cost, expected_resale, roi, break_even_prob = roi_projection(p, tgt_league, success_prob, inj_prob, transfer_fee, weekly_wage, years)

start_value = float(p["market_value_eur"]) * float(league_mm[tgt_league])
curve = simulate_value_curve(start_value, years, success_prob, inj_prob, int(p["age"]), seed=42)

pros, risks = pros_and_risks(p, t)
inj_label, inj_emoji, _ = label_from_score(inj_prob)
games_missed = expected_games_missed(inj_prob, season_games=50)
availability = int(np.clip(round((1 - games_missed / 50) * 100), 0, 100))

report_input = {
    "player_name": player_name,
    "position": str(p["position"]),
    "age": int(p["age"]),
    "current_team": str(p["current_team"]),
    "current_league": str(p["current_league"]),
    "target_team": str(target_team),
    "target_league": str(tgt_league),
    "target_system": str(t["default_system"]),
    "success_probability": float(success_prob),
    "tactical_fit": float(fit),
    "injury_risk": float(inj_prob),
    "injury_label": str(inj_label),
    "roi": float(roi),
    "break_even_probability": float(break_even_prob),
    "expected_resale": float(expected_resale),
    "total_cost": float(total_cost),
    "transfer_fee": float(transfer_fee),
    "weekly_wage": float(weekly_wage),
    "contract_years": int(years),
    "league_jump": float(league_jump),
    "cost_pressure": float(cost_pressure),
    "suggested_role": role_suggestion(p["position"], t["default_system"]),
    "pros": pros,
    "risks": risks,
    "expected_games_missed": int(games_missed),
    "availability_percent": int(availability),
}

# =========================
# DECISION LAYERS
# =========================
sporting_verdict = "Approve" if success_prob >= 0.70 and fit >= 0.70 else "Cautious"
financial_verdict = "Approve" if roi >= 0.05 else ("Cautious" if roi >= -0.05 else "Reject")
medical_verdict = "Approve" if inj_prob < 0.20 else ("Cautious" if inj_prob < 0.35 else "Reject")

if sporting_verdict == "Approve" and financial_verdict == "Approve" and medical_verdict != "Reject":
    board_verdict = "Buy"
elif financial_verdict == "Reject" or medical_verdict == "Reject":
    board_verdict = "Negotiate"
else:
    board_verdict = "Monitor"

outlook_df = simulate_committee_outlook(
    years=years,
    success_prob=success_prob,
    tactical_fit_score=fit,
    roi_value=roi,
    availability_pct=availability,
    league_jump=league_jump,
    age=int(p["age"]),
)

deal_context = build_deal_context(report_input, board_verdict)
ai_verdict = ai_final_verdict(report_input)
ai_reason = ai_key_reason(report_input)

if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "Final Recommendation"

# =========================
# SIDEBAR LLM ASSISTANT
# =========================
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Ask Transfer AI")

    llm_question = st.text_input(
        "Ask about this deal",
        placeholder="Why is this only monitor?",
        key="llm_question_input"
    )

    ask_llm = st.button("Ask Deal Assistant", use_container_width=True)

    if ask_llm and llm_question.strip():
        with st.spinner("Thinking..."):
            st.session_state["llm_answer"] = ask_transfer_ai(llm_question.strip(), deal_context)
            st.session_state["llm_last_question"] = llm_question.strip()

    if "llm_answer" in st.session_state:
        st.markdown("**Latest answer**")
        st.markdown(
            f"""
            <div class="sidebar-llm-card">
                <div style="font-size:11px; opacity:0.62; font-weight:700; letter-spacing:0.35px; text-transform:uppercase; margin-bottom:8px;">
                    {st.session_state.get("llm_last_question", "Deal question")}
                </div>
                {st.session_state["llm_answer"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Success",
    "🧩 Tactical Fit",
    "🏥 Injury",
    "💰 ROI",
    "🤖 Board Decision",
    "🧠 AI Decision Room"
])

with tab1:
    st.metric("Success probability", pct(success_prob))
    st.write(f"**{player_name}** ({p['position']}, age {p['age']})")
    st.write(f"From **{p['current_team']}** ({p['current_league']}) → **{target_team}** ({tgt_league})")
    st.write(f"League jump factor: **{league_jump:.2f}**")
    st.write(f"Cost pressure (deal cost / player value): **{cost_pressure:.2f}**")
    with st.expander("Show how fee/wage affects success (debug)"):
        st.write(f"Base ML probability: **{pct(base_ml)}**")
        st.write(f"Economic adjustment: **{adj:+.3f}**")
        st.write(f"Final probability: **{pct(success_prob)}**")

with tab2:
    st.metric("Tactical fit", pct(fit))
    st.write(f"Target system: **{t['default_system']}**")
    st.write(f"Suggested role: **{role_suggestion(p['position'], t['default_system'])}**")

    st.subheader("Fit breakdown")
    bdf = fit_breakdown(p, t)
    for _, row in bdf.iterrows():
        colA, colB = st.columns([1, 4])
        with colA:
            st.write(row["Dimension"])
        with colB:
            st.progress(float(row["Fit"]))
            st.caption(f"Player {row['Player']:.2f} vs Team {row['Team']:.2f} (gap {row['Gap']:.2f})")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Pros")
        for x in pros:
            st.write(f"- {x}")
    with c2:
        st.subheader("⚠️ Risks")
        for x in risks:
            st.write(f"- {x}")

with tab3:
    st.metric("Injury risk", pct(inj_prob))
    st.write(f"Risk level: **{inj_emoji} {inj_label}**")

    raw, base, age_f, inten_f, wage_f = injury_components(p, tgt_league, weekly_wage)
    drivers = pd.DataFrame([
        {"Driver": "Injury-prone index (player)", "Contribution": base},
        {"Driver": "Age factor", "Contribution": age_f},
        {"Driver": "League intensity factor", "Contribution": inten_f},
        {"Driver": "Wage/load factor", "Contribution": wage_f},
    ])
    drivers["Contribution"] = drivers["Contribution"].map(lambda x: f"{x:.3f}")

    st.subheader("Risk drivers")
    st.table(drivers)

    st.subheader("Availability estimate")
    st.write(f"- Expected games missed (season): **{games_missed} / 50**")
    st.write(f"- Expected availability: **{availability}%**")

    st.subheader("Recommendations")
    for r in injury_recommendations(inj_label, years):
        st.write(f"- {r}")

with tab4:
    st.metric("Transfer fee", euro(transfer_fee))
    st.metric("Weekly wage", euro(weekly_wage))
    st.metric("Total cost", euro(total_cost))
    st.metric("Expected resale", euro(expected_resale))
    st.metric("ROI", f"{roi*100:.1f}%")
    st.metric("Break-even probability", pct(break_even_prob))

    st.subheader("Projected market value trajectory")
    curve_plot = curve.copy()
    curve_plot["Value (€)"] = curve_plot["Value (€)"].round(0)

    chart = (
        alt.Chart(curve_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", scale=alt.Scale(domain=[0, years])),
            y=alt.Y("Value (€):Q", axis=alt.Axis(format=",.0f")),
            tooltip=[
                alt.Tooltip("Year:Q"),
                alt.Tooltip("Value (€):Q", format=",.0f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    curve_disp = curve.copy()
    curve_disp["Value (€)"] = curve_disp["Value (€)"].map(euro)
    st.table(curve_disp)

with tab5:
    st.subheader("🤖 Transfer Committee Review")
    st.markdown(
        f"""
        <div style="padding:12px 4px 2px 4px;">
            <div style="
                display:flex;
                align-items:center;
                gap:18px;
                border:1px solid rgba(255,255,255,0.10);
                border-radius:18px;
                padding:18px 20px;
                background: rgba(255,255,255,0.02);
            ">
                <div style="flex:0 0 84px;">
                    <img src="{PLAYER_PLACEHOLDER}" width="78" style="border-radius:12px;" />
                </div>
                <div style="flex:1;">
                    <div style="font-size:26px; font-weight:700; margin-bottom:3px;">{player_name}</div>
                    <div style="font-size:15px; opacity:0.82; margin-bottom:10px;">{p['position']} · Age {p['age']}</div>
                    <div style="font-size:15px; margin-bottom:9px;">{p['current_team']} → {target_team}</div>
                    <div style="font-size:14px; opacity:0.84; margin-bottom:6px;">
                        Fee: {euro(transfer_fee)} &nbsp;&nbsp;•&nbsp;&nbsp;
                        Wage: {euro(weekly_wage)} &nbsp;&nbsp;•&nbsp;&nbsp;
                        Contract: {years}y
                    </div>
                    <div style="font-size:13px; opacity:0.72;">
                        Suggested role: {role_suggestion(p["position"], t["default_system"])}
                    </div>
                </div>
                <div style="flex:0 0 150px; text-align:right;">
                    <div style="font-size:12px; opacity:0.7; margin-bottom:8px;">Board verdict</div>
                    <div>{verdict_badge(board_verdict)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Transfer outlook over contract years")
    st.plotly_chart(make_outlook_chart(outlook_df), use_container_width=True)

    row1, row2, row3, row4 = st.columns(4)
    with row1:
        st.markdown("#### ⚽ Sporting")
        st.markdown(verdict_badge(sporting_verdict), unsafe_allow_html=True)
        st.metric("Success", pct(success_prob))
        st.metric("Tactical Fit", pct(fit))
    with row2:
        st.markdown("#### 💰 Financial")
        st.markdown(verdict_badge(financial_verdict), unsafe_allow_html=True)
        st.metric("ROI", f"{roi*100:.1f}%")
        st.metric("Cost Pressure", f"{cost_pressure:.2f}")
    with row3:
        st.markdown("#### 🏥 Medical")
        st.markdown(verdict_badge(medical_verdict), unsafe_allow_html=True)
        st.metric("Injury Risk", pct(inj_prob))
        st.metric("Availability", f"{availability}%")
    with row4:
        st.markdown("#### 🏛 Board")
        st.markdown(verdict_badge(board_verdict), unsafe_allow_html=True)
        st.metric("League Jump", f"{league_jump:.2f}")
        st.metric("Resale", euro(expected_resale))

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("✅ Key positives")
        for x in pros:
            st.write(f"- {x}")
    with right:
        st.subheader("⚠️ Key blockers")
        for x in risks:
            st.write(f"- {x}")

with tab6:
    st.subheader("🧠 AI Decision Room")
    st.caption("A compact assistant layer that turns the other tabs into one clear football decision.")

    render_ai_mini_strip(ai_verdict, ai_reason)
    st.markdown("### Ask Transfer AI")

    mode_display = st.radio(
        "AI Mode",
        [
            "📌 Final recommendation",
            "⚠️ Biggest risk",
            "🔧 What needs to change?",
            "🤝 Negotiation advice",
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if mode_display == "📌 Final recommendation":
        st.session_state.ai_mode = "Final Recommendation"
    elif mode_display == "⚠️ Biggest risk":
        st.session_state.ai_mode = "Biggest Risk"
    elif mode_display == "🔧 What needs to change?":
        st.session_state.ai_mode = "What Needs to Change?"
    else:
        st.session_state.ai_mode = "Negotiation Advice"

    row1, row2 = st.columns([1, 0.24])
    with row1:
        st.text_input(
            "Selected question",
            value=st.session_state.ai_mode,
            disabled=True,
            label_visibility="collapsed"
        )
    with row2:
        ask_now = st.button("Ask AI", use_container_width=True)

    if ask_now or "ai_card_response" not in st.session_state:
        st.session_state.ai_card_response = generate_ai_card(st.session_state.ai_mode, report_input)

    render_ai_response_card(st.session_state.ai_card_response, report_input)