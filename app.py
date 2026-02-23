import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Transfer Decision Engine", layout="wide")

st.title("⚽ AI Transfer Decision Engine")
st.caption("Select a player + target club to predict success probability, injury risk, tactical fit, and ROI.")

# =========================
# LOAD DATA + MODEL
# =========================
players = pd.read_csv("players_app.csv")
teams = pd.read_csv("teams.csv")
leagues = pd.read_csv("leagues.csv")

# ---- STREAMLIT CLOUD FIX (ONLY CHANGE HERE) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    train_df = pd.read_csv(os.path.join(BASE_DIR, "transfers_train.csv"))

    # ✅ Your CSV uses 'transfer_success' as the label column
    y = train_df["transfer_success"]
    X = train_df.drop(columns=["transfer_success"])

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
# ---- END FIX ----

league_diff = dict(zip(leagues["league"], leagues["difficulty"]))
league_intensity = dict(zip(leagues["league"], leagues["intensity"]))
league_mm = dict(zip(leagues["league"], leagues["market_multiplier"]))

SYSTEM_ROLE_NEEDS = {
    "4-3-3":  {"wing_play": 0.8, "press_intensity": 0.7, "build_up": 0.7, "directness": 0.5, "duel_focus": 0.5},
    "4-2-3-1":{"wing_play": 0.7, "press_intensity": 0.6, "build_up": 0.75,"directness": 0.55,"duel_focus": 0.55},
    "3-5-2":  {"wing_play": 0.55,"press_intensity": 0.65,"build_up": 0.6, "directness": 0.65,"duel_focus": 0.7},
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

def label_from_score(x: float):
    # returns (label, emoji, color_name)
    if x < 0.20:
        return "Low", "🟢", "good"
    if x < 0.35:
        return "Moderate", "🟠", "warning"
    return "High", "🔴", "bad"

# =========================
# STYLE / FEATURES
# =========================
STYLE_DIMENSIONS = [
    ("press_intensity", "Pressing"),
    ("build_up", "Build-up"),
    ("wing_play", "Wing Play"),
    ("directness", "Directness"),
    ("duel_focus", "Duels"),
]

def player_style_vector(p):
    # all 0..1
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
    # team profile columns in teams.csv are expected as t["press_intensity"] etc.
    out = []
    for key, label in STYLE_DIMENSIONS:
        team_val = float(t[key])
        player_val = float(ps[key])
        gap = abs(player_val - team_val)
        score = clamp01(1.0 - gap)  # 1 perfect, 0 worst
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
        if pos in ["CB"]:
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
        if pos in ["CB"]:
            return "Wide CB / Sweeper"
        if pos in ["ST", "CF"]:
            return "Strike Partner (runner/finisher)"
        if pos in ["CM", "CDM", "DM"]:
            return "Central Controller"
    return "Role depends on coach preferences"

def pros_and_risks(p, t):
    ps = player_style_vector(p)
    needs = system_needs(t)

    pros = []
    risks = []

    # Big matches
    if ps["press_intensity"] > float(t["press_intensity"]) + 0.12:
        pros.append("Strong presser — can raise team’s intensity off the ball.")
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
        pros.append("Strong in duels — good for physical matches and set pieces.")
    elif ps["duel_focus"] < float(t["duel_focus"]) - 0.12:
        risks.append("Duel strength could be a weakness in this environment.")

    # Age & injury
    age = int(p["age"])
    inj = float(p["injury_prone"])
    if age >= 30:
        risks.append("Age profile suggests reduced upside / resale potential.")
    if inj >= 0.35:
        risks.append("High injury-prone index — medical risk needs mitigation.")

    # Keep it neat
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

    # Visible economic realism layer
    wage_pressure = weekly_wage / 250_000  # 1.0 around 250k/week
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
    # simple mapping: 0.1 -> 2-3 games, 0.3 -> ~8 games, 0.5 -> ~15 games
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

# =========================
# SIDEBAR
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

total_cost, expected_resale, roi, break_even_prob = roi_projection(
    p, tgt_league, success_prob, inj_prob, transfer_fee, weekly_wage, years
)

start_value = float(p["market_value_eur"]) * float(league_mm[tgt_league])
curve = simulate_value_curve(start_value, years, success_prob, inj_prob, int(p["age"]), seed=42)

# =========================
# UI OUTPUT
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Success", "🧩 Tactical Fit", "🏥 Injury", "💰 ROI"])

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

    # Bars per dimension
    for _, row in bdf.iterrows():
        colA, colB = st.columns([1, 4])
        with colA:
            st.write(row["Dimension"])
        with colB:
            st.progress(float(row["Fit"]))
            st.caption(f"Player {row['Player']:.2f} vs Team {row['Team']:.2f} (gap {row['Gap']:.2f})")

    pros, risks = pros_and_risks(p, t)
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
    inj_label, inj_emoji, _ = label_from_score(inj_prob)
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

    games_missed = expected_games_missed(inj_prob, season_games=50)
    availability = int(np.clip(round((1 - games_missed / 50) * 100), 0, 100))
    st.subheader("Availability estimate")
    st.write(f"- Expected games missed (season): **{games_missed} / 50**")
    st.write(f"- Expected availability: **{availability}%**")

    st.subheader("Recommendations")
    recs = injury_recommendations(inj_label, years)
    for r in recs:
        st.write(f"- {r}")

with tab4:
    st.metric("Transfer fee", euro(transfer_fee))
    st.metric("Weekly wage", euro(weekly_wage))
    st.metric("Total cost", euro(total_cost))
    st.metric("Expected resale", euro(expected_resale))
    st.metric("ROI", f"{roi*100:.1f}%")
    st.metric("Break-even probability", pct(break_even_prob))

    st.subheader("Projected market value trajectory")

    # --- UPDATED GRAPH (commas in axis + tooltip) ---
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
    curve_disp["Value (€)"] = curve_disp["Value (€)"].map(lambda x: euro(x))
    st.table(curve_disp)





