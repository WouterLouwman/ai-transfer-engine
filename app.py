import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Transfer Decision Engine", layout="wide")

st.title("⚽ AI Transfer Decision Engine (Prototype)")
st.caption("Goal: help scouts/sporting directors estimate success, injury risk, and ROI for a transfer.")

# Tiny placeholder data (we replace with a bigger CSV later)
players = pd.DataFrame([
    {"name": "Player 1", "age": 24, "position": "ST", "current_league": "La Liga", "market_value_eur": 35_000_000},
    {"name": "Player 2", "age": 28, "position": "CM", "current_league": "Bundesliga", "market_value_eur": 28_000_000},
    {"name": "Player 3", "age": 22, "position": "CB", "current_league": "Serie A", "market_value_eur": 22_000_000},
])

leagues = pd.DataFrame([
    {"league": "Premier League", "difficulty": 1.10, "intensity": 1.10, "market_multiplier": 1.35},
    {"league": "La Liga",        "difficulty": 1.05, "intensity": 1.00, "market_multiplier": 1.25},
    {"league": "Bundesliga",     "difficulty": 1.00, "intensity": 1.05, "market_multiplier": 1.15},
    {"league": "Serie A",        "difficulty": 0.98, "intensity": 0.95, "market_multiplier": 1.10},
    {"league": "Ligue 1",        "difficulty": 0.95, "intensity": 1.00, "market_multiplier": 1.05},
])

SYSTEMS = ["4-3-3", "4-2-3-1", "3-5-2"]

with st.sidebar:
    st.header("Transfer Setup")
    player_name = st.selectbox("Select player", players["name"].tolist())
    target_league = st.selectbox("Target league", leagues["league"].tolist())
    system = st.selectbox("Tactical system (optional)", SYSTEMS)
    transfer_fee = st.number_input("Transfer fee (€)", min_value=0, value=30_000_000, step=1_000_000)
    weekly_wage = st.number_input("Weekly wage (€)", min_value=0, value=120_000, step=5_000)
    contract_years = st.slider("Contract length (years)", 1, 5, 4)
    risk_tol = st.select_slider("Risk tolerance (optional)", ["Low", "Medium", "High"], value="Medium")

player = players[players["name"] == player_name].iloc[0]
tgt = leagues[leagues["league"] == target_league].iloc[0]

# Placeholder model outputs (next step we replace with real logic)
success_prob = 0.62
injury_risk = 0.28

total_cost = transfer_fee + weekly_wage * 52 * contract_years
expected_resale = player["market_value_eur"] * float(tgt["market_multiplier"]) * 1.10
break_even_prob = 0.55
roi = (expected_resale - total_cost) / max(total_cost, 1)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🏥 Injury Risk", "💰 Financial", "✅ Decision"])

with tab1:
    st.metric("Success probability", f"{success_prob*100:.1f}%")
    st.write(f"**{player_name}** | {player['position']} | Current league: {player['current_league']}")
    st.info("Placeholder for now — next step we compute success from stats + league difficulty + tactical fit.")

with tab2:
    band = "Low" if injury_risk < 0.33 else "Medium" if injury_risk < 0.66 else "High"
    st.metric("Injury risk", f"{injury_risk*100:.1f}% ({band})")
    st.info("Placeholder for now — next step we compute this from age + minutes load + target league intensity.")

with tab3:
    st.metric("Total cost", f"€{total_cost:,.0f}")
    st.metric("Expected resale value", f"€{expected_resale:,.0f}")
    st.metric("Break-even probability", f"{break_even_prob*100:.1f}%")
    st.metric("ROI", f"{roi*100:.1f}%")

with tab4:
    decision = "SIGN ✅" if (success_prob > 0.55 and break_even_prob > 0.50) else "RISKY ⚠️"
    st.subheader(f"Recommendation: {decision}")
    st.write(f"System: **{system}** | Risk tolerance: **{risk_tol}**")
