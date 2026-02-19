import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

players = pd.read_csv("players_app.csv")
teams = pd.read_csv("teams.csv")
leagues = pd.read_csv("leagues.csv")

league_diff = dict(zip(leagues["league"], leagues["difficulty"]))

# Map system roles (very simplified)
SYSTEM_ROLE_NEEDS = {
    "4-3-3":  {"wing_play": 0.8, "press_intensity": 0.7, "build_up": 0.7, "directness": 0.5, "duel_focus": 0.5},
    "4-2-3-1":{"wing_play": 0.7, "press_intensity": 0.6, "build_up": 0.75,"directness": 0.55,"duel_focus": 0.55},
    "3-5-2":  {"wing_play": 0.55,"press_intensity": 0.65,"build_up": 0.6, "directness": 0.65,"duel_focus": 0.7},
}

def tactical_fit(player_row, team_row):
    # player traits 0..1
    # team style 0..1
    # weighted similarity -> 0..1
    p = player_row
    t = team_row

    # Convert player traits into style “signals”
    player_style = {
        "press_intensity": p["pressing"],
        "build_up":        p["creativity"] * 0.6 + p["ball_carrying"] * 0.4,
        "wing_play":       p["pace"] * 0.6 + p["ball_carrying"] * 0.4,
        "directness":      p["finishing"] * 0.7 + p["pace"] * 0.3,
        "duel_focus":      p["duel_strength"],
    }

    sys_need = SYSTEM_ROLE_NEEDS.get(t["default_system"], SYSTEM_ROLE_NEEDS["4-3-3"])

    # Weighted distance (smaller is better)
    dist = 0.0
    wsum = 0.0
    for k, w in sys_need.items():
        dist += w * abs(player_style[k] - t[k])
        wsum += w
    fit = 1.0 - (dist / max(wsum, 1e-9))
    return float(np.clip(fit, 0, 1))

rows = []
N_SCENARIOS = 1800

for _ in range(N_SCENARIOS):
    p = players.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
    team = teams.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]

    curr_league = p["current_league"]
    tgt_league = team["league"]

    league_jump = league_diff[tgt_league] / league_diff[curr_league]
    fit = tactical_fit(p, team)

    # Hidden "true" success score -> used to create label
    score = (
        0.10 * (p["overall"] - 75)
        + 1.2 * p["xg_p90"]
        + 1.0 * p["xa_p90"]
        + 0.25 * (p["tackles_p90"] + p["interceptions_p90"]) / 2
        - 0.9 * p["injury_prone"]
        - 0.04 * (p["age"] - 26) ** 2
        - 2.0 * (league_jump - 1.0) ** 2
        + 1.2 * fit
    )

    prob = 1 / (1 + np.exp(-score))
    success = rng.binomial(1, prob)

    rows.append({
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
        "target_team": team["team"],
        "target_league": tgt_league,
        "transfer_success": success
    })

df = pd.DataFrame(rows)
df.to_csv("transfers_train.csv", index=False)
print("Created transfers_train.csv with", len(df), "rows")
