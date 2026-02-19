import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# --------- LEAGUES ----------
LEAGUES = [
    ("Premier League", 1.10, 1.10, 1.35),
    ("La Liga",        1.05, 1.00, 1.25),
    ("Bundesliga",     1.00, 1.05, 1.15),
    ("Serie A",        0.98, 0.95, 1.10),
    ("Ligue 1",        0.95, 1.00, 1.05),
]

leagues_df = pd.DataFrame(LEAGUES, columns=["league", "difficulty", "intensity", "market_multiplier"])
league_diff = dict(zip(leagues_df["league"], leagues_df["difficulty"]))
league_int  = dict(zip(leagues_df["league"], leagues_df["intensity"]))
league_mm   = dict(zip(leagues_df["league"], leagues_df["market_multiplier"]))

# --------- TEAMS (BIG CLUBS) ----------
# style numbers are 0..1 where higher means more of that style
TEAMS = [
    # EPL
    ("Liverpool", "Premier League", "4-3-3", 0.85, 0.75, 0.70, 0.55, 0.70, 0.55),
    ("Manchester City", "Premier League", "4-3-3", 0.70, 0.90, 0.65, 0.40, 0.75, 0.45),
    ("Arsenal", "Premier League", "4-3-3", 0.75, 0.80, 0.65, 0.45, 0.70, 0.45),
    ("Chelsea", "Premier League", "4-2-3-1", 0.70, 0.70, 0.60, 0.50, 0.65, 0.50),

    # La Liga
    ("Real Madrid", "La Liga", "4-3-3", 0.65, 0.80, 0.60, 0.55, 0.70, 0.45),
    ("FC Barcelona", "La Liga", "4-3-3", 0.65, 0.88, 0.60, 0.40, 0.70, 0.40),
    ("Atletico Madrid", "La Liga", "3-5-2", 0.75, 0.60, 0.45, 0.70, 0.55, 0.70),

    # Bundesliga
    ("Bayern Munich", "Bundesliga", "4-2-3-1", 0.70, 0.78, 0.60, 0.50, 0.70, 0.50),
    ("Borussia Dortmund", "Bundesliga", "4-3-3", 0.80, 0.70, 0.70, 0.55, 0.65, 0.55),

    # Serie A
    ("Inter", "Serie A", "3-5-2", 0.70, 0.65, 0.55, 0.60, 0.60, 0.65),
    ("AC Milan", "Serie A", "4-2-3-1", 0.75, 0.65, 0.65, 0.55, 0.65, 0.55),
    ("Juventus", "Serie A", "4-3-3", 0.65, 0.65, 0.55, 0.60, 0.60, 0.60),

    # Ligue 1
    ("Paris Saint-Germain", "Ligue 1", "4-3-3", 0.60, 0.75, 0.65, 0.45, 0.65, 0.40),
]

teams_df = pd.DataFrame(TEAMS, columns=[
    "team", "league", "default_system",
    "press_intensity", "build_up", "wing_play", "directness", "defensive_line", "duel_focus"
])

# --------- PLAYERS (REAL NAMES, MIX, NOT SMALL/UNKNOWN) ----------
# We'll assign them to the big clubs above (squad-ish). If duplicates/old names, it still works for prototype.
PLAYERS = [
    # Liverpool
    ("Mohamed Salah","RW",31,"Liverpool"),
    ("Virgil van Dijk","CB",32,"Liverpool"),
    ("Trent Alexander-Arnold","RB",25,"Liverpool"),
    ("Alisson","GK",31,"Liverpool"),
    ("Dominik Szoboszlai","CM",23,"Liverpool"),
    ("Luis Díaz","LW",27,"Liverpool"),
    ("Darwin Núñez","ST",24,"Liverpool"),
    ("Alexis Mac Allister","CM",25,"Liverpool"),

    # Man City
    ("Erling Haaland","ST",23,"Manchester City"),
    ("Kevin De Bruyne","CAM",32,"Manchester City"),
    ("Rodri","CDM",27,"Manchester City"),
    ("Phil Foden","LW",23,"Manchester City"),
    ("Bernardo Silva","RW",29,"Manchester City"),
    ("Rúben Dias","CB",26,"Manchester City"),
    ("Ederson","GK",30,"Manchester City"),

    # Arsenal
    ("Bukayo Saka","RW",22,"Arsenal"),
    ("Martin Ødegaard","CAM",25,"Arsenal"),
    ("Declan Rice","CDM",25,"Arsenal"),
    ("William Saliba","CB",22,"Arsenal"),
    ("Gabriel Jesus","ST",26,"Arsenal"),

    # Chelsea
    ("Cole Palmer","CAM",21,"Chelsea"),
    ("Enzo Fernández","CM",23,"Chelsea"),
    ("Reece James","RB",24,"Chelsea"),
    ("Moises Caicedo","CDM",22,"Chelsea"),
    ("Christopher Nkunku","ST",26,"Chelsea"),

    # Real Madrid
    ("Jude Bellingham","CM",20,"Real Madrid"),
    ("Vinícius Júnior","LW",23,"Real Madrid"),
    ("Kylian Mbappé","ST",25,"Real Madrid"),
    ("Federico Valverde","CM",25,"Real Madrid"),
    ("Antonio Rüdiger","CB",30,"Real Madrid"),
    ("Thibaut Courtois","GK",31,"Real Madrid"),

    # Barcelona
    ("Pedri","CM",21,"FC Barcelona"),
    ("Gavi","CM",19,"FC Barcelona"),
    ("Robert Lewandowski","ST",35,"FC Barcelona"),
    ("Ronald Araújo","CB",24,"FC Barcelona"),
    ("Frenkie de Jong","CM",26,"FC Barcelona"),
    ("Marc-André ter Stegen","GK",31,"FC Barcelona"),

    # Atletico
    ("Antoine Griezmann","ST",32,"Atletico Madrid"),
    ("Jan Oblak","GK",31,"Atletico Madrid"),
    ("José Giménez","CB",29,"Atletico Madrid"),

    # Bayern
    ("Harry Kane","ST",30,"Bayern Munich"),
    ("Jamal Musiala","CAM",20,"Bayern Munich"),
    ("Joshua Kimmich","CM",28,"Bayern Munich"),
    ("Manuel Neuer","GK",37,"Bayern Munich"),
    ("Alphonso Davies","LB",23,"Bayern Munich"),

    # Dortmund
    ("Marco Reus","CAM",34,"Borussia Dortmund"),
    ("Mats Hummels","CB",35,"Borussia Dortmund"),
    ("Gregor Kobel","GK",26,"Borussia Dortmund"),

    # Inter
    ("Lautaro Martínez","ST",26,"Inter"),
    ("Nicolò Barella","CM",27,"Inter"),
    ("Alessandro Bastoni","CB",24,"Inter"),

    # AC Milan
    ("Rafael Leão","LW",24,"AC Milan"),
    ("Theo Hernández","LB",26,"AC Milan"),
    ("Mike Maignan","GK",28,"AC Milan"),

    # Juventus
    ("Dušan Vlahović","ST",24,"Juventus"),
    ("Federico Chiesa","RW",26,"Juventus"),
    ("Bremer","CB",26,"Juventus"),

    # PSG
    ("Ousmane Dembélé","RW",26,"Paris Saint-Germain"),
    ("Achraf Hakimi","RB",25,"Paris Saint-Germain"),
    ("Gianluigi Donnarumma","GK",25,"Paris Saint-Germain"),
    ("Marquinhos","CB",29,"Paris Saint-Germain"),
]

# Expand to ~100 by adding well-known players (still not "small")
# (If you want, we can refine this list later; for now it fills the database.)
EXTRA = [
    ("Son Heung-min","LW",31,"Arsenal"),
    ("Bruno Fernandes","CAM",29,"Liverpool"),
    ("Casemiro","CDM",31,"Real Madrid"),
    ("Luka Modrić","CM",38,"Real Madrid"),
    ("Toni Kroos","CM",34,"Real Madrid"),
    ("João Félix","CAM",24,"FC Barcelona"),
    ("Ilkay Gündoğan","CM",33,"FC Barcelona"),
    ("Sergio Busquets","CDM",35,"Atletico Madrid"),
    ("Thomas Müller","CAM",34,"Bayern Munich"),
    ("Leroy Sané","LW",28,"Bayern Munich"),
    ("Kingsley Coman","LW",27,"Bayern Munich"),
    ("Joshua Zirkzee","ST",22,"Borussia Dortmund"),
    ("Nicolo Zaniolo","RW",24,"Juventus"),
    ("Paulo Dybala","CAM",30,"Juventus"),
    ("Romelu Lukaku","ST",30,"Inter"),
    ("Hakan Çalhanoğlu","CM",30,"Inter"),
    ("Olivier Giroud","ST",37,"AC Milan"),
    ("Adrien Rabiot","CM",28,"Juventus"),
    ("Sandro Tonali","CDM",23,"AC Milan"),
    ("Victor Osimhen","ST",25,"Paris Saint-Germain"),
    ("Khvicha Kvaratskhelia","LW",23,"Paris Saint-Germain"),
    ("Bernardo","RW",29,"Manchester City"),
    ("Martinelli","LW",22,"Arsenal"),
    ("Gabriel Martinelli","LW",22,"Arsenal"),
    ("Kai Havertz","ST",24,"Arsenal"),
    ("Gabriel Magalhães","CB",26,"Arsenal"),
    ("John Stones","CB",29,"Manchester City"),
    ("Kyle Walker","RB",33,"Manchester City"),
    ("Riyad Mahrez","RW",32,"Manchester City"),
    ("Karim Benzema","ST",36,"Real Madrid"),
    ("Thuram","ST",26,"Inter"),
    ("Taremi","ST",31,"Inter"),
    ("Leandro Paredes","CDM",29,"Paris Saint-Germain"),
    ("Donnarumma","GK",25,"Paris Saint-Germain"),
]

all_players = PLAYERS + EXTRA
# Ensure uniqueness by name
seen = set()
unique_players = []
for p in all_players:
    if p[0] not in seen:
        unique_players.append(p)
        seen.add(p[0])

# If still under 100, we duplicate "archetypes" by adding suffixes (only if needed)
# But likely this is already enough for prototype; we will sample/augment below.
base = unique_players.copy()

# Helper: generate traits/stat ranges by position
def gen_traits(position):
    # defaults
    pace = rng.normal(0.55, 0.15)
    pressing = rng.normal(0.55, 0.15)
    creativity = rng.normal(0.55, 0.15)
    ball = rng.normal(0.55, 0.15)
    finishing = rng.normal(0.55, 0.15)
    duel = rng.normal(0.55, 0.15)

    if position in ["ST"]:
        finishing += 0.15; pace += 0.05
    if position in ["RW","LW"]:
        pace += 0.12; ball += 0.10; creativity += 0.05
    if position in ["CAM"]:
        creativity += 0.15; ball += 0.08
    if position in ["CM"]:
        creativity += 0.05; pressing += 0.05
    if position in ["CDM"]:
        pressing += 0.12; duel += 0.12
    if position in ["CB"]:
        duel += 0.18; pressing += 0.05; pace -= 0.05
    if position in ["LB","RB"]:
        pace += 0.10; pressing += 0.08; ball += 0.05
    if position in ["GK"]:
        # keep neutral-ish; we'll ignore most traits later anyway
        pass

    # clamp 0..1
    vals = [pace, pressing, creativity, ball, finishing, duel]
    vals = [float(np.clip(v, 0, 1)) for v in vals]
    return vals

def gen_per90(position, overall):
    # overall influences outputs slightly
    boost = (overall - 75) / 25.0  # roughly -0.8..+0.6
    if position in ["ST"]:
        goals = np.clip(rng.normal(0.45 + 0.20*boost, 0.15), 0.05, 1.10)
        assists = np.clip(rng.normal(0.10 + 0.08*boost, 0.08), 0.00, 0.60)
        xg = goals * rng.uniform(1.05, 1.25)
        xa = assists * rng.uniform(1.05, 1.25)
        tackles = np.clip(rng.normal(0.5, 0.3), 0.0, 2.0)
        inter = np.clip(rng.normal(0.3, 0.2), 0.0, 1.5)
        aerial = np.clip(rng.normal(1.2, 0.7), 0.0, 4.0)
    elif position in ["RW","LW","CAM"]:
        goals = np.clip(rng.normal(0.25 + 0.15*boost, 0.14), 0.02, 0.85)
        assists = np.clip(rng.normal(0.18 + 0.12*boost, 0.10), 0.02, 0.80)
        xg = goals * rng.uniform(1.05, 1.25)
        xa = assists * rng.uniform(1.05, 1.30)
        tackles = np.clip(rng.normal(0.9, 0.5), 0.0, 3.0)
        inter = np.clip(rng.normal(0.6, 0.4), 0.0, 2.5)
        aerial = np.clip(rng.normal(0.4, 0.4), 0.0, 2.0)
    elif position in ["CM"]:
        goals = np.clip(rng.normal(0.10 + 0.08*boost, 0.07), 0.0, 0.35)
        assists = np.clip(rng.normal(0.12 + 0.10*boost, 0.08), 0.0, 0.50)
        xg = goals * rng.uniform(1.05, 1.20)
        xa = assists * rng.uniform(1.05, 1.25)
        tackles = np.clip(rng.normal(1.6, 0.5), 0.4, 3.6)
        inter = np.clip(rng.normal(1.1, 0.4), 0.2, 2.8)
        aerial = np.clip(rng.normal(0.7, 0.5), 0.0, 2.5)
    elif position in ["CDM","LB","RB","CB"]:
        goals = np.clip(rng.normal(0.05, 0.04), 0.0, 0.25)
        assists = np.clip(rng.normal(0.06, 0.06), 0.0, 0.40)
        xg = goals * rng.uniform(1.05, 1.25)
        xa = assists * rng.uniform(1.05, 1.25)
        tackles = np.clip(rng.normal(2.0 + 0.4*boost, 0.7), 0.3, 5.0)
        inter = np.clip(rng.normal(1.4 + 0.3*boost, 0.6), 0.2, 4.0)
        aerial = np.clip(rng.normal(1.6 + (0.8 if position=="CB" else 0.0), 0.8), 0.0, 5.5)
    else:  # GK
        goals=0.0; assists=0.0; xg=0.0; xa=0.0
        tackles=np.clip(rng.normal(0.05,0.05),0.0,0.3)
        inter=np.clip(rng.normal(0.05,0.05),0.0,0.3)
        aerial=np.clip(rng.normal(0.05,0.05),0.0,0.3)
    return float(goals), float(assists), float(xg), float(xa), float(tackles), float(inter), float(aerial)

def value_wage(overall, potential, age, league):
    # value curve: peak ~26, league market multiplier
    age_penalty = np.exp(-((age - 26) / 7) ** 2)
    quality = (overall - 55) / 35  # 0..1
    base = 2_000_000 + 90_000_000 * quality * age_penalty
    pot_bonus = max(0, potential - overall) * 600_000
    v = (base + pot_bonus) * league_mm[league]
    v = float(np.clip(v + rng.normal(0, 3_000_000), 500_000, 180_000_000))
    # wage roughly tied to value
    w = float(np.clip(v / 900 + rng.normal(0, 5000), 5_000, 450_000))
    return round(v), int(w)

# Build players to exactly 100 by sampling/augmenting base list
# If base list < 100, we create additional "known-type" duplicates with suffixes (for prototype only)
players_list = base.copy()
suffix = 2
while len(players_list) < 100:
    name, pos, age, team = players_list[len(players_list) % len(base)]
    players_list.append((f"{name} {suffix}", pos, age, team))
    suffix += 1

players_list = players_list[:100]

# Generate players_app.csv
team_to_league = dict(zip(teams_df["team"], teams_df["league"]))

rows = []
for i, (name, pos, age, team) in enumerate(players_list, start=1):
    league = team_to_league.get(team, "Premier League")

    # overall/potential: make stars higher, but keep mix
    # We approximate by using name length hash for deterministic variety
    h = sum(ord(c) for c in name) % 100
    overall = int(np.clip(72 + (h/100)*18 + rng.normal(0, 3), 65, 92))
    potential = int(np.clip(overall + rng.normal(2, 4), overall, 94))

    pace, pressing, creativity, ball, finishing, duel = gen_traits(pos)

    minutes = int(np.clip(rng.normal(2200, 650), 300, 3300))
    # injury days: higher with age + some randomness
    injury_days = int(np.clip(rng.normal(20 + (age-24)*1.8, 22), 0, 200))
    injury_prone = float(np.clip((injury_days/200) + (age-28)*0.02, 0, 1))

    g,a,xg,xa,t,inter,aer = gen_per90(pos, overall)
    mv, wage = value_wage(overall, potential, age, league)

    rows.append({
        "player_id": i,
        "name": name,
        "age": age,
        "position": pos,
        "current_team": team,
        "current_league": league,
        "overall": overall,
        "potential": potential,
        "goals_p90": round(g,3),
        "assists_p90": round(a,3),
        "xg_p90": round(xg,3),
        "xa_p90": round(xa,3),
        "tackles_p90": round(t,3),
        "interceptions_p90": round(inter,3),
        "aerial_wins_p90": round(aer,3),
        "pace": round(pace,3),
        "pressing": round(pressing,3),
        "creativity": round(creativity,3),
        "ball_carrying": round(ball,3),
        "finishing": round(finishing,3),
        "duel_strength": round(duel,3),
        "minutes_last_season": minutes,
        "injury_days_last2y": injury_days,
        "injury_prone": round(injury_prone,3),
        "market_value_eur": mv,
        "weekly_wage_eur": wage
    })

players_df = pd.DataFrame(rows)

# Save outputs
leagues_df.to_csv("leagues.csv", index=False)
teams_df.to_csv("teams.csv", index=False)
players_df.to_csv("players_app.csv", index=False)

print("Created leagues.csv, teams.csv, players_app.csv (100 players)")
