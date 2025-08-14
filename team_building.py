# team_building.py — スコア計算・厳密割当・8パターン生成（UI向けラベル適用）
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

LANES = ["TOP", "JG", "MID", "ADC", "SUP"]

# ランク変換規則
RANK_ORDER = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]
RANK_BASE = {tier: i for i, tier in enumerate(RANK_ORDER, start=1)}  # 1..10
DIV_BONUS = {"IV": 1, "III": 2, "II": 3, "I": 4}  # 1..4


@dataclass
class Player:
    name: str
    rank_str: str
    mmr: float
    level: float
    lane_prefs: List[str]  # 上から順
    rank_score: float = 0.0
    mmr_n: float = 0.0
    level_n: float = 0.0
    lane_weight_map: Dict[str, float] = None


def parse_rank_to_score(rank_str: str) -> float:
    if not isinstance(rank_str, str) or not rank_str.strip():
        return 0.0
    raw = rank_str.strip().upper()
    parts = raw.split()
    tier = parts[0]
    div = parts[1] if len(parts) > 1 else None

    tier_val = RANK_BASE.get(tier, 0)
    div_val = 0
    if div is not None:
        d = (div.replace("1", "I").replace("2", "II").replace("3", "III").replace("4", "IV"))
        div_val = DIV_BONUS.get(d, 0)

    raw_score = tier_val + (div_val / 4.0)  # 1.0..11.0
    return (raw_score - 1.0) / (11.0 - 1.0)


def minmax_normalize(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    smin, smax = series.min(), series.max()
    return (series - smin) / (smax - smin)


def build_players(df: pd.DataFrame) -> List[Player]:
    players: List[Player] = []
    pref_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    mmr_n = minmax_normalize(df["mmr"].astype(float))
    level_n = minmax_normalize(df["level"].astype(float))

    for i, row in df.iterrows():
        lane_list = [str(row.get(f"lane{k}", "")).strip().upper() for k in range(1, 6)]
        lane_weight_map = {}
        for idx, lane in enumerate(lane_list):
            if lane in LANES:
                lane_weight_map[lane] = pref_weights[idx]
        p = Player(
            name=str(row["name"]).strip(),
            rank_str=str(row["rank"]).strip(),
            mmr=float(row["mmr"]),
            level=float(row["level"]),
            lane_prefs=lane_list,
        )
        p.rank_score = parse_rank_to_score(p.rank_str)
        p.mmr_n = float(mmr_n.loc[i])
        p.level_n = float(level_n.loc[i])
        p.lane_weight_map = lane_weight_map
        players.append(p)
    return players


def skill_score(p: Player, lane: str, w_rank: float, w_mmr: float, w_lane: float, w_level: float) -> float:
    lane_w = p.lane_weight_map.get(lane, 0.0)
    return w_rank * p.rank_score + w_mmr * p.mmr_n + w_lane * lane_w + w_level * p.level_n


def make_skill_matrix(players: List[Player], weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    w_rank, w_mmr, w_lane, w_level = weights
    data = {lane: [skill_score(p, lane, w_rank, w_mmr, w_lane, w_level) for p in players] for lane in LANES}
    df = pd.DataFrame(data, index=[p.name for p in players])
    return df


# ===================== 厳密割当（各レーン2人を保証） =========================
def solve_lane_assignment(score_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    assert len(score_df.index) == 10, "solve_lane_assignment requires exactly 10 players."
    players = list(score_df.index)
    lanes = LANES
    S = {(pi, l): float(score_df.loc[players[pi], l]) for pi in range(10) for l in lanes}

    order = sorted(range(10), key=lambda pi: -max(S[(pi, l)] for l in lanes))

    def greedy_upper_bound(idx: int, lanes_left: Dict[str, int]) -> float:
        counts = lanes_left.copy()
        total = 0.0
        for k in range(idx, 10):
            pi = order[k]
            best_sc = -1.0
            best_lane = None
            for l in lanes:
                if counts[l] > 0:
                    sc = S[(pi, l)]
                    if sc > best_sc:
                        best_sc, best_lane = sc, l
            if best_lane is None:
                continue
            total += best_sc
            counts[best_lane] -= 1
        return total

    best = {"score": -1.0, "assign": None}

    def dfs(idx: int, lanes_left: Dict[str, int], cur: List[Tuple[str, str, float]], cur_sum: float):
        nonlocal best
        if cur_sum + greedy_upper_bound(idx, lanes_left) <= best["score"]:
            return
        if idx == 10:
            if all(v == 0 for v in lanes_left.values()):
                if cur_sum > best["score"]:
                    best["score"], best["assign"] = cur_sum, cur[:]
            return

        pi = order[idx]
        options = [l for l in lanes if lanes_left[l] > 0]
        options.sort(key=lambda l: S[(pi, l)], reverse=True)
        for l in options:
            sc = S[(pi, l)]
            cur.append((players[pi], l, sc))
            lanes_left[l] -= 1
            dfs(idx + 1, lanes_left, cur, cur_sum + sc)
            lanes_left[l] += 1
            cur.pop()

    lanes_left0 = {l: 2 for l in lanes}
    dfs(0, lanes_left0, [], 0.0)
    if not best["assign"]:
        raise RuntimeError("Feasible assignment not found.")
    return sorted(best["assign"], key=lambda x: (x[1], -x[2]))


def split_teams_balanced_from_assignment(assign_10: List[Tuple[str, str, float]]
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    by_lane = {l: [] for l in LANES}
    for p, l, s in assign_10:
        by_lane[l].append((p, l, s))
    for l in LANES:
        if len(by_lane[l]) != 2:
            raise RuntimeError(f"Lane {l} does not have exactly 2 players.")

    best = None
    from itertools import product
    for bits in product([0, 1], repeat=5):
        A, B = [], []
        for i, l in enumerate(LANES):
            a = by_lane[l][bits[i]]
            b = by_lane[l][1 - bits[i]]
            A.append(a); B.append(b)
        sumA = sum(x[2] for x in A)
        sumB = sum(x[2] for x in B)
        diff = abs(sumA - sumB)
        if (best is None) or (diff < best[0]):
            best = (diff, A, B)
    return best[1], best[2]


# =============== 追加パターン実装用ヘルパ & 分割方針 ========================
def _name2player_map(players: List[Player]) -> Dict[str, Player]:
    return {p.name: p for p in players}

def split_teams_minimize_lane_diff(assign_10: List[Tuple[str, str, float]]
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    by_lane = {l: [] for l in LANES}
    for p, l, s in assign_10:
        by_lane[l].append((p, l, s))
    for l in LANES:
        if len(by_lane[l]) != 2:
            raise RuntimeError(f"Lane {l} does not have exactly 2 players.")

    best = None
    from itertools import product
    for bits in product([0, 1], repeat=5):
        A, B = [], []
        lane_diff_sum = 0.0
        for i, l in enumerate(LANES):
            a = by_lane[l][bits[i]]
            b = by_lane[l][1 - bits[i]]
            A.append(a); B.append(b)
            lane_diff_sum += abs(a[2] - b[2])
        diff_total = abs(sum(x[2] for x in A) - sum(x[2] for x in B))
        key = (lane_diff_sum, diff_total)
        if (best is None) or (key < best[0]):
            best = (key, A, B)
    return best[1], best[2]

def split_teams_anchor_objective(assign_10: List[Tuple[str, str, float]]
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    by_lane = {l: [] for l in LANES}
    for p, l, s in assign_10:
        by_lane[l].append((p, l, s))
    for l in LANES:
        if len(by_lane[l]) != 2:
            raise RuntimeError(f"Lane {l} does not have exactly 2 players.")

    best = None
    from itertools import product
    for bits in product([0, 1], repeat=5):
        A, B = [], []
        for i, l in enumerate(LANES):
            a = by_lane[l][bits[i]]
            b = by_lane[l][1 - bits[i]]
            A.append(a); B.append(b)
        laneA = {l: next(x[2] for x in A if x[1]==l) for l in LANES}
        laneB = {l: next(x[2] for x in B if x[1]==l) for l in LANES}
        anchor_score = min(laneA["JG"], laneB["JG"]) + min(laneA["MID"], laneB["MID"])
        total_diff = abs(sum(x[2] for x in A) - sum(x[2] for x in B))
        key = (-anchor_score, total_diff)
        if (best is None) or (key < best[0]):
            best = (key, A, B)
    return best[1], best[2]

def _emphasize_prefs_df(players: List[Player], base_df: pd.DataFrame,
                        weights: Tuple[float,float,float,float,float]) -> pd.DataFrame:
    name2p = _name2player_map(players)
    df = base_df.copy()
    for name in df.index:
        p = name2p[name]
        prefs = [l for l in p.lane_prefs if l in LANES]
        for idx, lane in enumerate(prefs):
            mult = weights[idx] if idx < len(weights) else 1.0
            df.loc[name, lane] *= mult
    return df.clip(0.0, 1.0)


# ============================ 3基本 + 追加5パターン ===========================
def pattern_best(players: List[Player], score_df: pd.DataFrame):
    assign = solve_lane_assignment(score_df)
    return split_teams_balanced_from_assignment(assign)

def pattern_handicap(players: List[Player], score_df: pd.DataFrame):
    df = score_df.copy()
    overall = df.max(axis=1).sort_values(ascending=False)
    k = max(1, int(round(len(overall) * 0.2)))
    top_names = list(overall.head(k).index)

    name2player = _name2player_map(players)
    for n in top_names:
        p = name2player.get(n); 
        if not p: continue
        prefs = [l for l in p.lane_prefs if l in LANES]
        best_two = prefs[:2]; worst_three = prefs[2:]
        for l in best_two:   df.loc[n, l] *= 0.70
        for l in worst_three: df.loc[n, l] = min(df.loc[n, l] * 1.05, 1.0)

    assign = solve_lane_assignment(df)
    return split_teams_balanced_from_assignment(assign)

def pattern_worst(players: List[Player], score_df: pd.DataFrame):
    df_bad = 1.0 - score_df.clip(0.0, 1.0)
    rng = np.random.RandomState(42)
    df_bad = df_bad + (1e-6 * rng.rand(*df_bad.shape))
    assign = solve_lane_assignment(df_bad)
    return split_teams_balanced_from_assignment(assign)

def pattern_comfort(players: List[Player], score_df: pd.DataFrame):  # ④
    df = _emphasize_prefs_df(players, score_df, weights=(1.25, 1.10, 1.00, 0.90, 0.80))
    assign = solve_lane_assignment(df)
    return split_teams_balanced_from_assignment(assign)

def pattern_lane_mirror(players: List[Player], score_df: pd.DataFrame):  # ⑥
    assign = solve_lane_assignment(score_df)
    return split_teams_minimize_lane_diff(assign)

def pattern_anchor(players: List[Player], score_df: pd.DataFrame):  # ⑦
    assign = solve_lane_assignment(score_df)
    return split_teams_anchor_objective(assign)

def pattern_mentor_boost(players: List[Player], score_df: pd.DataFrame):  # ⑧
    df = score_df.copy()
    levels = pd.Series({p.name: p.level for p in players})
    low_thr = levels.quantile(0.3)
    high_thr = levels.quantile(0.7)
    name2p = _name2player_map(players)
    for name in df.index:
        p = name2p[name]
        prefs = [l for l in p.lane_prefs if l in LANES]
        if not prefs: continue
        first = prefs[0]
        if p.level <= low_thr:
            df.loc[name, first] = min(df.loc[name, first] * 1.20, 1.0)
        elif p.level >= high_thr:
            df.loc[name, first] *= 0.85
    assign = solve_lane_assignment(df.clip(0.0, 1.0))
    return split_teams_balanced_from_assignment(assign)

def pattern_offrole_challenge(players: List[Player], score_df: pd.DataFrame):  # ⑭
    df = _emphasize_prefs_df(players, score_df, weights=(0.80, 0.90, 1.10, 1.08, 1.00))
    assign = solve_lane_assignment(df)
    return split_teams_balanced_from_assignment(assign)

def _sum(team): return sum(x[2] for x in team)

def compute_all_patterns(players: List[Player], score_df: pd.DataFrame) -> dict:
    """
    UI表示用の辞書を返す。
    key: ユーザ向け短いタイトル（絵文字つき）
    val: (A, B, sa, sb, diff)
    """
    A1, B1 = pattern_best(players, score_df)
    A2, B2 = pattern_handicap(players, score_df)
    A3, B3 = pattern_worst(players, score_df)
    A4, B4 = pattern_comfort(players, score_df)
    A6, B6 = pattern_lane_mirror(players, score_df)
    A7, B7 = pattern_anchor(players, score_df)
    A8, B8 = pattern_mentor_boost(players, score_df)
    A14, B14 = pattern_offrole_challenge(players, score_df)

    def pack(A, B):
        sa, sb = _sum(A), _sum(B)
        return (A, B, sa, sb, abs(sa - sb))

    # ★ ここでUI向けの“分かりやすい名前”に変更 ★
    return {
        "全力発揮型":                pack(A1, B1), 
        "ハンデ戦":                  pack(A2, B2), 
        "逆境チャレンジ":            pack(A3, B3), 
        "快適ロール優先":            pack(A4, B4), 
        "ガチ対面勝負":              pack(A6, B6),
        "中核安定型（JG/MID）":      pack(A7, B7),
        "メンターマッチ":            pack(A8, B8),
        "ロールシャッフル":          pack(A14, B14),
    }

def to_table(team: List[Tuple[str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(team, columns=["player", "lane", "score"])
    return df.sort_values("lane").reset_index(drop=True)
