# main.py — エントリーポイント（余白が出ないよう縦積みに変更）
import streamlit as st
import pandas as pd

from ui_components import (
    inject_style,
    render_title,
    render_weight_sliders,
    render_skill_table,
    render_team_tabs,
    render_sidebar_template_dl_and_uploader,
)
from data_utils import read_uploaded_csv, REQUIRED_COLUMNS
from player_selection import manage_player_selection
from team_building import build_players, make_skill_matrix, compute_all_patterns

# ページ設定
st.set_page_config(page_title="LoL カスタム：メンバー振り分け支援", page_icon="🎮", layout="wide")

# CSS注入
inject_style("style.css")

# タイトル（上側切れ防止スペーサ込み）
render_title()

# ウェイト（ランク>MMR>レーン>Lv 推奨）
weights = render_weight_sliders()

# サイドバー：テンプレDL & アップローダ
uploaded = render_sidebar_template_dl_and_uploader()

if uploaded is None:
    st.info("CSVをアップロードしてください。ヘッダ: name,rank,mmr,level,lane1..lane5")
    st.stop()

# CSV読込 & バリデーション
try:
    raw_df = read_uploaded_csv(uploaded)
except Exception as e:
    st.error(f"CSV読込エラー: {e}")
    st.stop()

if not set(REQUIRED_COLUMNS).issubset(set([c.lower() for c in raw_df.columns])):
    st.error("CSVヘッダが不足しています。必須: name,rank,mmr,level,lane1..lane5")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 余白対策：まず“プレイヤー選択”をページ全幅で描画（右に空白を作らない）
# ─────────────────────────────────────────────────────────────
selected_names = manage_player_selection(raw_df)

# 10人チェック
if len(selected_names) < 10:
    st.warning("10人ちょうどを選んでください。")
    st.stop()
if len(selected_names) > 10:
    st.error("11人以上選択されています。10人に減らしてください。")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 結果セクションは“下に縦積み”で表示（右の巨大な空白は発生しない）
# ─────────────────────────────────────────────────────────────
st.markdown("---")

# 選抜10人に絞る
sel_df = raw_df[raw_df["name"].astype(str).isin(selected_names)].reset_index(drop=True)
players = build_players(sel_df)
score_df = make_skill_matrix(players, (weights["rank"], weights["mmr"], weights["lane"], weights["level"]))

# 推定スキル表（全幅）
render_skill_table(score_df)

# 8パターン計算（各レーン2名を厳密保証）→ タブで全幅表示
patterns = compute_all_patterns(players, score_df)
render_team_tabs(patterns)
