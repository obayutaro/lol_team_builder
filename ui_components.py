# ui_components.py — UI描画まわり（ユーザ向けパターン名＋説明表示）
import streamlit as st
import pandas as pd
from pathlib import Path

from data_utils import build_template_csv
from team_building import to_table

def inject_style(css_path: str):
    """style.css を読み込んで注入"""
    try:
        css = Path(css_path).read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"スタイル読み込みに失敗しました: {e}")

def render_title():
    # タイトル上切れ回避のための軽いスペーサ
    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    st.markdown("<h1>LoLカスタム：メンバー振り分け支援</h1>", unsafe_allow_html=True)

def render_weight_sliders():
    st.sidebar.subheader("重み設定")
    w_rank = st.sidebar.slider("実ランクの重み", 0.0, 1.0, 0.40, 0.01)
    w_mmr  = st.sidebar.slider("MMRの重み",     0.0, 1.0, 0.30, 0.01)
    w_lane = st.sidebar.slider("得意レーンの重み", 0.0, 1.0, 0.20, 0.01)
    w_level= st.sidebar.slider("プレイヤーレベルの重み", 0.0, 1.0, 0.10, 0.01)
    total_w = w_rank + w_mmr + w_lane + w_level
    if abs(total_w - 1.0) > 1e-6:
        st.sidebar.caption(f"現在の合計: {total_w:.2f}（1.00付近を推奨）")
    return {"rank": w_rank, "mmr": w_mmr, "lane": w_lane, "level": w_level}

def render_sidebar_template_dl_and_uploader():
    st.sidebar.markdown("---")
    st.sidebar.caption("CSVテンプレートをダウンロード")
    st.sidebar.download_button(
        label="CSVテンプレートをDL",
        data=build_template_csv(),
        file_name="players_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    return st.sidebar.file_uploader("プレイヤーCSVをアップロード", type=["csv"])

def render_skill_table(score_df: pd.DataFrame):
    st.subheader("推定スキル（レーン別）")
    st.dataframe(score_df.style.format("{:.3f}"), use_container_width=True)

# ★ パターンの“説明文”をUIで補足表示するための辞書
PATTERN_DESCRIPTIONS = {
    "全力発揮型": "得意ロールを最大限活かして、総合力が最大になるように編成します（その後、A/Bの総合差を最小化）。",
    "ハンデ戦": "上位プレイヤーの第1/第2得意ロールを抑え、バランスを整える構成です。",
    "逆境チャレンジ": "苦手ロール寄りの割当で、あえて難易度を上げるお遊び構成です。",
    "快適ロール優先": "各自の得意ロールに強く寄せて、快適にプレイできる編成にします。",
    "ガチ対面勝負": "各レーンのA/B対面差を最小化し、1対1が拮抗するように調整します。",
    "中核安定型（JG/MID）": "両チームともJG/MIDが強くなるように分割し、ゲームの主導権を取りやすくします。",
    "メンターマッチ": "上級者と初心者をマッチングして、経験を共有しやすい構成にします。",
    "ロールシャッフル": "普段やらないロールに挑戦しやすいよう、得意順位を反転気味にして割当します。",
}

def render_team_tabs(patterns: dict):
    """
    patterns: {tab_title(=ユーザ向け名): (A, B, sa, sb, diff)}
    タブ見出しは短い“わかりやすい名前”、タブ内の先頭に説明文を表示する。
    """
    st.subheader("チーム提案")
    labels = list(patterns.keys())
    tabs = st.tabs(labels)

    def _render_tab(A, B, label, sa, sb, d):
        # パターン説明（あれば表示）
        desc = PATTERN_DESCRIPTIONS.get(label)
        if desc:
            st.caption(desc)

        # メトリクス行
        st.markdown(
            f'<div class="metric-line">{label} ｜ Team A: <strong>{sa:.3f}</strong> ／ '
            f'Team B: <strong>{sb:.3f}</strong> ／ 差: <strong>{d:.3f}</strong></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Team A")
            st.table(to_table(A))
        with c2:
            st.markdown("### Team B")
            st.table(to_table(B))

    for i, t in enumerate(labels):
        A, B, sa, sb, d = patterns[t]
        with tabs[i]:
            _render_tab(A, B, t, sa, sb, d)
