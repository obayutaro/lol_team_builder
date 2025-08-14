# player_selection.py — 参加プレイヤー選択UI/ロジック
import streamlit as st
import random
from typing import List
import pandas as pd


def _chunk(lst, n):
    """lst を n 個ずつのチャンクに分割"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _clear_checkbox_keys(all_names: List[str]):
    """チェックボックスの Session State キーを一括削除（ランダム/全解除時に使用）"""
    for n in all_names:
        st.session_state.pop(f"pick__{n}", None)


def manage_player_selection(raw_df: pd.DataFrame) -> List[str]:
    # 永続化される唯一の“真実”の選択セット
    if "selected_names" not in st.session_state:
        st.session_state.selected_names = []

    st.subheader("参加プレイヤー選択")
    all_names = raw_df["name"].astype(str).tolist()

    # ── 検索 + 操作用ボタン ───────────────────────────────────────────────
    q_col, btn_col = st.columns([1, 1])
    with q_col:
        query = st.text_input("検索（部分一致）", value="", placeholder="名前で絞り込み")
    with btn_col:
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="btn-row">', unsafe_allow_html=True)
            if st.button("ランダム10", use_container_width=True):
                if len(all_names) >= 10:
                    st.session_state.selected_names = sorted(random.sample(all_names, 10))
                    _clear_checkbox_keys(all_names)  # 前回のチェック状態に引きずられないようクリア
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-row">', unsafe_allow_html=True)
            if st.button("全解除", use_container_width=True):
                st.session_state.selected_names = []
                _clear_checkbox_keys(all_names)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── 表示対象のリスト（検索フィルタ） ──────────────────────────────────
    names_view = [n for n in all_names if query.strip().lower() in n.lower()] if query.strip() else all_names

    # ── ここがポイント：再実行の冒頭で “既存チェックボックスの状態” を反映して選択集合を再構築 ──
    selected = set(st.session_state.selected_names)
    for n in all_names:
        key = f"pick__{n}"
        if key in st.session_state:  # 前回レンダリング済みのチェックボックス
            if st.session_state[key]:
                selected.add(n)
            else:
                selected.discard(n)

    # ===== 横並びグリッド表示 =====
    total = len(names_view)
    if total >= 36:
        cols_n = 6
    elif total >= 24:
        cols_n = 5
    elif total >= 12:
        cols_n = 4
    else:
        cols_n = 3

    st.markdown('<div class="picker-panel">', unsafe_allow_html=True)
    st.markdown("**リストからチェックして10人を選択**")

    # 行ごとに columns() を作り、各列にチェックボックスを置く
    for row_names in _chunk(names_view, cols_n):
        cols = st.columns(cols_n, gap="small")
        for i, n in enumerate(row_names):
            # 10人上限：既に10人選ばれている場合、未選択のチェックボックスは無効化
            disable_new_select = (len(selected) >= 10) and (n not in selected)

            with cols[i]:
                checked = st.checkbox(
                    n,
                    value=(n in selected),       # 常に“いまの選択集合”から決める
                    key=f"pick__{n}",
                    disabled=disable_new_select  # ★ ここが 9人以下になれば自動的に False になる
                )

                # 状態同期
                if checked:
                    selected.add(n)
                else:
                    selected.discard(n)

    st.markdown("</div>", unsafe_allow_html=True)  # picker-panel

    # 下部ステータス
    count = len(selected)
    st.markdown('<div class="btn-row">', unsafe_allow_html=True)
    st.write(f"現在：**{count} / 10**")
    if count >= 10:
        st.caption("※ 10人に達したため、未選択のチェックボックスは自動的に無効化されています。")
    st.markdown("</div>", unsafe_allow_html=True)

    # 最終的な選択を保存
    st.session_state.selected_names = sorted(selected)
    return st.session_state.selected_names
