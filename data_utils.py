# data_utils.py — CSV入出力まわり（テンプレDL含む）
import io
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["name", "rank", "mmr", "level", "lane1", "lane2", "lane3", "lane4", "lane5"]

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    # 列名を小文字化（柔軟に）
    df.columns = [c.lower() for c in df.columns]
    return df

def build_template_csv() -> bytes:
    example = pd.DataFrame(
        [
            {"name": "Alice", "rank": "Gold IV", "mmr": 1800, "level": 120,
             "lane1": "TOP", "lane2": "MID", "lane3": "ADC", "lane4": "SUP", "lane5": "JG"},
            {"name": "Bob", "rank": "Platinum II", "mmr": 2100, "level": 230,
             "lane1": "JG", "lane2": "TOP", "lane3": "MID", "lane4": "SUP", "lane5": "ADC"},
        ],
        columns=REQUIRED_COLUMNS,
    )
    buf = io.StringIO()
    example.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
