# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="보이스피싱 탐지 포트폴리오 | 개요 대시보드", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data
def load_csv(path_or_buffer) -> pd.DataFrame:
    """
    CSV 인코딩 자동 감지(utf-8 계열 우선, 안되면 cp949/euc-kr).
    path_or_buffer: 파일경로(str) 또는 업로드된 파일 객체.
    """
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path_or_buffer, encoding=enc)
        except Exception:
            continue
    # 마지막 fallback
    return pd.read_csv(path_or_buffer)


def load_or_upload(filename: str, key: str) -> pd.DataFrame:
    """현재 폴더에 있으면 로드, 없으면 업로더로 받음."""
    if os.path.exists(filename):
        return load_csv(filename)
    up = st.sidebar.file_uploader(f"{filename} 업로드", type=["csv"], key=key)
    if up is None:
        st.sidebar.error(f"'{filename}'가 현재 폴더에 없고 업로드도 되지 않았습니다.")
        st.stop()
    return pd.read_csv(up)

def fmt_int(x):
    return "-" if pd.isna(x) else f"{int(round(float(x))):,}"

def fmt_float(x, d=1):
    return "-" if pd.isna(x) else f"{float(x):,.{d}f}"

def fmt_cases_k(x):
    """발생건수: 237,334건"""
    if pd.isna(x):
        return "-"
    return f"{int(round(float(x))):,}건"

def fmt_eok_to_jo_eok(x):
    """
    피해액이 '억원' 단위로 들어올 때: 47,575.0 -> 4조7575억
    (반올림: 억 단위 정수 기준)
    """
    if pd.isna(x):
        return "-"
    eok = int(round(float(x)))  # 억 단위 정수화
    jo = eok // 10000
    rem = eok % 10000
    if jo > 0:
        return f"{jo}조{rem:04d}억"  # 4조7575억처럼 4자리 유지
    return f"{rem:,}억"

def fmt_man_unit(x, d=1):
    """만원 단위: 2004.6만 / 4100.5만"""
    if pd.isna(x):
        return "-"
    return f"{float(x):.{d}f}만"


def ensure_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -----------------------------
# Sidebar
# -----------------------------
st.title("보이스피싱 탐지 딥러닝 포트폴리오 | 서론용 대시보드")
st.caption("구성: 현황 → 심각성 → 유형별 비교 → 단속/검거 → 기대효과 (데이터 기반 동기 제시)")

st.sidebar.header("데이터 로드")
annual = load_or_upload("annual.csv", "u_annual")
dim_date_month = load_or_upload("dim_date_month.csv", "u_dim")
monthly = load_or_upload("monthly_cases.csv", "u_monthly")
types_only = load_or_upload("types_only.csv", "u_types")
age = load_or_upload("age.csv", "u_age")


st.sidebar.divider()
loss_unit = st.sidebar.selectbox(
    "피해액 단위(표기)",
    ["억원(Loss_Eok)", "만원(Loss_Man)"],
    index=0,
    key="sb_loss_unit"
)
loss_col = "Loss_Eok" if loss_unit.startswith("억원") else "Loss_Man"
loss_label = "피해액(억원)" if loss_col == "Loss_Eok" else "피해액(만원)"

show_yoy = st.sidebar.checkbox("YoY 지표(증감) 표시", value=True, key="sb_show_yoy")
show_indices = st.sidebar.checkbox("Index 지표 표시", value=False, key="sb_show_indices")

# -----------------------------
# Preprocess
# -----------------------------
annual = annual.copy()
monthly = monthly.copy()
types_only = types_only.copy()
dim_date_month = dim_date_month.copy()

ensure_datetime(monthly, "Date")
ensure_datetime(dim_date_month, "Date")

required_annual = {"Year", "Type", "Cases", "Loss_Eok", "Loss_Man", "Loss_Per_Case_Man"}
required_monthly = {"Date", "Year", "Month", "Cases_Monthly"}
required_types = {"Year", "Type", "Cases", "Loss_Eok", "Loss_Man", "Loss_Per_Case_Man"}
required_age = {"구분", "20대이하", "30대", "40대", "50대", "60대", "70대이상"}

missing_a = sorted(list(required_annual - set(annual.columns)))
missing_m = sorted(list(required_monthly - set(monthly.columns)))
missing_t = sorted(list(required_types - set(types_only.columns)))
missing_age = sorted(list(required_age - set(age.columns)))


if missing_a:
    st.error(f"annual.csv에 필요한 컬럼이 없습니다: {missing_a}")
    st.stop()
if missing_m:
    st.error(f"monthly_cases.csv에 필요한 컬럼이 없습니다: {missing_m}")
    st.stop()
if missing_t:
    st.error(f"types_only.csv에 필요한 컬럼이 없습니다: {missing_t}")
    st.stop()
if missing_age:
    st.error(f"age.csv에 필요한 컬럼이 없습니다: {missing_age}")
    st.stop()

years = sorted(pd.unique(annual["Year"].dropna()).tolist())
ymin, ymax = int(min(years)), int(max(years))

year_range = st.sidebar.slider("연도 범위", ymin, ymax, (ymin, ymax), key="sb_year_range")
selected_year = st.sidebar.selectbox(
    "유형 비교 기준 연도",
    list(range(ymin, ymax + 1)),
    index=(ymax - ymin),
    key="sb_selected_year"
)

annual_f = annual[(annual["Year"] >= year_range[0]) & (annual["Year"] <= year_range[1])].copy()
monthly_f = monthly[(monthly["Date"].dt.year >= year_range[0]) & (monthly["Date"].dt.year <= year_range[1])].copy()
types_f = types_only[(types_only["Year"] >= year_range[0]) & (types_only["Year"] <= year_range[1])].copy()

annual_total = annual_f[annual_f["Type"] == "전체"].sort_values("Year")
annual_types = annual_f[annual_f["Type"] != "전체"].sort_values(["Type", "Year"])

# -----------------------------
# KPI (현황 스냅샷)
# -----------------------------
if not annual_total.empty:
    total_cases = annual_total["Cases"].sum()
    total_loss = annual_total[loss_col].sum()
    avg_lpc_man = (annual_total["Loss_Man"].sum() / total_cases) if total_cases else np.nan
    latest_row = annual_total[annual_total["Year"] == year_range[1]].head(1)
else:
    total_cases = annual_types["Cases"].sum()
    total_loss = annual_types[loss_col].sum()
    avg_lpc_man = (annual_types["Loss_Man"].sum() / total_cases) if total_cases else np.nan
    latest_row = pd.DataFrame()

k1, k2, k3, k4 = st.columns(4)
k1.metric("기간 합계 발생건수", fmt_cases_k(total_cases))

# 피해액은 '억원' 선택일 때만 4조7575억 포맷 적용
if loss_col == "Loss_Eok":
    k2.metric("기간 합계 피해액", fmt_eok_to_jo_eok(total_loss))
else:
    # 만원일 때는 단위 그대로 표기(예: 47,575,000만원 형태로 원하면 여기서 별도 포맷)
    k2.metric("기간 합계 피해액", f"{fmt_float(total_loss, 0)}만원")

k3.metric("기간 평균 건당피해액(만원/건)", fmt_man_unit(avg_lpc_man, 1))

if not latest_row.empty:
    k4.metric(f"{year_range[1]} 건당피해액(만원/건)", fmt_man_unit(latest_row["Loss_Per_Case_Man"].iloc[0], 1))
else:
    k4.metric(f"{year_range[1]} 건당피해액(만원/건)", "-")


st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab_age, tab3, tab4, tab5 = st.tabs(
    ["1) 현황", "2) 심각성", "3) 연령대 피해자", "4) 유형별 비교", "5) 단속/검거", "6) 기대효과"]
)


# =========================================================
# 1) 현황
# =========================================================
with tab1:
    st.subheader("현황: 연도별·월별로 관측되는 반복적 금융범죄(트렌드/패턴)")

    cA, cB = st.columns([2, 1])

    with cA:
        if not annual_total.empty:
            fig = px.line(annual_total, x="Year", y="Cases", markers=True, title="연도별 발생건수(총합)")
        else:
            fig = px.line(annual_types, x="Year", y="Cases", color="Type", markers=True, title="연도별 발생건수(유형)")
        fig.update_layout(xaxis_title="연도", yaxis_title="건")
        st.plotly_chart(fig, use_container_width=True, key="tab1_cases_year")

        if not annual_total.empty:
            fig2 = px.line(annual_total, x="Year", y=loss_col, markers=True, title=f"연도별 {loss_label}(총합)")
        else:
            fig2 = px.line(annual_types, x="Year", y=loss_col, color="Type", markers=True, title=f"연도별 {loss_label}(유형)")
        fig2.update_layout(xaxis_title="연도", yaxis_title=loss_label)
        st.plotly_chart(fig2, use_container_width=True, key="tab1_loss_year")

    with cB:
        m = monthly_f.sort_values("Date")
        figm = px.line(m, x="Date", y="Cases_Monthly", title="월별 발생건수(전체)")
        figm.update_layout(xaxis_title="월", yaxis_title="건")
        st.plotly_chart(figm, use_container_width=True, key="tab1_cases_month")

        if "Is_Peak_Top10pct" in m.columns:
            peaks = m[m["Is_Peak_Top10pct"] == True][["Year", "Month", "YearMonth", "Cases_Monthly"]].copy()
            if not peaks.empty:
                st.markdown("**피크 월(Top 10%)**")
                st.dataframe(peaks.sort_values(["Year", "Month"]), use_container_width=True, height=250)

    st.info(
        "서론 연결 문장 예시: "
        "연도별·월별 지표에서 변동성과 피크가 반복적으로 관측되며, 이는 보이스피싱이 데이터 관점에서 탐지 가능한 구조(패턴/이상치)를 가진다는 근거가 된다."
    )

# =========================================================
# 2) 심각성
# =========================================================
with tab2:
    st.subheader("심각성: 피해액과 건당 피해액(대형 피해화)")

    if not annual_total.empty:
        df2 = annual_total.sort_values("Year")
        fig = px.bar(df2, x="Year", y="Loss_Per_Case_Man", title="연도별 건당피해액(만원/건) - 총합")
    else:
        df2 = annual_types.sort_values(["Type", "Year"])
        fig = px.line(df2, x="Year", y="Loss_Per_Case_Man", color="Type", markers=True, title="연도별 건당피해액(만원/건) - 유형")
    fig.update_layout(xaxis_title="연도", yaxis_title="만원/건")
    st.plotly_chart(fig, use_container_width=True, key="tab2_lpc")

    if not annual_total.empty and not df2.empty:
        idx = df2["Loss_Per_Case_Man"].idxmax()
        worst_year = int(df2.loc[idx, "Year"])
        worst_lpc = float(df2.loc[idx, "Loss_Per_Case_Man"])
        st.markdown(f"- **건당피해액 최대 연도:** {worst_year}년 (약 {worst_lpc:,.1f} 만원/건)")

    st.info(
        "서론 연결 문장 예시: "
        "동일한 발생건수라도 건당 피해액이 상승하는 구간에서는 사회적 피해가 비선형적으로 확대되므로, 사후 구제 중심 대응만으로는 한계가 있다."
    )


# =========================================================
# 3) 연령대 피해자
# =========================================================
with tab_age:
    st.subheader("연령대 피해자: 연간별 추세")
    
    age_df = age.copy().rename(columns={"구분": "Year"})
    age_df["Year"] = pd.to_numeric(age_df["Year"], errors="coerce")

    age_cols = [c for c in age_df.columns if c != "Year"]
    for c in age_cols:
        age_df[c] = pd.to_numeric(age_df[c], errors="coerce")

    age_df = age_df.dropna(subset=["Year"]).copy()
    age_df["Year"] = age_df["Year"].astype(int)
    age_df = age_df[(age_df["Year"] >= year_range[0]) & (age_df["Year"] <= year_range[1])].copy()
    age_df = age_df.sort_values("Year")

    age_long = age_df.melt(id_vars="Year", value_vars=age_cols, var_name="AgeGroup", value_name="Victims")
    age_long = age_long.dropna(subset=["Victims"]).copy()

    fig_age_line = px.line(
        age_long.sort_values(["AgeGroup", "Year"]),
        x="Year", y="Victims", color="AgeGroup",
        markers=True,
        title="연간별 연령대 피해자수 추세"
    )
    fig_age_line.update_layout(xaxis_title="연도", yaxis_title="피해자 수(명)", height=520, legend_title_text="연령대")
    st.plotly_chart(fig_age_line, use_container_width=True, key="tab_age_line_trend")

# =========================================================
# 3) 유형별 비교
# =========================================================
with tab3:
    st.subheader("유형별 비교")
    st.caption("상단은 연도별 유형 비교(선그래프), 하단은 선택 연도 기준 도넛 3개로 요약합니다.")

    df3 = types_f.sort_values(["Type", "Year"])

    top1, top2 = st.columns(2)
    with top1:
        fig = px.line(df3, x="Year", y="Cases", color="Type", markers=True, title="유형별 발생건수(연도별)")
        fig.update_layout(xaxis_title="연도", yaxis_title="건")
        st.plotly_chart(fig, use_container_width=True, key="tab3_cases_line")

    with top2:
        fig2 = px.line(df3, x="Year", y=loss_col, color="Type", markers=True, title=f"유형별 {loss_label}(연도별)")
        fig2.update_layout(xaxis_title="연도", yaxis_title=loss_label)
        st.plotly_chart(fig2, use_container_width=True, key="tab3_loss_line")

    st.divider()

    st.markdown(f"##### {selected_year}년 유형 분포(도넛)")

    dfy = types_only[types_only["Year"] == selected_year].copy()
    if dfy.empty:
        st.warning("types_only.csv에서 선택 연도의 데이터가 없습니다.")
        st.stop()

    dfy = dfy.sort_values("Cases", ascending=False)

    donut_hole = 0.62
    loss_unit_text = "억원" if loss_col == "Loss_Eok" else "만원"

    legend_layout = dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="center",
        x=0.5
    )
    common_layout = dict(
        showlegend=True,
        legend=legend_layout,
        margin=dict(l=10, r=10, t=70, b=110),
        height=460
    )
    common_traces = dict(
        textposition="outside",
        automargin=True,
        textfont_size=12
    )

    fig_cases = px.pie(dfy, names="Type", values="Cases", hole=donut_hole, title="유형별 발생건수")
    fig_cases.update_traces(
        **common_traces,
        texttemplate="%{label}<br>%{value:,}건<br>(%{percent})",
        hovertemplate="<b>%{label}</b><br>발생건수: %{value:,}건<br>비율: %{percent}<extra></extra>"
    )
    fig_cases.update_layout(**common_layout)
    fig_cases.add_annotation(text="발생 건수", x=0.5, y=0.5, font=dict(size=16), showarrow=False)

    dfy["_loss_round"] = dfy[loss_col].round(0)
    fig_loss = px.pie(dfy, names="Type", values="_loss_round", hole=donut_hole, title=f"유형별 피해액({loss_unit_text})")
    fig_loss.update_traces(
        **common_traces,
        texttemplate=f"%{{label}}<br>%{{value:,.0f}}{loss_unit_text}<br>(%{{percent}})",
        hovertemplate=f"<b>%{{label}}</b><br>피해액: %{{value:,.0f}}{loss_unit_text}<br>비율: %{{percent}}<extra></extra>"
    )
    fig_loss.update_layout(**common_layout)
    fig_loss.add_annotation(text="피해 금액", x=0.5, y=0.5, font=dict(size=16), showarrow=False)

    dfy["_lpc_round"] = dfy["Loss_Per_Case_Man"].round(0)
    fig_lpc = px.pie(dfy, names="Type", values="_lpc_round", hole=donut_hole, title="유형별 건당피해액(만원/건)")
    fig_lpc.update_traces(
        **common_traces,
        texttemplate="%{label}<br>%{value:,.0f}만원/건<br>(%{percent})",
        hovertemplate="<b>%{label}</b><br>건당피해액: %{value:,.0f}만원/건<br>비율: %{percent}<extra></extra>"
    )
    fig_lpc.update_layout(**common_layout)
    fig_lpc.add_annotation(text="건당 피해액", x=0.5, y=0.5, font=dict(size=16), showarrow=False)

    d1, d2, d3 = st.columns(3, vertical_alignment="top")
    with d1:
        st.plotly_chart(fig_cases, use_container_width=True, key="tab3_donut_cases")
    with d2:
        st.plotly_chart(fig_loss, use_container_width=True, key="tab3_donut_loss")
    with d3:
        st.plotly_chart(fig_lpc, use_container_width=True, key="tab3_donut_lpc")

    with st.expander("선택 연도 유형별 수치(검증용)"):
        st.dataframe(
            dfy[["Type", "Cases", loss_col, "Loss_Per_Case_Man"]].rename(
                columns={loss_col: loss_label, "Loss_Per_Case_Man": "건당피해액(만원/건)"}
            ).reset_index(drop=True),
            use_container_width=True
        )

    st.info(
        "서론 연결 문장 예시: "
        "유형별로 확산 규모(발생건수)와 피해 강도(건당 피해액)가 다르게 나타나므로, 탐지 모델은 단일 기준보다 유형/위험도 관점의 설계가 타당하다."
    )

# =========================================================
# 4) 단속/검거
# =========================================================
with tab4:
    st.subheader("단속/검거: 효율 지표(커버리지 & 검거 1건당 피해액)")
    st.caption("‘정확한 검거율’이 아니라, 발생-검거의 괴리를 보여주는 커버리지 지표로 해석합니다.")

    need_cols = {"Year", "Cases", loss_col, "Arrest_Cases"}
    if not need_cols.issubset(set(annual_f.columns)):
        st.warning("annual.csv에 Arrest_Cases(검거건수) 컬럼이 없거나, 필수 컬럼이 부족하여 시각화를 생략합니다.")
        st.stop()

    if (not annual_total.empty) and {"Arrest_Cases"}.issubset(set(annual_total.columns)):
        df4 = annual_total[["Year", "Cases", loss_col, "Arrest_Cases"]].sort_values("Year").copy()
    else:
        df4 = (
            annual_types.groupby("Year", as_index=False)[["Cases", loss_col, "Arrest_Cases"]].sum()
            .sort_values("Year")
            .copy()
        )

    df4["Coverage"] = df4["Arrest_Cases"] / df4["Cases"]
    df4["Coverage_pct"] = df4["Coverage"] * 100

    df4["Loss_per_ArrestCase"] = np.where(
        df4["Arrest_Cases"] > 0,
        df4[loss_col] / df4["Arrest_Cases"],
        np.nan
    )

    loss_unit_text = "억원" if loss_col == "Loss_Eok" else "만원"
    loss_per_arrest_label = f"검거 1건당 피해액({loss_unit_text}/건)"

    latest = df4.iloc[-1]
    prev = df4.iloc[-2] if len(df4) >= 2 else None

    def delta(curr, prevv):
        if prev is None or pd.isna(prevv) or pd.isna(curr):
            return None
        return curr - prevv

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "발생 대비 검거 커버리지(%)",
        "-" if pd.isna(latest["Coverage_pct"]) else f"{latest['Coverage_pct']:.1f}%",
        None if prev is None else f"{delta(latest['Coverage_pct'], prev['Coverage_pct']):+.1f}p"
    )
    k2.metric("발생건수", f"{int(latest['Cases']):,}" if pd.notna(latest["Cases"]) else "-")
    k3.metric(
        "검거건수",
        f"{int(latest['Arrest_Cases']):,}" if pd.notna(latest["Arrest_Cases"]) else "-",
        None if prev is None else f"{int(delta(latest['Arrest_Cases'], prev['Arrest_Cases'])):+,}"
    )
    k4.metric(
        loss_per_arrest_label,
        "-" if pd.isna(latest["Loss_per_ArrestCase"]) else f"{latest['Loss_per_ArrestCase']:.0f}{loss_unit_text}/건",
        None if prev is None else f"{delta(latest['Loss_per_ArrestCase'], prev['Loss_per_ArrestCase']):+.0f}{loss_unit_text}/건"
    )

    st.divider()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df4["Year"], y=df4["Cases"], name="발생건수", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df4["Year"], y=df4["Arrest_Cases"], name="검거건수", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df4["Year"], y=df4["Coverage_pct"], name="검거 커버리지(%)", mode="lines+markers"), secondary_y=True)

    fig.update_layout(
        title="발생 vs 검거(절대값) + 발생 대비 검거 커버리지(%)",
        xaxis_title="연도",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
        margin=dict(l=30, r=30, t=60, b=90),
        height=480
    )
    fig.update_yaxes(title_text="건(발생/검거)", secondary_y=False)
    fig.update_yaxes(title_text="커버리지(%)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True, key="tab4_cover")

    fig2 = px.line(df4, x="Year", y="Loss_per_ArrestCase", markers=True, title=f"{loss_per_arrest_label} 추이")
    fig2.update_layout(xaxis_title="연도", yaxis_title=f"{loss_unit_text}/건", height=450)
    st.plotly_chart(fig2, use_container_width=True, key="tab4_loss_per_arrest")

    st.info(
        "해석 포인트: "
        "① 커버리지는 ‘정확한 검거율’이 아니라 발생 대비 검거의 괴리/개선 정도를 보여주는 지표입니다. "
        "② ‘검거 1건당 피해액’은 큰 피해 사건의 검거가 동행되는지(또는 뒤따르는지)를 보조적으로 해석할 수 있습니다."
    )

    with st.expander("연도별 단속/검거 효율 지표(표)"):
        show_cols = ["Year", "Cases", "Arrest_Cases", loss_col, "Coverage_pct", "Loss_per_ArrestCase"]
        out = df4[show_cols].copy().rename(columns={
            loss_col: loss_label,
            "Coverage_pct": "Coverage(%)",
            "Loss_per_ArrestCase": loss_per_arrest_label
        })
        st.dataframe(out, use_container_width=True)

# =========================================================
# 5) 기대효과
# =========================================================
with tab5:
    st.subheader("기대효과: 딥러닝 탐지의 정량적 가치(피해 예방/우선순위/운영 효율)")

    st.markdown(
        """
- **피해 예방(Prevention):** 통화/메시지 단계에서 위험 신호를 조기 탐지하여 사후 구제 이전에 차단  
- **우선순위 기반 대응(Prioritization):** 고피해형 유형(건당 피해액↑)에 경보/차단을 강화  
- **운영 효율(Efficiency):** 상담/모니터링 인력의 부담을 줄이고 고위험 케이스 우선 분류  
- **확장성(Scalability):** 금융/통신/앱 서비스에 적용 가능한 탐지 파이프라인 제시
        """
    )

    st.markdown("##### 간단한 정량 시뮬레이션(가정): 탐지가 피해를 얼마나 줄일 수 있는가?")
    colA, colB = st.columns([1, 2])

    with colA:
        reduction = st.slider("발생건수 감소율(가정)", 0, 50, 10, step=1, key="tab5_reduction")
        target_year = st.selectbox(
            "기준 연도(총합)",
            list(range(year_range[0], year_range[1] + 1)),
            index=(year_range[1] - year_range[0]),
            key="tab5_target_year"
        )

        if not annual_total.empty:
            base = annual_total[annual_total["Year"] == target_year]
        else:
            base = annual_types.groupby("Year", as_index=False)[["Cases", "Loss_Man", "Loss_Eok"]].sum()
            base = base[base["Year"] == target_year]

        if base.empty:
            st.warning("선택한 연도의 기준 데이터가 없습니다.")
            st.stop()

        base_cases = float(base["Cases"].iloc[0])
        base_loss = float(base[loss_col].iloc[0])
        prevented_cases = base_cases * (reduction / 100.0)
        prevented_loss = base_loss * (reduction / 100.0)

        st.metric("예방 가능(가정) 발생건수", fmt_int(prevented_cases))
        st.metric(f"예방 가능(가정) {loss_label}", fmt_float(prevented_loss, 1))

    with colB:
        sim_df = pd.DataFrame({
            "항목": ["기준 발생건수", f"기준 {loss_label}", "감소율(가정)", "예방 발생건수(가정)", f"예방 {loss_label}(가정)"],
            "값": [fmt_int(base_cases), fmt_float(base_loss, 1), f"{reduction}%", fmt_int(prevented_cases), fmt_float(prevented_loss, 1)]
        })
        st.dataframe(sim_df, use_container_width=True, height=240)

    if show_indices and (not annual_total.empty):
        idx_cols = [c for c in ["Cases_Index", "Loss_Eok_Index", "Loss_Per_Case_Index"] if c in annual_total.columns]
        if idx_cols:
            st.markdown("##### Index(총합, 기준연도 대비 상대 변화)")
            st.dataframe(annual_total[["Year"] + idx_cols].sort_values("Year"), use_container_width=True)

st.divider()

# -----------------------------
# Appendix: Raw tables
# -----------------------------
with st.expander("원본 데이터 테이블 보기(검증용)"):
    st.markdown("**annual.csv**")
    st.dataframe(annual.sort_values(["Year", "Type"]), use_container_width=True)
    st.markdown("**types_only.csv**")
    st.dataframe(types_only.sort_values(["Year", "Type"]), use_container_width=True)
    st.markdown("**monthly_cases.csv**")
    st.dataframe(monthly.sort_values("Date"), use_container_width=True)
    st.markdown("**dim_date_month.csv**")
    st.dataframe(dim_date_month.sort_values("Date"), use_container_width=True)
