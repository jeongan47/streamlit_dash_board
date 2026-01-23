import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
# ✅ [추가] 오디오 분석용 라이브러리
import librosa
import graphviz
import matplotlib.pyplot as plt
import altair as alt
import os, glob

st.set_page_config(page_title="보이스피싱 탐지 대시보드", layout="wide")

# -----------------------------
# 1. 유틸리티 및 데이터 로드
# -----------------------------
@st.cache_data
def load_csv(path_or_buffer) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path_or_buffer, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path_or_buffer)

def load_or_upload(filename: str, key: str) -> pd.DataFrame:
    if os.path.exists(filename):
        return load_csv(filename)
    up = st.sidebar.file_uploader(f"{filename} 업로드", type=["csv"], key=key)
    if up is None:
        return pd.DataFrame() 
    return pd.read_csv(up)

def fmt_man_unit(x):
    """천단위 콤마가 포함된 만원 단위 문자열 반환"""
    if pd.isna(x) or x == 0 or np.isinf(x): return "-"
    return f"{float(x):,.0f}만"

def fmt_jo_eok(x):
    """조/억 단위 변환"""
    if pd.isna(x): return "-"
    val = int(round(float(x)))
    jo, rem = divmod(val, 10000)
    if jo > 0: return f"{jo}조 {rem:,}억"
    return f"{rem:,}억"

def load_crime_data(filename, key_suffix):
    df = load_or_upload(filename, f"u_{key_suffix}")
    if not df.empty and '연도' in df.columns:
        # 연도를 인덱스로 설정하고 숫자형으로 변환
        df = df.set_index('연도')
        # 혹시 모를 쉼표(,) 제거 및 숫자 변환
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
    return df

# -----------------------------
# 2. 데이터 로드
# -----------------------------
st.title("보이스피싱 탐지 대시보드")
# 구분선 코드(st.markdown("---")) 삭제됨

st.sidebar.header("데이터 로드")
annual = load_or_upload("annual.csv", "u_annual")
monthly = load_or_upload("monthly_cases.csv", "u_monthly")
types_only = load_or_upload("types_only.csv", "u_types")
age = load_or_upload("age2.csv", "u_age")
data_2025 = load_or_upload("25년(11월).csv", "u_25") 
leak_df = load_or_upload("역대유출사고.csv", "u_leak")
df_occur = load_crime_data("발생건수.csv", "occur")
df_arrest = load_crime_data("검거건수.csv", "arrest")
df_rate = load_crime_data("검거율.csv", "rate")

# [신규] 비정상 흐름 분석 데이터 로드
st.sidebar.markdown("---")
st.sidebar.header("흐름 분석 데이터")
# 1. 주요 키워드 데이터
df_keywords = load_or_upload("주요 키워드.csv", "u_keyword_csv")
# 2. 단계 시퀀스(흐름) 데이터
df_sequences = load_or_upload("단계_시퀀스.csv", "u_sequence_csv")
# 3. 단계 전이(Transition) 데이터
df_transitions = load_or_upload("단계_전이.csv", "u_transition_csv")

# ... (이하 날짜 전처리 및 데이터 처리 코드는 기존과 동일하게 유지) ...

# monthly 데이터 날짜 전처리 (Tab 3에서 사용)
if not monthly.empty:
    if "Date" in monthly.columns:
        monthly["Date"] = pd.to_datetime(monthly["Date"])
    elif "Year" in monthly.columns and "Month" in monthly.columns:
        monthly["Date"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")

target_types = ["기관사칭형", "대출사기형"]
color_map = {"기관사칭형": "#003366", "대출사기형": "#87CEEB"}
type_colors = ["#003366", "#87CEEB"] 

# -----------------------------
# 3. 데이터 전처리 (2025년 강제 생성 로직)
# -----------------------------

# (1) 2025년 총계 파악
total_cases_2025 = 0
total_loss_2025 = 0

# A. 발생건수
if not data_2025.empty and "전화금융사기 발생건수" in data_2025.columns:
    total_cases_2025 = data_2025["전화금융사기 발생건수"].sum()

# B. 피해액
if not annual.empty and "Year" in annual.columns:
    row_25_all = annual[(annual["Year"] == 2025) & (annual["Type"] == "전체")]
    if not row_25_all.empty:
        total_loss_2025 = row_25_all["Loss_Eok"].values[0]
        if total_cases_2025 == 0:
            total_cases_2025 = row_25_all["Cases"].values[0]

# [중요] 2025년 피해액 누락 시 2024년 기준 추정
if total_loss_2025 == 0 and total_cases_2025 > 0:
    row_24 = annual[(annual["Year"] == 2024) & (annual["Type"] == "전체")]
    if not row_24.empty:
        lpc_24 = row_24["Loss_Per_Case_Man"].values[0]
        total_loss_2025 = (total_cases_2025 * lpc_24) / 10000
    else:
        total_loss_2025 = 2000 

# (2) 데이터 통합
types_final = types_only.copy()
annual_final = annual.copy()

if total_cases_2025 > 0:
    types_final = types_final[types_final["Year"] != 2025]
    annual_final = annual_final[(annual_final["Year"] != 2025) | (annual_final["Type"] != "전체")]

    loss_imp = total_loss_2025 * 0.75
    loss_loan = total_loss_2025 * 0.25
    cases_imp = total_cases_2025 * 0.4
    cases_loan = total_cases_2025 * 0.6
    
    new_rows = [
        {"Year": 2025, "Type": "기관사칭형", "Cases": cases_imp, "Loss_Eok": loss_imp},
        {"Year": 2025, "Type": "대출사기형", "Cases": cases_loan, "Loss_Eok": loss_loan}
    ]
    types_final = pd.concat([types_final, pd.DataFrame(new_rows)], ignore_index=True)
    
    new_total = pd.DataFrame([{
        "Year": 2025, "Type": "전체", 
        "Cases": total_cases_2025, "Loss_Eok": total_loss_2025
    }])
    annual_final = pd.concat([annual_final, new_total], ignore_index=True)

# (3) 건당 피해액 재계산
def calc_lpc(row):
    if pd.notna(row["Cases"]) and row["Cases"] > 0:
        return (row["Loss_Eok"] * 10000) / row["Cases"]
    return 0

types_final["Loss_Per_Case_Man"] = types_final.apply(calc_lpc, axis=1)
annual_final["Loss_Per_Case_Man"] = annual_final.apply(calc_lpc, axis=1)

# (4) 필터링
years = sorted(list(set(annual_final["Year"].dropna().astype(int))))
if not years: years = [2016, 2025]
min_y, max_y = min(years), max(years)
year_range = st.sidebar.slider("연도 범위 선택", min_y, max_y, (min_y, max_y))

annual_f = annual_final[(annual_final["Year"] >= year_range[0]) & (annual_final["Year"] <= year_range[1])].copy()
types_f = types_final[(types_final["Year"] >= year_range[0]) & (types_final["Year"] <= year_range[1])].copy()
types_f_main = types_f[types_f["Type"].isin(target_types)].copy()

# 4. 탭 구성 (Tab Definition)
# -----------------------------
# KPI 섹션을 탭 정의 아래로 이동시켰으므로, 탭을 먼저 생성합니다.
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 현황 & 심각성", "2️⃣ 연령별 피해", "3️⃣ 대응의 한계 & 가치", "4️⃣ 딥보이스 분석"])


# ==============================================================================
# TAB 1: 현황 & 심각성
# ==============================================================================
with tab1:

    # ------------------------------------------------------------------
    # [데이터 전처리]
    # ------------------------------------------------------------------
    target_types = ["기관사칭형", "대출사기형"]
    df_types = annual[annual['Type'].isin(target_types)].copy()
    
    # 전체 합계 데이터
    if "전체" in annual['Type'].values:
        df_total = annual[annual['Type'] == "전체"].copy()
    else:
        df_total = annual.groupby('Year', as_index=False)[['Cases', 'Loss_Eok']].sum()
        df_total['Loss_Per_Case_Man'] = df_total.apply(
            lambda x: (x['Loss_Eok'] * 10000) / x['Cases'] if x['Cases'] > 0 else 0, axis=1
        )
        df_total['Type'] = '전체'

    type_colors = ["#003366", "#87CEEB"]
    color_map = {"기관사칭형": "#003366", "대출사기형": "#87CEEB"}

    # ------------------------------------------------------------------
    # [KPI 섹션]
    # ------------------------------------------------------------------
    st.subheader(f"현황 KPI (2016 ~ 2025 전체)")

    full_period = df_total[(df_total["Year"] >= 2016) & (df_total["Year"] <= 2025)]
    total_cases_all = full_period["Cases"].sum()
    total_loss_all = full_period["Loss_Eok"].sum()

    row_2025 = df_total[df_total["Year"] == 2025]
    if not row_2025.empty:
        cases_2025_val = row_2025["Cases"].values[0]
        loss_2025_val = row_2025["Loss_Eok"].values[0]
        lpc_2025_val = row_2025["Loss_Per_Case_Man"].values[0]
    else:
        cases_2025_val, loss_2025_val, lpc_2025_val = 0, 0, 0

    row_2023 = df_total[df_total["Year"] == 2023]
    loss_2023_val = row_2023["Loss_Eok"].values[0] if not row_2023.empty else 0
    diff_amount = loss_2025_val - loss_2023_val
    diff_pct = (diff_amount / loss_2023_val) * 100 if loss_2023_val > 0 else 0

    # [수정] 5번째 컬럼 제거 및 4개 비율 재조정 (1:1:1.2:1.5)
    k1, k2, k3, k4 = st.columns([1, 1, 1.2, 1.5])
    
    k1.metric("총 발생건수 (16~25)", f"{total_cases_all:,.0f}건")
    k2.metric("총 피해액 (16~25)", fmt_jo_eok(total_loss_all))
    k3.metric("2025년 건당 피해액", fmt_man_unit(lpc_2025_val))
    k4.metric("딥보이스 확산 영향도 (23→25)", f"{diff_amount:+,.0f}억원 ({diff_pct:+.1f}%)", delta="2023년 대비 증가", delta_color="inverse")

    st.markdown("---")

    # ------------------------------------------------------------------
    # [메인 차트] 요구사항 완벽 반영
    # ------------------------------------------------------------------
    st.markdown("#### 연도별 피해 규모 및 심각성 증가 추이")
    col_main, col_sub = st.columns([2, 1])
    
    with col_main:
        st.markdown("**① 피해액(막대) 및 발생건수(선)**")
        
        # ------------------------------------------------------------------
        # [1] 데이터 준비 (annual 데이터 사용)
        # ------------------------------------------------------------------
        # 1. 막대 그래프용 (유형별)
        target_types = ["기관사칭형", "대출사기형"]
        df_types = annual[annual['Type'].isin(target_types)].copy()
        
        # 2. 라인 그래프용 (전체 합계)
        if "전체" in annual['Type'].values:
            df_total = annual[annual['Type'] == "전체"].copy()
        else:
            df_total = annual.groupby('Year', as_index=False)[['Cases', 'Loss_Eok']].sum()
            # 건당 피해액 계산 (분석 텍스트용)
            df_total['Loss_Per_Case_Man'] = df_total.apply(
                lambda x: (x['Loss_Eok'] * 10000) / x['Cases'] if x['Cases'] > 0 else 0, axis=1
            )
            df_total['Type'] = '전체'

        # 색상 설정 (기존 코드 참조)
        type_colors = ["#003366", "#87CEEB"]
        
        # ------------------------------------------------------------------
        # [2] 차트 생성 (주신 코드 로직 유지)
        # ------------------------------------------------------------------
        # base를 df_types(annual 기반)로 변경
        base = alt.Chart(df_types).encode(x=alt.X('Year:O', title='연도'))
        stack_order = alt.Order('Type:N', sort='descending') 

        bars = base.mark_bar().encode(
            y=alt.Y('Loss_Eok:Q', title='피해액(억원)', axis=alt.Axis(titleColor='#003366', grid=True, orient='left'), stack='zero'),
            color=alt.Color('Type:N', scale=alt.Scale(domain=target_types, range=type_colors), legend=alt.Legend(title="유형")),
            order=stack_order, 
            tooltip=['Year', 'Type', alt.Tooltip('Loss_Eok', format=',', title='피해액')]
        )
        
        text_loan = base.mark_text(color='white', baseline='top', fontSize=10, dy=15).encode(
            y=alt.Y('Loss_Eok:Q', stack='zero'),
            detail='Type:N',
            order=stack_order,
            text=alt.condition((alt.datum.Type == '대출사기형') & (alt.datum.Loss_Eok > 0), alt.Text('Loss_Eok:Q', format=',.0f'), alt.value(''))
        )

        text_imperson = base.mark_text(color='white', baseline='top', fontSize=10, dy=5).encode(
            y=alt.Y('Loss_Eok:Q', stack='zero'),
            detail='Type:N',
            order=stack_order,
            text=alt.condition((alt.datum.Type != '대출사기형') & (alt.datum.Loss_Eok > 0), alt.Text('Loss_Eok:Q', format=',.0f'), alt.value(''))
        )
        
        # 라인 차트 데이터: df_total(annual 기반) 사용
        line_base = alt.Chart(df_total).encode(x=alt.X('Year:O', title='연도'))
        
        lines = line_base.mark_line(color='#FF4B4B', point=True).encode(
            y=alt.Y('Cases:Q', title='전체 발생건수(건)', axis=alt.Axis(titleColor='#FF4B4B', labelColor='#FF4B4B', orient='right', grid=False)),
            tooltip=['Year', alt.Tooltip('Cases', format=',', title='발생건수')]
        )
        
        line_text = line_base.mark_text(dy=-15, color='#FF4B4B', fontSize=11, fontWeight='bold').encode(
            y=alt.Y('Cases:Q'),
            text=alt.Text('Cases:Q', format=',')
        )

        # 2023년 이슈 표시
        anno_df = pd.DataFrame([{
            "Year": 2023, 
            "Label": "딥보이스 상용화"
        }])
        
        rule = alt.Chart(anno_df).mark_rule(
            color='red', strokeDash=[4, 4], opacity=0.7
        ).encode(
            x=alt.X('Year:O'),
            y=alt.value(310), 
            y2=alt.value(20) 
        )

        text_top = alt.Chart(anno_df).mark_text(
            color='red', dy=0, fontSize=12, fontWeight='bold', align='center', baseline='top'
        ).encode(
            x=alt.X('Year:O'),
            y=alt.value(0), 
            text='Label'
        )

        layer_bar_group = bars + text_loan + text_imperson
        layer_line_group = lines + line_text + rule + text_top 
        
        combined = alt.layer(
            layer_bar_group, 
            layer_line_group
        ).resolve_scale(
            y='independent'
        ).properties(
            height=400
        )
        
        st.altair_chart(combined, use_container_width=True)

        # ------------------------------------------------------------------
        # [3] 텍스트 분석 (df_total 사용)
        # ------------------------------------------------------------------
        try:
            # df_total에서 2023년 데이터 추출
            base_23 = df_total[df_total['Year'] == 2023].iloc[0]
            loss_23 = base_23['Loss_Eok']
            sev_23 = base_23['Loss_Per_Case_Man']
            
            analysis_msg = []
            
            # 2024년 데이터 확인
            if 2024 in df_total['Year'].values:
                data_24 = df_total[df_total['Year'] == 2024].iloc[0]
                rate_scale_24 = ((data_24['Loss_Eok'] - loss_23) / loss_23) * 100
                rate_sev_24 = ((data_24['Loss_Per_Case_Man'] - sev_23) / sev_23) * 100
                analysis_msg.append(f"""
                * **2024년 (확산기):** 23년 대비 피해 규모는 **{rate_scale_24:+.1f}%**, 
                    건당 피해액은 **{rate_sev_24:+.1f}%** 증가하며 기술 피해가 확산되었습니다.
                """)

            # 2025년 데이터 확인
            if 2025 in df_total['Year'].values:
                data_25 = df_total[df_total['Year'] == 2025].iloc[0]
                rate_scale_25 = ((data_25['Loss_Eok'] - loss_23) / loss_23) * 100
                rate_sev_25 = ((data_25['Loss_Per_Case_Man'] - sev_23) / sev_23) * 100
                analysis_msg.append(f"""
                * **2025년 (고착화기):** 23년 대비 피해 규모는 **{rate_scale_25:+.1f}%**, 
                    건당 피해액은 **{rate_sev_25:+.1f}%** 까지 치솟았습니다. 
                    이제 딥보이스 피싱은 일시적 현상이 아닌 **구조적인 위협**으로 고착화되었습니다.
                """)

            if analysis_msg:
                final_text = "\n".join(analysis_msg)
                st.info(f"""
                **📊 딥보이스 상용화(2023) 이후 범죄 가속화 양상**
                
                기술 상용화 원년인 2023년을 기준으로, 이후의 변화는 다음과 같습니다.
                {final_text}
                """, icon="📈")
            else:
                st.info("📊 2023년 딥보이스 상용화 이후, 기술 고도화에 따른 추적 관찰이 필요합니다.", icon="🔎")

        except IndexError:
            st.error("데이터에서 2023년 기준 정보를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

    # ------------------------------------------------------------------
    # [서브 차트] 건당 피해액
    # ------------------------------------------------------------------
    with col_sub:
        st.markdown("**② 건당 피해액 추이 (심각성)**")
        
        lpc_base = alt.Chart(df_total).encode(x=alt.X('Year:O', title='연도'))
        
        lpc_line = lpc_base.mark_line(point=True, color='orange').encode(
            y=alt.Y('Loss_Per_Case_Man:Q', title='건당 피해액(만원)'),
            tooltip=['Year', alt.Tooltip('Loss_Per_Case_Man', format=',.0f', title='건당피해액')]
        )
        
        lpc_text = lpc_base.mark_text(align='center', dy=-15, color='orange', fontWeight='bold').encode(
            y='Loss_Per_Case_Man:Q',
            text=alt.Text('Loss_Per_Case_Man:Q', format=',.0f')
        )

        anno_df_sub = pd.DataFrame([{"Year": 2023, "Label": "딥보이스 상용화"}])
        lpc_rule = alt.Chart(anno_df_sub).mark_rule(color='red', strokeDash=[4, 4], opacity=0.7).encode(x=alt.X('Year:O'))
        
        lpc_anno_text = alt.Chart(anno_df_sub).mark_text(
            color='red', dy=0, fontSize=11, fontWeight='bold', align='center', lineHeight=13
        ).encode(
            x=alt.X('Year:O'),
            y=alt.value(30),
            text=alt.value(["딥보이스", "상용화"]) 
        )

        st.altair_chart(
            (lpc_line + lpc_text + lpc_rule + lpc_anno_text).properties(height=500),
            use_container_width=True
        )

    # ------------------------------------------------------------------
    # [하단] 도넛 차트
    # ------------------------------------------------------------------
    st.markdown("#### 유형별 상세 비교 (선택 연도)")
    
    avail_years = sorted(df_types["Year"].unique(), reverse=True)
    sel_year = st.selectbox("비교 기준 연도 선택", avail_years, key="donut_year")
    
    donut_df = df_types[df_types["Year"] == sel_year].copy()
    
    if not donut_df.empty:
        donut_df["Cases"] = donut_df["Cases"].round(0).astype(int)
        donut_df["Type"] = pd.Categorical(donut_df["Type"], categories=target_types, ordered=True)
        donut_df = donut_df.sort_values("Type")
        
        donut_hole = 0.62
        legend_layout = dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        common_layout = dict(showlegend=True, legend=legend_layout, margin=dict(l=10, r=10, t=50, b=80), height=350)
        common_traces = dict(textposition="outside", automargin=True, textfont_size=12, sort=False)

        fig_cases = px.pie(donut_df, names="Type", values="Cases", hole=donut_hole, title="① 발생건수", 
                           color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_cases.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,}건<br>(%{percent})")
        fig_cases.update_layout(**common_layout)
        fig_cases.add_annotation(text="발생 건수", x=0.5, y=0.5, font=dict(size=14), showarrow=False)

        fig_loss = px.pie(donut_df, names="Type", values="Loss_Eok", hole=donut_hole, title="② 피해액(억원)", 
                          color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_loss.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,.0f}억<br>(%{percent})")
        fig_loss.update_layout(**common_layout)
        fig_loss.add_annotation(text="피해 금액", x=0.5, y=0.5, font=dict(size=14), showarrow=False)

        fig_lpc = px.pie(donut_df, names="Type", values="Loss_Per_Case_Man", hole=donut_hole, title="③ 건당피해액(만원)", 
                         color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_lpc.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,.0f}만<br>(%{percent})")
        fig_lpc.update_layout(**common_layout)
        fig_lpc.add_annotation(text="건당 피해액", x=0.5, y=0.5, font=dict(size=14), showarrow=False)

        d1, d2, d3 = st.columns(3)
        with d1: st.plotly_chart(fig_cases, use_container_width=True)
        with d2: st.plotly_chart(fig_loss, use_container_width=True)
        with d3: st.plotly_chart(fig_lpc, use_container_width=True)
# ==============================================================================
# TAB 2: 연령별 피해
# ==============================================================================
with tab2:
    st.subheader("연령대별 피해 현황")
    
    if not age.empty:
        age_cols = [c for c in age.columns if "대" in c]
        if not age_cols: 
            age_cols = ["20대이하", "30대", "40대", "50대", "60대", "70대이상"]
        
        # 파일에 존재하는 컬럼만 선택
        valid_cols = [c for c in age_cols if c in age.columns]
        
        if valid_cols:
            age_sub = age[["Year"] + valid_cols].copy()
            age_long = age_sub.melt(id_vars="Year", value_vars=valid_cols, var_name="AgeGroup", value_name="Victims")
            age_long["Year"] = pd.to_numeric(age_long["Year"], errors='coerce').fillna(0).astype(int)
            
            # [수정] 외부 year_range 변수 대신 2016~2025 전체 범위를 강제로 설정하여 2025년이 잘리거나 누락되지 않게 함
            age_long_f = age_long[(age_long["Year"] >= 2016) & (age_long["Year"] <= 2025)]

            col_pie, col_trend = st.columns([1, 1.5])
            
            with col_pie:
                st.markdown("##### 연도별 피해 비중")
                valid_years = sorted(age_long_f["Year"].unique(), reverse=True)
                target_year_age = st.selectbox("확인할 연도를 선택하세요", valid_years, key="age_year_select")
                
                pie_data = age_long_f[age_long_f["Year"] == target_year_age].copy()
                
                if pie_data.empty:
                    st.warning("선택한 연도의 데이터가 없습니다.")
                else:
                    pie_data = pie_data.sort_values("Victims", ascending=False)
                    fig_age = px.pie(pie_data, names="AgeGroup", values="Victims", title=f"{target_year_age}년 연령대별 피해 분포", hole=0.4)
                    fig_age.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="<b>%{label}</b><br>피해자 수: %{value:,}명<br>비율: %{percent}<extra></extra>")
                    fig_age.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), margin=dict(l=20, r=20, t=50, b=50), height=400)
                    st.plotly_chart(fig_age, use_container_width=True)

            with col_trend:
                st.markdown("##### 연령대별 피해 추이 (Trend)")
                age_line = alt.Chart(age_long_f).mark_line(point=True).encode(
                    x=alt.X("Year:O", title="연도"),
                    y=alt.Y("Victims:Q", title="피해자 수(명)"),
                    color=alt.Color("AgeGroup:N", legend=alt.Legend(title="연령대")),
                    tooltip=["Year", "AgeGroup", alt.Tooltip("Victims", format=",", title="피해자 수")]
                ).properties(height=450)
                # 2025년 강조 룰
                rule = alt.Chart(pd.DataFrame({'Year': [2025]})).mark_rule(color='red', strokeDash=[4,4]).encode(x='Year:O')
                
                st.altair_chart(age_line + rule, use_container_width=True)
        else:
            st.error("연령대 데이터 컬럼을 찾을 수 없습니다.")
    else:
        st.warning("연령별 데이터(age.csv)가 로드되지 않았습니다.")
# ==============================================================================
# TAB 3: 월별 추이 & 이슈
# ==============================================================================
with tab3:
    # ==============================================================================
    # [1] 데이터 전처리 & KPI 계산
    # ==============================================================================
    target_years = [2024, 2025]
    avg_loss_per_case = 25000000 
    
    if 'annual' in locals() and not annual.empty:
        df_calc = annual[(annual["Type"]=="전체") & (annual["Year"].isin(target_years))].copy()
        if not df_calc.empty:
            sum_loss = df_calc["Loss_Eok"].sum() * 100000000 
            sum_cases = df_calc["Cases"].sum()
            if sum_cases > 0:
                avg_loss_per_case = sum_loss / sum_cases

    social_cost_factor = 1.5 
    unit_prevent_value = avg_loss_per_case * social_cost_factor

    total_occur = 0
    total_arrest = 0
    total_gap = 0
    avg_arrest_rate = 0.0
    df_arr = pd.DataFrame()

    source_df = annual_f if 'annual_f' in locals() and not annual_f.empty else annual
    
    if not source_df.empty:
        if 'Arrest_Cases' not in source_df.columns and '검거건수' in source_df.columns:
            source_df = source_df.rename(columns={'검거건수': 'Arrest_Cases'})
        if 'Cases' not in source_df.columns and '발생건수' in source_df.columns:
            source_df = source_df.rename(columns={'발생건수': 'Cases'})

        if "Arrest_Cases" in source_df.columns:
            if 'Type' in source_df.columns:
                df_arr = source_df[(source_df["Type"] == "전체") & (source_df["Year"] < 2025)].copy()
            else:
                df_arr = source_df[source_df["Year"] < 2025].copy()
                
            df_arr = df_arr.sort_values("Year")
            
            if not df_arr.empty:
                total_occur = df_arr["Cases"].sum()
                total_arrest = df_arr["Arrest_Cases"].sum()
                total_gap = total_occur - total_arrest
                if total_occur > 0:
                    avg_arrest_rate = (total_arrest / total_occur) * 100

    # ==============================================================================
    # [2] 상단 KPI
    # ==============================================================================
    st.markdown("### '3년 늦은 정의': 물리적 대응의 한계")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("총 발생 (누적)", f"{total_occur:,.0f} 건")
    with k2: st.metric("총 검거 (누적)", f"{total_arrest:,.0f} 건", delta=f"{avg_arrest_rate:.1f}% (검거율)")
    with k3: st.metric("미검거 잔존", f"{total_gap:,.0f} 건", delta="-대응 시차 발생", delta_color="inverse")
    with k4: st.metric("1건당 예방가치", f"{unit_prevent_value/10000:,.0f} 만원", delta="사회적 비용 포함")

    st.markdown("---")

 # ==============================================================================
    # [3] 중간 차트: 16~24년 평균 검거율 비교 (Rank)
    # ==============================================================================
    st.subheader("📊 범죄별 평균 검거율 비교 (2016~2024)")
    st.caption("지난 9년간의 평균 데이터를 분석한 결과, **보이스피싱의 검거율**이 타 범죄 대비 현저히 낮음을 확인할 수 있습니다.")

    if 'df_rate' in locals() and not df_rate.empty:
        import plotly.graph_objects as go
        
        # 1. 데이터 가공 (평균 계산 및 정렬)
        # 2016~2024년 데이터 필터링 (인덱스가 연도인 경우)
        target_df = df_rate.loc[2016:2024]
        
        # 컬럼별 평균 계산
        avg_rates = target_df.mean()
        
        # 정렬: Plotly 가로 막대는 리스트의 마지막 요소가 차트의 '맨 위'에 그려집니다.
        # 따라서 '내림차순'으로 보이게 하려면 값을 '오름차순(Ascending)'으로 정렬해야 합니다.
        avg_rates = avg_rates.sort_values(ascending=True)
        
        # 2. 색상 설정 ('보이스피싱'만 빨간색, 나머지는 회색)
        colors = []
        opacity_vals = []
        
        for crime in avg_rates.index:
            if crime == '보이스피싱':
                colors.append('#D32F2F')  # 강조색 (빨강)
                opacity_vals.append(1.0)  # 불투명
            else:
                colors.append('#B0BEC5')  # 기본색 (회색)
                opacity_vals.append(0.6)  # 약간 투명하게
        
        # 3. 차트 그리기
        fig_rank = go.Figure()

        fig_rank.add_trace(go.Bar(
            x=avg_rates.values,       # X축: 검거율
            y=avg_rates.index,        # Y축: 범죄명
            orientation='h',          # 가로 막대
            marker=dict(color=colors, opacity=opacity_vals), # 색상 적용
            text=avg_rates.values,
            texttemplate='%{text:.1f}%', # 소수점 1자리
            textposition='outside',   # 막대 끝에 숫자 표시
            hovertemplate='<b>%{y}</b><br>평균 검거율: %{x:.1f}%<extra></extra>'
        ))

        # 4. 레이아웃 설정
        fig_rank.update_layout(
            title=dict(text="<b>[Warning] 보이스피싱 검거율 순위</b>", font=dict(size=16)),
            xaxis=dict(
                title="평균 검거율 (%)", 
                range=[0, 115], # 텍스트 여백 확보
                showgrid=True,
                gridcolor='#eee'
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=12)
            ),
            height=600, # 항목이 많을 수 있으므로 높이 확보
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        # 보이스피싱 위치에 주석(Annotation) 추가
        vp_rate = avg_rates.get('보이스피싱', 0)
        if vp_rate > 0:
            fig_rank.add_annotation(
                x=vp_rate,
                y='보이스피싱',
                text="<b>📉 Lowest Efficiency</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                ax=60, ay=0,
                bgcolor="white", bordercolor="#D32F2F", borderwidth=1,
                font=dict(color="#D32F2F", size=11)
            )

        st.plotly_chart(fig_rank, use_container_width=True)

        # 텍스트 인사이트
        vp_rank = len(avg_rates) - list(avg_rates.index).index('보이스피싱') # 뒤에서부터 순위 계산
        st.error(f"""
        **🚨 분석 결과**
        * **보이스피싱**의 9년 평균 검거율은 **{vp_rate:.1f}%**로, 전체 {len(avg_rates)}개 범죄 유형 중 **{vp_rank}위**를 기록했습니다.
        * 이는 물리적 검거 방식이 비대면 범죄인 보이스피싱에는 효과적이지 않음을 증명합니다.
        """)

    else:
        st.warning("데이터가 로드되지 않았습니다.")

    # 3. [Simulator]
    st.markdown("---")
    st.subheader("우리 기술의 가치 시뮬레이션 (ROI)")
    
    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        st.markdown("**예방 건수 설정**")
        detect_target = st.slider("월간 예상 탐지 건수", min_value=1, max_value=1000, value=100, step=10)
        st.caption(f"기준: 최근 {min(target_years)}~{max(target_years)}년 평균 피해액")
        
        with st.expander("1.5배 산출 근거 (UK Home Office)"):
            st.markdown("""
            **🇬🇧 영국 내무부 (2018 Report)**
            * **Page 15, Figure 2:** Cost of Crime Model
            * **Total Cost** = Financial Loss + Emotional Impact + CJS Costs
            * 위 공식에 따라 직접 피해액의 **약 1.5배~2배**가 실제 사회적 비용으로 산출됨.
            """)

    with col_sim2:
        monthly_saving = unit_prevent_value * detect_target
        yearly_saving = monthly_saving * 12
        
        st.markdown(f"#### 1건 탐지 시: :blue[{unit_prevent_value/10000:,.0f}만 원] 사회적 비용 절감")
        st.caption(f"(직접 피해액 {avg_loss_per_case/10000:,.0f}만원 × 사회적 비용 계수 {social_cost_factor}배)")
        
        st.success(f"""
        **영국 내무부 비용 모델 (The Cost of Crime Model) 적용**
        
        $$
        \\text{{Total Benefit}} = \\text{{Financial Loss}} + \\text{{Emotional Impact}} + \\text{{CJS Costs}}
        $$
        
        * **직접 피해 방어:** {avg_loss_per_case/10000:,.0f}만 원
        * **사회적 비용 절감:** +{(unit_prevent_value - avg_loss_per_case)/10000:,.0f}만 원 (수사비용, 정신적 피해 등)
        
        **월 {detect_target}건 차단 시: 월 {monthly_saving/100000000:,.2f} 억원 경제 효과**
        """)

    st.info("""
    **Insight for Investors:**
    경찰의 검거(사후 조치)는 피해금을 돌려받기 어렵지만, **AI의 탐지(사전 차단)는 피해액 전액(100%)을 보존**합니다.
    이것이 우리가 단순한 '범죄 예방'을 넘어 **'금융 자산 보호 솔루션'**인 이유입니다.
    """)
# ==============================================================================
# TAB 4: 딥보이스 vs 실제 음성 비교 (신규 생성)
# ==============================================================================
with tab4:
    # ##########################################################################
    # [SECTION A] 딥보이스 vs 실제 음성 주파수 분석
    # ##########################################################################
    st.header("1️딥보이스 기술적 탐지 (주파수 분석)")
    st.markdown("""
    **탐지 원리 (High-Frequency Cutoff):**
    * **실제 음성:** 16kHz 이상의 고주파 대역까지 에너지가 자연스럽게 뻗어 있습니다.
    * **딥보이스:** 학습 데이터 한계로 **11kHz 부근에서 에너지가 뚝 끊기거나(Cut-off)** 사라집니다.
    """)

    # 1. 오디오 파일 자동 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__))
    real_files_list = sorted(glob.glob(os.path.join(current_dir, "문장*.m4a")))
    fake_files_list = sorted(glob.glob(os.path.join(current_dir, "딥보이스*.mp3")))

    # 2. 분석 함수
    @st.cache_data
    def calculate_frequency_stats(real_paths, fake_paths, target_sr=44100):
        def get_avg_spectrum(file_list):
            specs = []
            for f in file_list:
                try:
                    y, sr = librosa.load(f, sr=target_sr) 
                    D = np.abs(librosa.stft(y))
                    S_db = librosa.amplitude_to_db(D, ref=np.max)
                    specs.append(np.mean(S_db, axis=1))
                except:
                    continue
            if not specs: return None, None
            return np.array(specs), librosa.fft_frequencies(sr=target_sr)

        real_specs, freqs = get_avg_spectrum(real_paths)
        fake_specs, _ = get_avg_spectrum(fake_paths)
        return real_specs, fake_specs, freqs

    # 3. 시각화
    if not real_files_list or not fake_files_list:
        st.warning("분석할 오디오 파일(문장*.m4a, 딥보이스*.mp3)을 폴더에서 찾지 못했습니다.")
    else:
        with st.spinner("주파수 스펙트럼 분석 중..."):
            real_specs, fake_specs, freqs = calculate_frequency_stats(real_files_list, fake_files_list)

        if real_specs is not None and fake_specs is not None:
            mean_real = np.mean(real_specs, axis=0)
            min_real = np.min(real_specs, axis=0)
            max_real = np.max(real_specs, axis=0)
            mean_fake = np.mean(fake_specs, axis=0)
            min_fake = np.min(fake_specs, axis=0)
            max_fake = np.max(fake_specs, axis=0)

            fig = go.Figure()
            # Real 범위
            fig.add_trace(go.Scatter(x=freqs, y=max_real, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=freqs, y=min_real, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='Real 범위', showlegend=False, hoverinfo='skip'))
            # Fake 범위
            fig.add_trace(go.Scatter(x=freqs, y=max_fake, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=freqs, y=min_fake, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='Fake 범위', showlegend=False, hoverinfo='skip'))
            # 평균 선
            fig.add_trace(go.Scatter(x=freqs, y=mean_real, mode='lines', name='Real (실제 음성)', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=freqs, y=mean_fake, mode='lines', name='Fake (딥보이스)', line=dict(color='red', width=2)))
            # 임계치
            threshold_hz = 11025
            fig.add_vline(x=threshold_hz, line_width=2, line_dash="dash", line_color="green")
            
            fig.update_layout(
                title="<b>주파수별 에너지 분포 (Real vs Fake)</b>",
                xaxis_title="주파수 (Hz)", yaxis_title="에너지 (dB)", template="plotly_white", height=450,
                xaxis=dict(range=[0, 22050]), yaxis=dict(range=[-80, 0]), legend=dict(x=0.8, y=0.95),
                annotations=[dict(x=threshold_hz, y=-10, xref="x", yref="y", text="임계치 (11kHz)", showarrow=True, arrowhead=1, ax=40, ay=-40, font=dict(color="green", size=12))]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            ### 전문가 분석: 주파수 스펙트럼 심층 해석
            **1. 나이퀴스트 정리(Nyquist Theorem)와 '11kHz의 비밀'**
            * **원리:** 디지털 신호는 샘플링 속도의 **절반(1/2)**까지만 주파수를 표현할 수 있습니다.
            * **증거:** 이 파일은 48,000Hz 형식이지만, **딥보이스는 약 11,025Hz에서 에너지가 사라집니다.** 이는 **저화질(22kHz) AI 생성물**을 파일 형식만 고음질로 강제 변환했다는 증거입니다.
            
            **2. 🔵 실제 음성 vs 🔴 딥보이스**
            * **정상:** 16kHz 이상 초고주파 대역까지 에너지가 살아있고, 미세한 노이즈(Range 두께)가 관찰됩니다.
            * **딥보이스:** 임계치 이후 에너지가 -80dB(완벽한 무음)로 떨어지며, 인위적으로 깨끗한 직선을 그립니다.
            """)

    st.markdown("---") 
    st.header("유형별 범죄 시나리오 흐름 (Scenario Flow)")
    st.caption("데이터(`단계_전이.csv`)를 분석하여 가장 빈번한 '핵심 경로(Main Path)'와 파생되는 '변칙 경로'를 시각화했습니다.")

    # 1. 유형 선택 UI
    type_selection = st.radio(
        "분석할 범죄 유형을 선택하세요:",
        ("기관사칭형 (검찰/금감원 사칭)", "대출사기형 (저금리 대출 빙자)"),
        horizontal=True
    )

    # --------------------------------------------------------------------------
    # 공통 헬퍼 함수
    # --------------------------------------------------------------------------
    def get_transition_ratio(df, source, target):
        """이동 비율 계산 (데이터가 없으면 0% 리턴)"""
        if df is None or df.empty: return "0%"
        # 해당 단계(source)에서 나가는 전체 건수
        total_out = df[df['from_stage'] == source]['count'].sum()
        if total_out == 0: return "0%"
        # 특정 다음 단계(target)로 가는 건수
        target_count = df[(df['from_stage'] == source) & (df['to_stage'] == target)]['count'].sum()
        return f"{(target_count / total_out) * 100:.1f}%"

    # --------------------------------------------------------------------------
    # 2. 차트 생성 로직 (기관사칭 / 대출사기)
    # --------------------------------------------------------------------------
    
    # (1) 기관사칭형 차트
    def generate_impersonation_chart(df_trans):
        df = df_trans[df_trans['label_name'] == '기관사칭']
        dot = 'digraph G {\n'
        dot += '  rankdir="LR"; splines=ortho;\n'
        dot += '  node [fontname="Malgun Gothic", shape="box", style="rounded,filled", fillcolor="white", penwidth="1.5", height="0.5"];\n'
        dot += '  edge [fontname="Malgun Gothic", fontsize="10", arrowhead="vee"];\n'

        # 노드 정의 (키워드 없이 깔끔하게)
        dot += '  "권위_사칭" [fillcolor="#E3F2FD"];\n'
        dot += '  "위협_사건제시" [fillcolor="#E3F2FD"];\n'
        dot += '  "정보확보_인증" [fillcolor="#BBDEFB"];\n'
        dot += '  "top_add" [label="추가지시_마무리", fillcolor="#FFF3E0"];\n'
        dot += '  "top_ctrl" [label="통제_압박", fillcolor="#FFEBEE"];\n'
        dot += '  "bot_ctrl" [label="통제_압박", fillcolor="#FFEBEE"];\n'
        dot += '  "bot_add" [label="추가지시_마무리", fillcolor="#FFF3E0"];\n'

        # 메인 흐름
        r1 = get_transition_ratio(df, "권위_사칭", "위협_사건제시")
        r2 = get_transition_ratio(df, "위협_사건제시", "정보확보_인증")
        dot += f'  "권위_사칭" -> "위협_사건제시" [label="{r1}", color="#003366", penwidth="2.0"];\n'
        dot += f'  "위협_사건제시" -> "정보확보_인증" [label="{r2}", color="#003366", penwidth="2.0"];\n'

        # 분기점 (상단/하단 경로)
        ru1 = get_transition_ratio(df, "정보확보_인증", "추가지시_마무리")
        ru2 = get_transition_ratio(df, "추가지시_마무리", "통제_압박")
        dot += f'  "정보확보_인증" -> "top_add" [label="상단:{ru1}", color="#FF9800"];\n'
        dot += f'  "top_add" -> "top_ctrl" [label="{ru2}", color="#FF9800"];\n'

        rd1 = get_transition_ratio(df, "정보확보_인증", "통제_압박")
        rd2 = get_transition_ratio(df, "통제_압박", "추가지시_마무리")
        dot += f'  "정보확보_인증" -> "bot_ctrl" [label="하단:{rd1}", color="#D32F2F"];\n'
        dot += f'  "bot_ctrl" -> "bot_add" [label="{rd2}", color="#D32F2F"];\n'
        
        dot += '}'
        return dot

    # (2) 대출사기형 차트
    def generate_loan_chart(df_trans):
        df = df_trans[df_trans['label_name'] == '대출사기']
        dot = 'digraph G {\n'
        dot += '  rankdir="LR"; splines=curved;\n' # 순환 구조라 곡선 사용
        dot += '  node [fontname="Malgun Gothic", shape="box", style="rounded,filled", fillcolor="white", penwidth="1.5", height="0.5"];\n'
        dot += '  edge [fontname="Malgun Gothic", fontsize="10", arrowhead="vee"];\n'

        # 노드 정의
        dot += '  "대출권유_조건제시" [fillcolor="#E0F7FA"];\n'
        dot += '  "추가지시_마무리" [fillcolor="#E0F2F1"];\n'
        dot += '  "통제_압박" [fillcolor="#FFF3E0"];\n'
        dot += '  "금전요구" [fillcolor="#FFCCBC"];\n'
        dot += '  "선입금_보증료" [fillcolor="#FFAB91", penwidth="2.5", color="#D32F2F"];\n' # 피해 발생 지점 강조

        # 엣지 연결 (분석된 Top Transitions 반영)
        r1 = get_transition_ratio(df, "대출권유_조건제시", "추가지시_마무리")
        dot += f'  "대출권유_조건제시" -> "추가지시_마무리" [label="{r1}", color="#006064", penwidth="2.0"];\n'

        r2 = get_transition_ratio(df, "추가지시_마무리", "통제_압박")
        dot += f'  "추가지시_마무리" -> "통제_압박" [label="{r2}", color="#006064", penwidth="2.0"];\n'

        # 순환 구간 (통제 <-> 권유)
        r_back = get_transition_ratio(df, "통제_압박", "대출권유_조건제시")
        dot += f'  "통제_압박" -> "대출권유_조건제시" [label="회유:{r_back}", color="#BBBBBB", style="dashed", constraint=false];\n'

        r3 = get_transition_ratio(df, "통제_압박", "금전요구")
        dot += f'  "통제_압박" -> "금전요구" [label="{r3}", color="#BF360C", penwidth="2.5"];\n'

        r4 = get_transition_ratio(df, "금전요구", "선입금_보증료")
        dot += f'  "금전요구" -> "선입금_보증료" [label="{r4}", color="#D32F2F", penwidth="3.0"];\n'

        dot += '}'
        return dot

    # --------------------------------------------------------------------------
    # 3. 화면 렌더링 (차트 + 카드)
    # --------------------------------------------------------------------------
    if 'df_transitions' in locals() and not df_transitions.empty:
        
        # (A) 차트 영역
        if "기관사칭" in type_selection:
            dot_code = generate_impersonation_chart(df_transitions)
            st.graphviz_chart(dot_code, use_container_width=True)
            
            # 기관사칭 상세 키워드 정의
            keyword_data = [
                {
                    "title": "1. 권위 사칭",
                    "desc": "기관을 사칭해 신뢰 형성",
                    "words": ["서울중앙지검", "첨단범죄수사팀", "금융감독원", "과장/수사관", "사건번호", "공문 발송", "녹취 시작", "사건 번호"],
                    "color": "blue"
                },
                {
                    "title": "2. 위협·사건",
                    "desc": "범죄 연루 사실 통보",
                    "words": ["대포통장 개설", "명의 도용", "중고나라 사기", "피해자 입증", "자금 세탁", "계좌 동결", "고소장 접수", "출석 요구서"],
                    "color": "blue"
                },
                {
                    "title": "3. 정보확보",
                    "desc": "개인/금융정보 탈취",
                    "words": ["자산 내역 확인", "IP 주소 추적", "OTP 번호", "신분증 촬영", "팀뷰어(원격)", "본인 인증", "계좌 비밀번호", "공인인증서"],
                    "color": "blue"
                },
                {
                    "title": "4. 통제·압박",
                    "desc": "심리적 고립 및 협박",
                    "words": ["조용한 곳 이동", "제3자 발설 금지", "전화 끊지 마세요", "주변 소음 차단", "수사 방해", "공무집행방해", "구속 영장 청구"],
                    "color": "red"
                },
                {
                    "title": "5. 마무리",
                    "desc": "금전 탈취 및 증거인멸",
                    "words": ["국가안전계좌 이체", "현금 인출 후 전달", "상품권 핀번호", "악성 앱 설치", "대출 실행", "카카오톡 탈퇴", "통화 기록 삭제"],
                    "color": "red"
                }
            ]
        else:
            # 대출사기 해석 팁
            st.info(" **해석 포인트:** 대출 권유와 통제(심사) 단계를 오가며 신뢰를 쌓은 후, **'보증료/선입금'** 명목으로 금전을 요구하는 구조가 뚜렷합니다.")
            
            dot_code = generate_loan_chart(df_transitions)
            st.graphviz_chart(dot_code, use_container_width=True)
            
            # 대출사기 상세 키워드 정의
            keyword_data = [
                {
                    "title": "1. 대출권유",
                    "desc": "저금리 대출 유혹",
                    "words": ["정부지원 자금", "저금리 대환 대출", "햇살론/버팀목", "신용등급 상향", "1금융권", "마이너스 통장", "신청 대상자", "무보증/무담보"],
                    "color": "blue"
                },
                {
                    "title": "2. 추가지시",
                    "desc": "개인정보 및 앱 설치",
                    "words": ["금융기관 앱 설치", "신분증 사본 전송", "기존 대출 상환", "카톡 친구추가", "신청서 작성", "입출금 내역", "재직 증명서"],
                    "color": "blue"
                },
                {
                    "title": "3. 통제·압박",
                    "desc": "가짜 심사 및 교란",
                    "words": ["심사 진행 중", "신용 평점 부족", "법무사 통화", "금융법 위반", "전산 처리", "모니터링 감지", "부결 사유", "중복 신청"],
                    "color": "red"
                },
                {
                    "title": "4. 금전요구",
                    "desc": "각종 비용 청구",
                    "words": ["보증 보험료", "예치금 납부", "공탁금 설정", "인지세/수수료", "상환 처리 비용", "신용 보증금", "계좌 해지 비용"],
                    "color": "red"
                },
                {
                    "title": "5. 선입금",
                    "desc": "최종 금전 갈취",
                    "words": ["가상계좌 발급", "편법 상환 처리", "선입금 입금", "담당자 계좌", "무통장 입금", "즉시 이체", "현금 인출 전달"],
                    "color": "red"
                }
            ]

        # (B) 하단 키워드 카드 영역 (공통 렌더링)
        st.markdown("#### 단계별 주요 식별 키워드 (Keyword Detail)")
        st.caption(f"선택하신 **{type_selection.split(' ')[0]}**에서 범인이 실제로 사용하는 상세 어휘 리스트입니다.")
        
        cols = st.columns(5)
        for i, item in enumerate(keyword_data):
            with cols[i]:
                # 테두리 및 텍스트 색상
                border_color = item['color']
                
                with st.container(border=True):
                    # 제목
                    if border_color == "blue":
                        st.markdown(f"** {item['title']}**")
                    else:
                        st.markdown(f"**:red[{item['title']}]**")
                    
                    # 설명
                    st.markdown(f"<div style='font-size:12px; color:gray; margin-bottom:8px;'>{item['desc']}</div>", unsafe_allow_html=True)
                    
                    # 단어 리스트
                    for word in item['words']:
                        if border_color == "red":
                            st.markdown(f"- :red[{word}]")
                        else:
                            st.markdown(f"- {word}")

    else:
        st.error("데이터(단계_전이.csv)를 불러올 수 없습니다.")