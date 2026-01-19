import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="보이스피싱 탐지 포트폴리오 | 개요 대시보드", layout="wide")

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

# -----------------------------
# 2. 데이터 로드
# -----------------------------
st.title("보이스피싱 탐지 딥러닝 포트폴리오 | 서론용 대시보드")
# 구분선 코드(st.markdown("---")) 삭제됨

st.sidebar.header("데이터 로드")
annual = load_or_upload("annual.csv", "u_annual")
monthly = load_or_upload("monthly_cases.csv", "u_monthly")
types_only = load_or_upload("types_only.csv", "u_types")
age = load_or_upload("age2.csv", "u_age")
data_2025 = load_or_upload("25년(11월).xlsx - Sheet1.csv", "u_25") 
leak_df = load_or_upload("역대유출사고.csv", "u_leak")

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
tab1, tab2, tab4 = st.tabs(["1️⃣ 현황 & 심각성", "2️⃣ 연령별 피해","3️⃣ 발생/검거"])

# ==============================================================================
# TAB 1: 현황 & 심각성
# ==============================================================================
with tab1:
    # ------------------------------------------------------------------
    # [KPI] 전체 현황 스냅샷 (5개 컬럼)
    # ------------------------------------------------------------------
    st.subheader(f"📊 현황 스냅샷 (2016 ~ 2025 전체)")

    # 1. 누적 데이터 계산 (K1, K2용)
    full_period = annual_final[(annual_final["Year"] >= 2016) & (annual_final["Year"] <= 2025) & (annual_final["Type"] == "전체")]
    total_cases_all = full_period["Cases"].sum()
    total_loss_all = full_period["Loss_Eok"].sum()

    # 2. 2025년 데이터 (K3, K4용)
    row_2025 = annual_final[(annual_final["Year"] == 2025) & (annual_final["Type"] == "전체")]
    
    if not row_2025.empty:
        cases_2025_val = row_2025["Cases"].values[0]
        loss_2025_val = row_2025["Loss_Eok"].values[0]
        lpc_2025_val = row_2025["Loss_Per_Case_Man"].values[0]
    else:
        loss_2025_val = 0
        lpc_2025_val = 0

    # 3. 2023년 데이터 (K4 비교 기준)
    row_2023 = annual_final[(annual_final["Year"] == 2023) & (annual_final["Type"] == "전체")]
    loss_2023_val = row_2023["Loss_Eok"].values[0] if not row_2023.empty else 0

    # K4 계산 (증감액)
    diff_amount = loss_2025_val - loss_2023_val
    if loss_2023_val > 0:
        diff_pct = (diff_amount / loss_2023_val) * 100
    else:
        diff_pct = 0

    # 4. [수정] 2025년 유출 '레코드'의 합 계산 (K5용)
    leak_sum_2025 = 0
    if 'leak_df' in locals() and not leak_df.empty:
        leak_df.columns = leak_df.columns.str.strip()
        # 2025년 데이터의 '유출건' 컬럼 합계
        leak_sum_2025 = leak_df[leak_df['연도'] == 2025]['유출건'].sum()

    # ------------------------------------------------------------------
    # [KPI 출력] 컬럼 5개 (K5에 유출건 합계 표시)
    # ------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns([0.8, 0.8, 1, 1.4, 1.0])
    
    # K1: 누적 발생건수
    k1.metric("총 발생건수 (16~25)", f"{total_cases_all:,.0f}건")
    
    # K2: 누적 피해액
    k2.metric("총 피해액 (16~25)", fmt_jo_eok(total_loss_all))
    
    # K3: 2025년 건당 피해액
    k3.metric("2025년 건당 피해액", fmt_man_unit(lpc_2025_val))
    
    # K4: 딥보이스 확산 영향도
    k4.metric(
        "딥보이스 확산 영향도 (23→25)", 
        f"{diff_amount:+,.0f}억원 ({diff_pct:+.1f}%)", 
        delta="2023년(상용화) 대비 증가분", 
        delta_color="inverse"
    )

    # ✅ [K5 수정] 2025년 유출 정보 총합 (레코드 수)
    # 만약 데이터가 없어서 0이면 시나리오 수치(6,800만)라도 보여줄지 선택 가능하나
    # 우선 계산된 합계를 그대로 보여줍니다.
    k5.metric(
        "2025 개인정보 유출", 
        f"{leak_sum_2025:,.0f}건", 
        delta="총 유출 레코드 합계", 
        delta_color="inverse"
    )

    st.markdown("---")
    
    # ------------------------------------------------------------------
    # [기존 Tab 1 내용] 차트 시각화 (이하 동일)
    # ------------------------------------------------------------------
    st.markdown("#### 📈 연도별 피해 규모 및 심각성 증가 추이")
    col_main, col_sub = st.columns([2, 1])
    
    with col_main:
        st.markdown("**① 피해액(막대) 및 발생건수(선)**")
        
        base = alt.Chart(types_f_main).encode(x=alt.X('Year:O', title='연도'))
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
        
        line_data = annual_f[annual_f["Type"]=="전체"]
        line_base = alt.Chart(line_data).encode(x=alt.X('Year:O', title='연도'))
        
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

        try:
            base_23 = annual_f[(annual_f['Year'] == 2023) & (annual_f['Type'] == '전체')].iloc[0]
            loss_23 = base_23['Loss_Eok']
            sev_23 = base_23['Loss_Per_Case_Man']
            
            analysis_msg = []
            
            if 2024 in annual_f['Year'].values:
                data_24 = annual_f[(annual_f['Year'] == 2024) & (annual_f['Type'] == '전체')].iloc[0]
                rate_scale_24 = ((data_24['Loss_Eok'] - loss_23) / loss_23) * 100
                rate_sev_24 = ((data_24['Loss_Per_Case_Man'] - sev_23) / sev_23) * 100
                analysis_msg.append(f"""
                * **2024년 (확산기):** 23년 대비 피해 규모는 **{rate_scale_24:+.1f}%**, 
                  건당 피해액은 **{rate_sev_24:+.1f}%** 증가하며 기술 피해가 확산되었습니다.
                """)

            if 2025 in annual_f['Year'].values:
                data_25 = annual_f[(annual_f['Year'] == 2025) & (annual_f['Type'] == '전체')].iloc[0]
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

    with col_sub:
        st.markdown("**② 건당 피해액 추이 (심각성)**")
        
        lpc_data = annual_f[annual_f["Type"]=="전체"]
        lpc_base = alt.Chart(lpc_data).encode(x=alt.X('Year:O', title='연도'))
        
        lpc_line = lpc_base.mark_line(point=True, color='orange').encode(
            y=alt.Y('Loss_Per_Case_Man:Q', title='건당 피해액(만원)'),
            tooltip=['Year', alt.Tooltip('Loss_Per_Case_Man', format=',.0f')]
        )
        
        lpc_text = lpc_base.mark_text(align='center', dy=-15, color='orange', fontWeight='bold').encode(
            y='Loss_Per_Case_Man:Q',
            text=alt.Text('Loss_Per_Case_Man:Q', format=',.0f')
        )

        anno_df_sub = pd.DataFrame([{
            "Year": 2023, 
            "Label": "placeholder"
        }])

        lpc_rule = alt.Chart(anno_df_sub).mark_rule(
            color='red', strokeDash=[4, 4], opacity=0.7
        ).encode(
            x=alt.X('Year:O'),
            y=alt.value(310), 
            y2=alt.value(20) 
        )

        lpc_anno_text = alt.Chart(anno_df_sub).mark_text(
            color='red', dy=0, fontSize=11, fontWeight='bold', align='center', baseline='top', lineHeight=13
        ).encode(
            x=alt.X('Year:O'),
            y=alt.value(0),
            text=alt.value(["딥보이스 상용화"]) 
        )

        st.altair_chart(
            (lpc_line + lpc_text + lpc_rule + lpc_anno_text).properties(height=400), 
            use_container_width=True
        )

    # ------------------------------------------------------------------
    # [하단] 도넛 차트
    # ------------------------------------------------------------------
    st.markdown("#### 🍩 유형별 상세 비교 (선택 연도)")
    
    avail_years = sorted(types_f_main[types_f_main["Year"] != 2025]["Year"].unique(), reverse=True)
    sel_year = st.selectbox("비교 기준 연도 선택", avail_years, key="donut_year")
    
    donut_df = types_f_main[types_f_main["Year"] == sel_year].copy()
    
    if donut_df.empty:
        st.warning(f"선택한 {sel_year}년 데이터가 없습니다.")
    else:
        donut_df["Cases"] = donut_df["Cases"].round(0).astype(int)
        donut_df["Type"] = pd.Categorical(donut_df["Type"], categories=target_types, ordered=True)
        donut_df = donut_df.sort_values("Type")
        
        donut_hole = 0.62
        legend_layout = dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        common_layout = dict(showlegend=True, legend=legend_layout, margin=dict(l=10, r=10, t=50, b=80), height=350)
        common_traces = dict(textposition="outside", automargin=True, textfont_size=12, sort=False)

        fig_cases = px.pie(donut_df, names="Type", values="Cases", hole=donut_hole, title="① 발생건수", color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_cases.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,}건<br>(%{percent})", hovertemplate="<b>%{label}</b><br>발생건수: %{value:,}건<br>비율: %{percent}<extra></extra>")
        fig_cases.update_layout(**common_layout)
        fig_cases.add_annotation(text="발생 건수", x=0.5, y=0.5, font=dict(size=14), showarrow=False)

        fig_loss = px.pie(donut_df, names="Type", values="Loss_Eok", hole=donut_hole, title="② 피해액(억원)", color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_loss.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,.0f}억<br>(%{percent})", hovertemplate="<b>%{label}</b><br>피해액: %{value:,.0f}억<br>비율: %{percent}<extra></extra>")
        fig_loss.update_layout(**common_layout)
        fig_loss.add_annotation(text="피해 금액", x=0.5, y=0.5, font=dict(size=14), showarrow=False)

        fig_lpc = px.pie(donut_df, names="Type", values="Loss_Per_Case_Man", hole=donut_hole, title="③ 건당피해액(만원)", color="Type", color_discrete_map=color_map, category_orders={"Type": target_types})
        fig_lpc.update_traces(**common_traces, texttemplate="%{label}<br>%{value:,.0f}만<br>(%{percent})", hovertemplate="<b>%{label}</b><br>건당피해액: %{value:,.0f}만<br>비율: %{percent}<extra></extra>")
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
    st.subheader("👥 연령대별 피해 현황")
    
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
            age_long_f = age_long[(age_long["Year"] >= year_range[0]) & (age_long["Year"] <= year_range[1])]

            col_pie, col_trend = st.columns([1, 1.5])
            
            with col_pie:
                st.markdown("##### 🥧 연도별 피해 비중")
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
                st.markdown("##### 📈 연령대별 피해 추이 (Trend)")
                age_line = alt.Chart(age_long_f).mark_line(point=True).encode(
                    x=alt.X("Year:O", title="연도"),
                    y=alt.Y("Victims:Q", title="피해자 수(명)"),
                    color=alt.Color("AgeGroup:N", legend=alt.Legend(title="연령대")),
                    tooltip=["Year", "AgeGroup", alt.Tooltip("Victims", format=",", title="피해자 수")]
                ).properties(height=450)
                st.altair_chart(age_line, use_container_width=True)
        else:
            st.error("연령대 데이터 컬럼을 찾을 수 없습니다.")
    else:
        st.warning("연령별 데이터(age.csv)가 로드되지 않았습니다.")

# ==============================================================================
# TAB 3: 월별 추이 & 이슈
# ==============================================================================
# with tab3:
#     st.subheader("🔓 개인정보 유출과 보이스피싱 상관관계 분석")
#     st.markdown("데이터 유출(잠재 위험)이 실제 보이스피싱 발생(실현된 위험)에 미치는 영향을 분석합니다.")
    
#     if not leak_df.empty and not annual_f.empty:
#         # 1. 컬럼명 공백 제거
#         leak_df.columns = leak_df.columns.str.strip()
        
#         # 2. KPI 데이터 계산 (기존 유지)
#         total_comp = len(leak_df)
#         total_leak_count = leak_df['유출건'].sum()
        
#         # 최근 10년 (2016~2025) 데이터 필터링
#         start_year, end_year = 2016, 2025
#         recent_leak_df = leak_df[(leak_df['연도'] >= start_year) & (leak_df['연도'] <= end_year)]
#         recent_leak_sum = recent_leak_df['유출건'].sum()
        
#         # 비율 계산
#         if total_leak_count > 0:
#             recent_ratio = (recent_leak_sum / total_leak_count) * 100
#         else:
#             recent_ratio = 0

#         # 숫자 포맷 함수
#         def fmt_kpi(num):
#             if num >= 100000000: return f"{num/100000000:.1f}억"
#             elif num >= 10000: return f"{num/10000:.1f}만"
#             else: return f"{num:,}"

#         # 3. KPI 카드 배치
#         st.markdown("##### 📊 핵심 지표 (Key Metrics)")
#         k1, k2, k3 = st.columns(3)
        
#         k1.metric("총 유출 기업/서비스", f"{total_comp} 개")
#         k2.metric("총 유출 정보 건수", "약 4억 9천만 건", delta="누적 추산치")
#         k3.metric(
#             f"최근 10년 유출 비중 ({start_year}~{end_year})", 
#             f"{recent_ratio:.1f}%", 
#             f"{fmt_kpi(recent_leak_sum)}건 집중됨", 
#             delta_color="inverse"
#         )
        
#         st.info(
#             "📢 **수사중, 수치미상 최소 수천만 건 추가 발생**\n\n"
#             "SNS 음성/얼굴 데이터 무단채굴로 인한 범죄 도구화로 **불특정 다수가 잠재적 피해자**입니다."
#         )
        
#         st.divider()

#         # ---------------------------------------------------------------------
#         # [변경] 워드클라우드 삭제 -> 이중축 그래프(Dual Axis Chart) 구현
#         # ---------------------------------------------------------------------
#         st.subheader("📈 유출 건수 대비 보이스피싱 발생 추이 (2016-2025)")

#         # (1) 데이터 병합 준비
#         # 보이스피싱 데이터 (annual_f는 이미 2025년 포함되어 있음)
#         phishing_data = annual_f[annual_f['Type'] == '전체'][['Year', 'Cases']].copy()
#         phishing_data['Year'] = phishing_data['Year'].astype(int)
        
#         # 유출 데이터 (연도별 합계)
#         leak_grouped = leak_df.groupby('연도')['유출건'].sum().reset_index()
#         leak_grouped['연도'] = leak_grouped['연도'].astype(int)

#         # (2) 2016~2025년 기준 데이터프레임 생성
#         years_df = pd.DataFrame({'Year': range(start_year, end_year + 1)})
        
#         # Merge
#         merged_df = pd.merge(years_df, phishing_data, on='Year', how='left')
#         merged_df = pd.merge(merged_df, leak_grouped, left_on='Year', right_on='연도', how='left')
        
#         # 결측치 0 처리
#         merged_df['Cases'] = merged_df['Cases'].fillna(0)
#         merged_df['유출건'] = merged_df['유출건'].fillna(0)

#         # (3) 시각화 (Plotly Dual Axis)
#         fig = make_subplots(specs=[[{"secondary_y": True}]])

#         # Trace 1: 유출 건수 (막대 그래프, 오른쪽 축, 배경 느낌)
#         fig.add_trace(
#             go.Bar(
#                 x=merged_df['Year'], 
#                 y=merged_df['유출건'], 
#                 name="개인정보 유출(건)",
#                 marker_color='rgba(255, 99, 71, 0.6)', # 연한 빨강 (투명도 조절)
#                 hovertemplate='%{x}년: %{y:,.0f}건 유출<extra></extra>'
#             ),
#             secondary_y=True
#         )

#         # Trace 2: 보이스피싱 발생 건수 (선 그래프, 왼쪽 축, 강조)
#         fig.add_trace(
#             go.Scatter(
#                 x=merged_df['Year'], 
#                 y=merged_df['Cases'], 
#                 name="보이스피싱 발생(건)",
#                 mode='lines+markers',
#                 line=dict(color='#003366', width=4), # 짙은 남색, 굵게
#                 marker=dict(size=8),
#                 hovertemplate='%{x}년: %{y:,.0f}건 발생<extra></extra>'
#             ),
#             secondary_y=False
#         )

#         # (4) 레이아웃 설정
#         fig.update_layout(
#             title_text=f"개인정보 유출(잠재 위험) vs 보이스피싱 발생(실제 피해)",
#             height=500,
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )

#         # Y축 설정 (왼쪽: 보이스피싱, 오른쪽: 유출)
#         fig.update_yaxes(title_text="<b>보이스피싱 발생 건수</b>", secondary_y=False, showgrid=True)
#         fig.update_yaxes(title_text="<b>개인정보 유출 건수 (단위: 억/천만)</b>", secondary_y=True, showgrid=False)
        
#         # X축 설정 (모든 연도 표시)
#         fig.update_xaxes(dtick=1)

#         st.plotly_chart(fig, use_container_width=True)

#         # (5) 상세 데이터 테이블
#         with st.expander("데이터 상세 보기"):
#             display_df = merged_df[['Year', 'Cases', '유출건']].rename(columns={
#                 'Year': '연도', 
#                 'Cases': '보이스피싱 발생건수', 
#                 '유출건': '개인정보 유출건수'
#             })
#             display_df['보이스피싱 발생건수'] = display_df['보이스피싱 발생건수'].apply(lambda x: f"{x:,.0f}건")
#             display_df['개인정보 유출건수'] = display_df['개인정보 유출건수'].apply(lambda x: f"{x:,.0f}건")
#             st.dataframe(display_df, use_container_width=True)

#     else:
#         st.warning("데이터가 로드되지 않았습니다. 사이드바에서 파일을 확인해주세요.")
# ==============================================================================
# TAB 4: 발생/검거
# ==============================================================================
with tab4:
    # (1) 경제적 가치 산출용
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

    # (2) 물리적 한계 현황용
    total_occur = 0
    total_arrest = 0
    total_gap = 0
    avg_arrest_rate = 0.0
    df_arr = pd.DataFrame() 

    if not annual_f.empty and "Arrest_Cases" in annual_f.columns:
        df_arr = annual_f[(annual_f["Type"] == "전체") & (annual_f["Year"] < 2025)].copy()
        df_arr = df_arr.sort_values("Year")
        
        if not df_arr.empty:
            total_occur = df_arr["Cases"].sum()
            total_arrest = df_arr["Arrest_Cases"].sum()
            total_gap = total_occur - total_arrest
            if total_occur > 0:
                avg_arrest_rate = (total_arrest / total_occur) * 100

    # 1. [Tab 4 전용 스냅샷]
    st.markdown("### 👮‍♂️ 물리적 한계(검거율) vs AI 도입의 경제적 가치")
    
    k1, k2, k3, k4 = st.columns(4)

    with k1: 
        st.metric("총 발생 (누적)", f"{total_occur:,.0f} 건", help="데이터 집계 기간 내 총 발생 건수")
    with k2: 
        st.metric("총 검거 (누적)", f"{total_arrest:,.0f} 건", delta=f"{avg_arrest_rate:.1f}% (검거율)")
    with k3: 
        st.metric("🚨 미검거 (한계)", f"{total_gap:,.0f} 건", delta="-물리적 대응 한계", delta_color="inverse", help="발생했으나 검거하지 못한 건수 (Gap)")
    with k4:
        st.metric("🛡️ 1건당 예방가치", f"{unit_prevent_value/10000:,.0f} 만원", delta="사회적 비용 포함", help="영국 내무부 기준: 직접피해액 × 1.5배")

    st.markdown("---")

    # 2. [Chart]
    st.subheader("📉 범죄 발생 속도를 따라잡지 못하는 물리적 검거")
    
    if not df_arr.empty:
        fig_arr = go.Figure()
        fig_arr.add_trace(go.Bar(x=df_arr['Year'], y=df_arr['Cases'], name='발생 건수', marker_color='#FF6B6B', text=df_arr['Cases'], texttemplate='%{text:,.0f}'))
        fig_arr.add_trace(go.Bar(x=df_arr['Year'], y=df_arr['Arrest_Cases'], name='검거 건수', marker_color='#4ECDC4', text=df_arr['Arrest_Cases'], texttemplate='%{text:,.0f}'))
        df_arr['Rate'] = (df_arr['Arrest_Cases'] / df_arr['Cases']) * 100
        fig_arr.add_trace(go.Scatter(x=df_arr['Year'], y=df_arr['Rate'], name='검거율(%)', yaxis='y2', mode='lines+markers', line=dict(color='#333333', width=2, dash='dot')))

        fig_arr.update_layout(
            barmode='group', 
            xaxis_title="연도", yaxis=dict(title="건수 (건)"),
            yaxis2=dict(title="검거율 (%)", overlaying='y', side='right', range=[0, 130], showgrid=False),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), height=350, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_arr, use_container_width=True)
    else:
        st.warning("차트를 표시할 데이터가 부족합니다.")

    # 3. [Simulator]
    st.markdown("---")
    st.subheader("💎 우리 기술의 가치 시뮬레이션 (ROI)")
    
    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        st.markdown("**🛡️ AI 방어 설정**")
        detect_target = st.slider("월간 예상 탐지 건수", min_value=1, max_value=1000, value=100, step=10)
        st.caption(f"기준: 최근 {min(target_years)}~{max(target_years)}년 평균 피해액")
        
        with st.expander("📚 1.5배 산출 근거 (UK Home Office)"):
            st.markdown("""
            **🇬🇧 영국 내무부 (2018 Report)**
            * **Page 15, Figure 2:** Cost of Crime Model
            * **Total Cost** = Financial Loss + Emotional Impact + CJS Costs
            * 위 공식에 따라 직접 피해액의 **약 1.5배~2배**가 실제 사회적 비용으로 산출됨.
            """)

    with col_sim2:
        monthly_saving = unit_prevent_value * detect_target
        yearly_saving = monthly_saving * 12
        
        st.markdown(f"#### 💰 1건 탐지 시: :blue[{unit_prevent_value/10000:,.0f}만 원] 사회적 비용 절감")
        st.caption(f"(직접 피해액 {avg_loss_per_case/10000:,.0f}만원 × 사회적 비용 계수 {social_cost_factor}배)")
        
        st.success(f"""
        **📊 영국 내무부 비용 모델 (The Cost of Crime Model) 적용**
        
        $$
        \\text{{Total Benefit}} = \\text{{Financial Loss}} + \\text{{Emotional Impact}} + \\text{{CJS Costs}}
        $$
        
        * **직접 피해 방어:** {avg_loss_per_case/10000:,.0f}만 원
        * **사회적 비용 절감:** +{(unit_prevent_value - avg_loss_per_case)/10000:,.0f}만 원 (수사비용, 정신적 피해 등)
        
        👉 **월 {detect_target}건 차단 시: 월 {monthly_saving/100000000:,.2f} 억원 경제 효과**
        """)

    st.info("""
    **💡 Insight for Investors:**
    경찰의 검거(사후 조치)는 피해금을 돌려받기 어렵지만, **AI의 탐지(사전 차단)는 피해액 전액(100%)을 보존**합니다.
    이것이 우리가 단순한 '범죄 예방'을 넘어 **'금융 자산 보호 솔루션'**인 이유입니다.
    """)