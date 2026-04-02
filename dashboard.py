import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="서울 부동산 AI 예측 대시보드",
    page_icon="🏠",
    layout="wide"
)

# --- 한글 폰트 설정 (Mac 기준) ---
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 및 모델 로드 함수 ---
@st.cache_resource
def load_analysis_results():
    """저장된 모델과 데이터를 로드합니다."""
    if os.path.exists('xgb_model.pkl') and os.path.exists('df_cleaned.pkl'):
        model = joblib.load('xgb_model.pkl')
        df = joblib.load('df_cleaned.pkl')
        return model, df
    return None, None

# --- 메인 대시보드 로직 ---
def main():
    st.title("🏠 서울 부동산 실거래가 AI 예측 시스템")
    st.markdown("---")

    # 데이터 및 모델 로드
    model, df = load_analysis_results()
    
    if model is None or df is None:
        st.error("⚠️ 분석 결과 파일(.pkl)을 찾을 수 없습니다.")
        st.info("먼저 터미널에서 `python3 app.py`를 실행하여 모델 학습과 데이터 정제를 완료해 주세요.")
        st.stop()
    
    # --- 사이드바: 사용자 입력 창 ---
    st.sidebar.header("📊 예측 조건 설정")
    
    selected_district = st.sidebar.selectbox("자치구 선택", sorted(df['자치구명'].unique()))
    
    # 평수 -> ㎡ 변환기
    pyeong = st.sidebar.slider("면적 선택 (평)", 5, 100, 34)
    area = pyeong * 3.3058
    st.sidebar.info(f"선택한 면적: 약 {area:.2f} ㎡")
    
    floor = st.sidebar.slider("층수 선택", -2, 70, 10)
    
    build_year = st.sidebar.number_input("건축년도 (완공시기)", 1960, 2030, 2024)
    
    bldg_type = st.sidebar.selectbox("건물 용도 선택", sorted(df['건물용도'].unique()))
    
    # --- 메인 화면: 예측 결과 ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 AI 예측 결과")
        input_df = pd.DataFrame([{
            '자치구명': selected_district,
            '건물면적(㎡)': area,
            '층': floor,
            '건축년도': build_year,
            '건물용도': bldg_type
        }])
        
        prediction = model.predict(input_df)[0]
        
        # 큰 글씨로 강조
        st.success(f"### 예상 매매가는 약 **{prediction/10000:.1f} 억원** ({prediction:,.0f} 만원) 입니다.")
        st.info("💡 위 금액은 현재 시장 데이터 91%의 정확도를 바탕으로 산출된 통계적 예상치입니다.")

    with col2:
        st.subheader("📍 선택한 조건 요약")
        st.write(f"- **지역**: 서울특별시 {selected_district}")
        st.write(f"- **면적**: {pyeong}평 ({area:.1f}㎡)")
        st.write(f"- **상태**: {build_year}년 완공, {floor}층")
        st.write(f"- **구분**: {bldg_type}")

    st.markdown("---")

    # --- 분석 시각화 섹션 ---
    st.subheader("🔍 데이터 기반 정밀 분석")
    
    tab1, tab2, tab3 = st.tabs(["💡 가격 결정 핵심 요인", "📊 실거래 트렌드", "🌡️ 상관관계 분석"])
    
    with tab1:
        st.write("### 💡 AI 모델의 가격 판단 근거")
        st.write("알고리즘의 판단 기준과 실제 데이터 기여도, 그리고 모델 전체 성능을 분석합니다.")
        
        # 모델 성능 비교 차트 (새로 추가)
        if os.path.exists('report/model_comparison_r2.png'):
            st.image('report/model_comparison_r2.png', caption="정밀 튜닝 전(Linear) vs 후(XGBoost) 성능 비교", use_container_width=True)
            st.success(f"📈 **XGBoost 튜닝 결과:** 선형 회귀 모델 대비 예측 정확도가 비약적으로 상승했습니다.")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists('report/feature_importance.png'):
                st.image('report/feature_importance.png', caption="AI 알고리즘 판단 중요도 (XGBoost)")
            else:
                st.warning("변수 중요도 리포트를 생성하는 중입니다.")
        
        with c2:
            if os.path.exists('report/permutation_importance.png'):
                st.image('report/permutation_importance.png', caption="실제 데이터 기여도 분석 (Permutation)")
            else:
                st.info("💡 정밀 분석 데이터(Permutation)를 생성하려면 app.py를 실행해 주세요.")
        
        st.info("📈 **분석 가이드:** 왼쪽 그래프는 AI 모델이 학습 과정에서 주목한 순서이며, 오른쪽은 실제 예측 성능에 결정적 영향을 미치는 변수 순위입니다. 두 순위가 일치할수록 모델의 신뢰도가 높습니다.")


    with tab2:
        st.write(f"최근 서울 전역에서 **{selected_district}**의 위상은 어느 정도일까요?")
        if os.path.exists('report/transaction_count_by_district.png'):
            st.image('report/transaction_count_by_district.png', caption="서울 자치구별 거래 활성도 비교")
        else:
            st.warning("거래량 통계 이미지를 찾을 수 없습니다.")

    with tab3:
        st.write("### 🌡️ 변수 간 상관관계 정밀 분석")
        st.write("각 데이터 항목들이 가격 및 서로에게 어떤 영향을 주는지 보여줍니다.")
        
        target_heatmap = 'report/correlation_heatmap_detailed.png' if os.path.exists('report/correlation_heatmap_detailed.png') else 'report/correlation_heatmap.png'
        
        if os.path.exists(target_heatmap):
            st.image(target_heatmap, caption="부동산 가격 결정 요인별 상관관계 히트맵")
            st.info("💡 **히트맵 보는 법:** 빨간색에 가까울수록 정(+)의 관계(같이 상승), 파란색에 가까울수록 부(-)의 관계(반비례)를 의미합니다. '물건금액' 행을 확인해 보세요.")
        else:
            st.warning("상관관계 분석 리포트를 찾을 수 없습니다. app.py를 먼저 실행해 주세요.")


if __name__ == "__main__":
    main()
