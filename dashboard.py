import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance

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
        st.write("알고리즘이 어떤 요소를 중요하게 생각하는지 실시간 분석 결과입니다.")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # 1. Feature Importance (XGBoost) - 실시간 생성
            st.write("#### [모델 학습 중요도]")
            try:
                # 파이프라인에서 모델과 전처리기 추출
                xgb_model = model.named_steps['regressor']
                preprocessor = model.named_steps['preprocessor']
                
                # 피처 이름 복원
                cat_features = ['자치구명', '건물용도']
                num_features = ['건물면적(㎡)', '층', '건축년도']
                ohe_feature_names = list(preprocessor.transformers_[1][1].get_feature_names_out(cat_features))
                all_feature_names = num_features + ohe_feature_names
                
                # 중요도 데이터프레임 생성
                importances = xgb_model.feature_importances_
                feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=feat_df, x='Importance', y='Feature', palette="rocket", ax=ax)
                ax.set_title("XGBoost 주요 학습 변수 (Top 15)")
                st.pyplot(fig)
            except Exception as e:
                st.info("💡 변수 중요도를 계산할 수 없습니다. (모델 구조 확인 필요)")

        with col_b:
            # 2. Permutation Importance - 실무적 영향력 (샘플링 사용)
            st.write("#### [실질적 가격 영향력]")
            if st.button("실질적 영향력 정밀 분석 실행 (약 5초 소요)"):
                with st.spinner("데이터 기여도를 분석 중입니다..."):
                    # 계산 집약적이므로 일부 샘플만 사용
                    sample_df = df.sample(min(2000, len(df)), random_state=42)
                    X_sample = sample_df[['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']]
                    y_sample = sample_df['물건금액(만원)']
                    
                    perm_result = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
                    
                    p_feat_df = pd.DataFrame({'Feature': X_sample.columns, 'Importance': perm_result.importances_mean})
                    p_feat_df = p_feat_df.sort_values(by='Importance', ascending=True)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=p_feat_df, x='Importance', y='Feature', palette="viridis", ax=ax2)
                    ax2.set_title("부동산 특징별 실질 영향도")
                    st.pyplot(fig2)
            else:
                st.info("👆 버튼을 누르면 실시간으로 데이터 기여도를 분석합니다.")

    with tab2:
        st.write(f"### 📊 자치구별 부동산 거래 현황")
        st.write(f"서울 전체 지역 중 {selected_district}의 거래 규모를 확인해 보세요.")
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='자치구명', order=df['자치구명'].value_counts().index, ax=ax3, palette="muted")
        plt.xticks(rotation=45)
        ax3.set_title('서울시 자치구별 거래 건수 비교')
        st.pyplot(fig3)
        st.info("💡 **분석 결과:** 거래 건수가 많을수록 시장 데이터가 풍부하여 예측 정확도가 더욱 높아지는 경향이 있습니다.")

    with tab3:
        st.write("### 🌡️ 변수 간 상관관계 정밀 분석")
        st.write("각 데이터 항목들이 가격 결정에 미치는 수치적 관계를 보여줍니다.")
        
        # 수치형 데이터 상관관계 계산
        numeric_df = df[['물건금액(만원)', '건물면적(㎡)', '층', '건축년도']]
        corr_matrix = numeric_df.corr()
        
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', fmt=".2f", ax=ax4, linewidths=0.5)
        ax4.set_title('부동산 가격 결정 요인별 상관관계 히트맵')
        st.pyplot(fig4)
        
        st.info("💡 **히트맵 분석:** 1.0에 가까울수록 강한 정비례(붉은색)를 의미합니다. '물건금액'과 '건물면적'의 관계를 눈여겨보세요.")


if __name__ == "__main__":
    main()
