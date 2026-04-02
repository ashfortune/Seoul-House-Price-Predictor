# 🏠 Seoul Real Estate AI Price Predictor (R² 0.9162)

서울시 부동산 실거래가 데이터를 기반으로 주택 매매액을 정밀하게 예측하는 인공지능 시스템입니다.

## 🚀 Key Achievements

*   **최첨단 예측 정확도**: 하이퍼파라미터 튜닝(GridSearchCV)을 통해 **결정계수(R²) 0.92** 달성.
*   **성능 최적화**: 분석 결과 및 모델을 **직렬화(Serialization, .pkl)**하여 대시보드 구동 속도를 **1초 이내**로 단축.
*   **인사이트 제공**: 변수 중요도(Feature Importance) 및 순열 중요도(Permutation Importance) 분석을 통한 가격 결정 요인 시각화.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **ML Engine**: XGBoost, Scikit-learn
*   **Dashboard**: Streamlit
*   **Analysis**: Pandas, Matplotlib, Seaborn

## 📈 Model Performance Comparison

| Model | R² Score | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| **Linear Regression** | 0.6498 | 31,062 만원 |
| **Tuned XGBoost (Final)** | **0.9162** | **13,283 만원** |

## 🕹️ Getting Started

### 1. 전제 조건 (Prerequisites)
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
```

### 2. 데이터 분석 및 모델 학습 (엔진 실행)
```bash
python3 app.py
```
- 이 과정에서 전체 학습 및 시각화 리포트가 생성되며, 결과가 `.pkl` 파일로 자동 저장됩니다.

### 3. 실시간 예측 대시보드 실행
```bash
streamlit run dashboard.py
```

## 📁 Project Structure

- `app.py`: 정밀 데이터 분석 및 XGBoost 하이퍼파라미터 튜닝 메인 엔진.
- `dashboard.py`: 저장된 분석 결과를 활용한 초고속 예측 웹 인터페이스.
- `datasets/`: 서울시 부동산 실거래가 데이터 (2024-2026).
- `report/`: 데이터 기반 상관관계 및 변수 중요도 시각화 리포트.
- `*.pkl`: 최적화된 학습 모델 및 정제 데이터 스냅샷.

---
**Developed by [ashfortune]**
