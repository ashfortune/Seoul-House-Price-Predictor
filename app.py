import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

# 한글 폰트 설정 (Mac 환경 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_path):
    """
    datasets 폴더 내의 서울시 부동산 실거래가 CSV 파일들을 로드하고 통합합니다.
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # 컬럼 정의 (서울시 공공데이터 기준)
    columns = [
        '접수연도', '자치구코드', '자치구명', '법정동코드', '법정동명', 
        '지번구분', '지번구분명', '본번', '부번', '건물명', 
        '계약일', '물건금액(만원)', '건물면적(㎡)', '토지면적(㎡)', '층', 
        '권리구분', '취득유형', '건축년도', '건물용도', '신고구분', '신고기관'
    ]
    
    df_list = []
    for file in csv_files:
        print(f"Loading: {os.path.basename(file)}")
        try:
            temp_df = pd.read_csv(file, encoding='cp949', names=columns, header=None)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 데이터 전처리
    combined_df['계약일'] = pd.to_datetime(combined_df['계약일'], format='%Y%m%d', errors='coerce')
    
    numeric_cols = ['물건금액(만원)', '건물면적(㎡)', '토지면적(㎡)', '층', '건축년도']
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    return combined_df

def perform_eda(df):
    """
    기초 탐색적 데이터 분석(EDA)을 수행합니다.
    """
    if df.empty:
        print("데이터가 비어있습니다.")
        return
        
    plt.figure(figsize=(10, 8))
    numeric_df = df[['물건금액(만원)', '건물면적(㎡)', '층', '건축년도']]
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('수치형 변수 간 상관관계 히트맵')
    plt.tight_layout()
    plt.savefig('report/correlation_heatmap.png')
    plt.close()
        
    print("\n--- [데이터 요약 정보] ---")
    print(df.info())
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='자치구명', order=df['자치구명'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('서울시 자치구별 부동산 거래 현황')
    plt.tight_layout()
    plt.savefig('report/transaction_count_by_district.png')
    plt.close()

def preprocess_data(df):
    """
    모델 학습에 적합하도록 데이터를 정제합니다. 
    """
    features = ['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']
    target = '물건금액(만원)'
    all_needed_cols = features + [target]
    
    df = df.dropna(subset=all_needed_cols)
    df = df[(df['건축년도'] > 0) & (df['물건금액(만원)'] > 0) & (df['건물면적(㎡)'] > 0)]
    
    return df[all_needed_cols]

def train_linear_regression(df):
    """
    선형 회귀 모델을 학습하고 평가합니다.
    """
    X = df[['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']]
    y = df['물건금액(만원)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['자치구명', '건물용도']
    numeric_features = ['건물면적(㎡)', '층', '건축년도']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    print("Linear Regression 모델 학습 중...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- [Linear Regression 모델 평가 결과] ---")
    print(f"평균 절대 오차 (MAE): {mae:,.2f} 만원")
    print(f"결정계수 (R² Score): {r2:.4f}")
    
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (만원)')
    plt.ylabel('Predicted Price (만원)')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('report/actual_vs_predicted.png')
    plt.close()
    
    return model

def train_xgboost(df):
    """
    XGBoost 모델을 학습하고 평가합니다.
    """
    X = df[['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']]
    y = df['물건금액(만원)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['자치구명', '건물용도']
    numeric_features = ['건물면적(㎡)', '층', '건축년도']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 튜닝 전 기본 모델 생성 (평가 지표 미리 설정)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, eval_metric='rmse'))
    ])
    
    # 하이퍼파라미터 튜닝을 위한 그리드 설정
    param_grid = {
        'regressor__n_estimators': [100, 300, 500],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.05, 0.1]
    }
    
    print("최적의 하이퍼파라미터를 찾는 중 (GridSearchCV)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    print(f"최적 파라미터: {grid_search.best_params_}")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- [XGBoost 모델 평가 결과] ---")
    print(f"평균 절대 오차 (MAE): {mae:,.2f} 만원")
    print(f"결정계수 (R² Score): {r2:.4f}")
    
    # 실제값 vs 예측값 시각화
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#FF4949') 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
    plt.xlabel('Actual Price (만원)', fontsize=12)
    plt.ylabel('Predicted Price (만원)', fontsize=12)
    plt.title(f'Tuned XGBoost: Actual vs Predicted\n(R²: {r2:.4f}, MAE: {mae:,.0f}만원)', fontsize=15)
    plt.tight_layout()
    plt.savefig('report/actual_vs_predicted_xgb.png')
    plt.close()

    # Feature Importance 시각화
    xgb_model = model.named_steps['regressor']
    ohe_columns = list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
    feature_names = numeric_features + ohe_columns
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="magma", hue=[feature_names[i] for i in indices], legend=False)
    plt.title('AI 모델이 학습 시 주목한 주요 변수 (XGBoost Top 10)', fontsize=15)
    plt.xlabel('상대적 중요도', fontsize=12)
    plt.tight_layout()
    plt.savefig('report/feature_importance.png')
    plt.close()

    # [NEW] 학습 이력(Learning Curve) 시각화를 위한 재학습 로직
    print("대표님, 학습 횟수에 따른 오차 변화를 분석하기 위해 정밀 재학습을 수행합니다.")
    best_xgb = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    
    # 학습 이력 확인을 위한 검증 데이터 분할
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train_sub_processed = preprocessor.transform(X_train_sub)
    X_val_processed = preprocessor.transform(X_val)
    
    # 버전 호환성을 위해 eval_metric은 생성자에서 지정된 것을 사용하거나 생략 가능
    best_xgb.fit(
        X_train_sub_processed, y_train_sub,
        eval_set=[(X_train_sub_processed, y_train_sub), (X_val_processed, y_val)],
        verbose=False
    )
    
    results = best_xgb.evals_result()
    # 지표 이름은 'rmse' 또는 'rmse'가 포함된 키값일 수 있음
    metric_key = list(results['validation_0'].keys())[0]
    epochs = len(results['validation_0'][metric_key])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0'][metric_key], label='Train (학습 데이터)', color='#2E86C1')
    plt.plot(x_axis, results['validation_1'][metric_key], label='Validation (검증 데이터)', color='#E74C3C')
    plt.legend()
    plt.ylabel(f'{metric_key.upper()} (부동산 가격 오차)', fontsize=12)
    plt.xlabel('학습 반복 횟수 (Iteration)', fontsize=12)
    plt.title('AI 모델 학습 이력 분석 (XGBoost Training History)', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('report/training_history.png')
    plt.close()
    print("학습 이력 차트(report/training_history.png)가 성공적으로 저장되었습니다.")

    return model

def perform_detailed_analysis(model, df):
    """
    순열 중요도 분석 및 상세 상관관계
    """
    X = df[['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']]
    y = df['물건금액(만원)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 상세 상관관계 히트맵
    plt.figure(figsize=(10, 8))
    numeric_cols = ['물건금액(만원)', '건물면적(㎡)', '층', '건축년도']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='RdYlBu_r', fmt=".2f")
    plt.title('부동산 가격 결정 요인별 상관관계 (Heatmap)', fontsize=15)
    plt.tight_layout()
    plt.savefig('report/correlation_heatmap_detailed.png')
    plt.close()
    
    # 2. Permutation Importance
    print("\nPermutation Importance 계산 중...")
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=result.importances_mean[sorted_idx], y=X.columns[sorted_idx], palette="viridis", hue=X.columns[sorted_idx], legend=False)
    plt.title("부동산 특징별 실질적 가격 영향력 (Test Set 분석)", fontsize=15)
    plt.xlabel("상대적 중요도 (정확도 기여도)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('report/permutation_importance.png')
    plt.close()
    
    print("정밀 분석 리포트 생성이 완료되었습니다.")
    
def save_model_comparison(lr_r2, xgb_r2):
    """
    선형 회귀와 XGBoost의 성능을 시각적으로 비교합니다.
    """
    models = ['Linear Regression', 'Tuned XGBoost']
    r2_scores = [lr_r2, xgb_r2]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, r2_scores, color=['#A9A9A9', '#FF4B4B'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.title('모델 성능 비교 (R² Score)', fontsize=15, pad=20)
    plt.ylabel('R² Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.axhline(y=xgb_r2, color='#FF4B4B', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/model_comparison_r2.png')
    plt.close()

if __name__ == "__main__":
    DATA_DIR = "datasets"
    raw_data = load_data(DATA_DIR)
    
    if not raw_data.empty:
        if not os.path.exists('report'):
            os.makedirs('report')
            
        df_cleaned = preprocess_data(raw_data)
        print(f"정제 후 데이터 수: {len(df_cleaned)}")
        
        perform_eda(df_cleaned)
        
        print("\n[STEP 1] Linear Regression 학습 시작")
        lr_model = train_linear_regression(df_cleaned)
        X_lr = df_cleaned.drop('물건금액(만원)', axis=1)
        y_lr = df_cleaned['물건금액(만원)']
        _, X_test_lr, _, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
        lr_r2 = r2_score(y_test_lr, lr_model.predict(X_test_lr))
        
        print("\n[STEP 2] XGBoost 학습 및 정밀 분석 시작")
        model_xgb = train_xgboost(df_cleaned)
        
        X_xgb = df_cleaned[['자치구명', '건물면적(㎡)', '층', '건축년도', '건물용도']]
        y_xgb = df_cleaned['물건금액(만원)']
        _, X_test_xgb, _, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
        xgb_r2 = r2_score(y_test_xgb, model_xgb.predict(X_test_xgb))
        
        perform_detailed_analysis(model_xgb, df_cleaned)
        save_model_comparison(lr_r2, xgb_r2)
        
        print("\n[STEP 3] 분석 결과 저장 중...")
        joblib.dump(model_xgb, 'xgb_model.pkl')
        joblib.dump(df_cleaned, 'df_cleaned.pkl')
        print("최적화된 XGBoost 모델과 정제 데이터가 .pkl 파일로 저장되었습니다.")
        
        print("\n모든 작업이 완료되었습니다.")
    else:
        print("데이터 로드 실패")
