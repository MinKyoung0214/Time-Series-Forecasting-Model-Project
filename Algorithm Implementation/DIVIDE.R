##
# 시드 설정 (재현성을 위해)
set.seed(2022)

# 시뮬레이션된 데이터에서 20개의 샘플을 무작위로 선택
id_sample <- sample(1:length(dataset_simulation_train), 20)

# 선택한 20개의 샘플로 학습 및 테스트 세트 구성
dataset_train_example <- dataset_simulation_train[id_sample]
dataset_test_example <- dataset_simulation_test[id_sample]

##Model training
#FIDE와 DIVIDE 예측 모델을 학습하고, 이를 위한 XGBoost 메타 모델을 훈련
#1. 간헐적 수요 예측 방법으로 예측 생성
for (i in 1:length(dataset_train_example)) {
  dataset_train_example[[i]] <- calculate_forec_point(dataset_train_example[[i]], h = 12, quantile = FALSE)
}

#2. 예측 오차 계산 (예: RMSSE)
dataset_train_example <- calc_errors_rmsse(dataset_train_example)

#3. FIDE: 간헐적 수요를 위한 시계열 피처 계산

for (i in 1:length(dataset_train_example)) {
  dataset_train_example[[i]]$features <- compute_ifeatures(dataset_train_example[[i]]$x)
}

#4. DIVIDE: 시계열의 다양성 벡터 계산
dataset_train_example <- compute_diversity(dataset_train_example)

#5. XGBoost 모델을 사용하여 최적의 조합 가중치 학습
library(M4metalearning)
set.seed(2022)

# 메타 학습을 위한 데이터 생성
train_data <- create_feat_classif_problem(dataset_train_example)

# XGBoost 메타 모델 학습
meta_model <- train_selection_ensemble(train_data$data, train_data$errors)



##Forecasting
# 1. 테스트 데이터에 대해 9개의 예측 방법을 사용해 예측 생성
for (i in 1:length(dataset_test_example)) {
  dataset_test_example[[i]] <- calculate_forec_point(dataset_test_example[[i]], h = 12, quantile = FALSE)
}

# 2. DIVIDE: 각 시계열에 대한 다양성 벡터 계산
dataset_test_example <- compute_diversity(dataset_test_example)

# 3. 사전 학습된 XGBoost 모델을 사용하여 조합 가중치 예측
final_data <- create_feat_classif_problem(dataset_test_example)
# final_data$data가 xgb.DMatrix가 아닌 경우 강제 변환
library(xgboost)

# final_data$data를 xgb.DMatrix로 변환
newdata <- xgb.DMatrix(as.matrix(final_data$data))

# 예측 수행
preds <- predict_selection_ensemble(meta_model, newdata)


# 4. 최적 가중치를 사용해 여러 예측 방법의 결과를 결합
dataset_test_example <- ensemble_forecast(preds, dataset_test_example)

# 5. RMSSE로 예측 성능 평가
predictions_res <- summary_performance_rmsse(dataset_test_example)


Average RMSSE: 0.6394137
##FIDE 방식보다 DIVIDE 방식이 이 데이터에 더 적합한 예측 성능
