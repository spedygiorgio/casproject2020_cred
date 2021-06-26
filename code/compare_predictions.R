require(tidyverse)
require(MLmetrics)
file_out = "./output/models_comparisons.RData"

df_dl <- data.table::fread(input = "./data/models_outputs/deeplearning_cpn_results.csv",data.table = F)
df_lgbm <- data.table::fread(input = "./data/models_outputs/boosting_cpn_results.csv",data.table = F) %>% select(ID, starts_with("claims_"))

#creating predictions df

## putting toghether the models using the ID column
df_predictions <- inner_join(x=df_dl, df_lgbm, by='ID')

## applying the scores
get_scores <- function(actual, predicted, label='') {
  normalized_gini <- MLmetrics::NormalizedGini(y_pred = predicted, y_true=actual)
  actual_predicted_ratio <- sum(actual)/sum(predicted)
  out <- list(label=label, normalized_gini=normalized_gini, actual_predicted_ratio=actual_predicted_ratio)
  return(out)
}

# deep learning scores
dl_mtk_scores<-with(df_predictions,get_scores(claims, claims_pred_dl_mkt,label='dl_mkt'))
dl_cpn_scores<-with(df_predictions,get_scores(claims, claims_pred_dl_cpn,label='dl_cpn'))
dl_trf_scores<-with(df_predictions,get_scores(claims, claims_pred_dl_trf,label='dl_trf'))

# lightgbm scores
bst_mtk_scores<-with(df_predictions,get_scores(claims, claims_pred_bst_mkt,label='bst_mkt'))
bst_cpn_scores<-with(df_predictions,get_scores(claims, claims_pred_bst_cpn,label='bst_cpn'))
bst_trf_scores<-with(df_predictions,get_scores(claims, claims_pred_bst_trf,label='bst_trf'))

ml_results <- as.data.frame(data.table::rbindlist(list(dl_mtk_scores,dl_cpn_scores,dl_trf_scores,bst_mtk_scores,bst_cpn_scores,bst_trf_scores)))

ml_results <- separate(ml_results, col='label',into=c('model','approach')) 

save(list = c('ml_results'),file=file_out)