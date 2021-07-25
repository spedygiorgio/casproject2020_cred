require(tidyverse)
require(MLmetrics)
file_out = "./output/models_comparisons.RData"

df_dl  <-  data.table::fread(input = "./data/models_outputs/deeplearning_cpn_results.csv",data.table = F)
df_lgbm  <-  data.table::fread(input = "./data/models_outputs/boosting_cpn_results.csv",data.table = F) %>% select(ID, starts_with("claims_"))

result_cred1  <-  read.csv("./data/models_outputs/credibility1_results.csv")
#summary(is.na(result_cred1))
#creating predictions df

## putting toghether the models using the ID column
df_predictions  <-  inner_join(x=df_dl, df_lgbm, by='ID')

sum(df_predictions$ID != result_cred1$ID)


## applying the scores
get_scores  <-  function(actual, predicted, label='') 
{
  nonNA <- !is.na(predicted)
  normalized_gini  <-  MLmetrics::NormalizedGini(y_pred = predicted[nonNA], y_true=actual[nonNA])
  MAE  <-  MLmetrics::MAE(y_pred = predicted[nonNA], y_true=actual[nonNA])
  MedianAE  <-  MLmetrics::MedianAE(y_pred = predicted[nonNA], y_true=actual[nonNA])
  RMSE  <-  MLmetrics::RMSE(y_pred = predicted[nonNA], y_true=actual[nonNA])
  actual_predicted_ratio  <-  sum(actual[nonNA])/sum(predicted[nonNA])
  mean_actual_predicted_ratio  <-  mean(actual[nonNA]/predicted[nonNA])
  out  <-  list(label=label, normalized_gini=normalized_gini, 
              actual_predicted_ratio=actual_predicted_ratio,
              mean_actual_pred_ratio=mean_actual_predicted_ratio,
              MeanAE = MAE, MedianAE=MedianAE, RMSE=RMSE)
  return(out)
}

# deep learning scores
dl_mtk_scores <- with(df_predictions,get_scores(claims, claims_pred_dl_mkt,label='dl_mkt'))
dl_cpn_scores <- with(df_predictions,get_scores(claims, claims_pred_dl_cpn,label='dl_cpn'))
dl_trf_scores <- with(df_predictions,get_scores(claims, claims_pred_dl_trf,label='dl_trf'))

# lightgbm scores
bst_mtk_scores <- with(df_predictions,get_scores(claims, claims_pred_bst_mkt,label='bst_mkt'))
bst_cpn_scores <- with(df_predictions,get_scores(claims, claims_pred_bst_cpn,label='bst_cpn'))
bst_trf_scores <- with(df_predictions,get_scores(claims, claims_pred_bst_trf,label='bst_trf'))

#credibility scores

cred1_cpn_scores <- get_scores(df_predictions$claims,result_cred1$claims_pred_cred1_cpn, label='cred1_cpn')
cred1_mkt_scores <- get_scores(df_predictions$claims,result_cred1$claims_pred_cred1_mkt, label='cred1_mkt')
cred1_full_scores <- get_scores(df_predictions$claims,result_cred1$claims_pred_cred1_full, label='cred1_trf')

reslist  <-  list(dl_mtk_scores, dl_cpn_scores, dl_trf_scores, bst_mtk_scores,
                bst_cpn_scores, bst_trf_scores, cred1_cpn_scores, cred1_mkt_scores, cred1_full_scores)

ml_results  <-  as.data.frame(data.table::rbindlist(reslist))

ml_results  <-  separate(ml_results, col='label',into=c('model','approach')) 

write.csv(ml_results, file = "./output/models_comparisons.csv")
save(list = c('ml_results'), file=file_out)
