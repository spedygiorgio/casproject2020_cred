#%% import config all
from config_all import *
import pandas as pd
import numpy as np
import logging
from argparse import ArgumentParser
logging.basicConfig(level=logging.INFO,format='%(process)d-%(levelname)s-%(message)s')
lgbm_logger = logging.getLogger(__name__)
predictors = categorical_preds + continuous_preds

#%% general config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import seaborn as sns
import hyperopt.pyll.stochastic
from hyperopt import hp, tpe, fmin
from hyperopt.pyll.stochastic import sample as sample_pars
random.seed(mystate)

#%% load data
dtypes_dict = {'claims':int, 'zone_id':str}
df_mkt=pd.read_csv(os.path.join(dataprocessed_dir, 'mkt_anonymized_data.csv'), sep=";", decimal=".", dtype=dtypes_dict)
df_cpn=pd.read_csv(os.path.join(dataprocessed_dir, 'cpn_anonymized_data.csv'), sep=";", decimal=".", dtype=dtypes_dict)
#%% files
mkt_model_file = os.path.join(modelsdir,'lightgbm','mkt_lgb_model.txt')
cpn_model_file = os.path.join(modelsdir,'lightgbm','cpn_lgb_model.txt')
trf_model_file = os.path.join(modelsdir,'lightgbm','trf_lgb_model.txt')
#%% create lightgbm dataset 
def create_lgb_dataset(df, init_score_col='log_exposure'):
    out = lgb.Dataset(data = df[predictors], 
                    label=df['claims'], 
                    feature_name = df[predictors].columns.tolist(),
                    categorical_feature= categorical_preds,
                    init_score= df[init_score_col],
                    free_raw_data=False)
    return out

#%% creating the market datasets
ds_lgb_mkt_train = create_lgb_dataset(df_mkt[df_mkt.group=='train'])
ds_lgb_mkt_valid = create_lgb_dataset(df_mkt[df_mkt.group=='valid'])
ds_lgb_mkt_test = create_lgb_dataset(df_mkt[df_mkt.group=='test'])

#%% creating the company datasets
ds_lgb_cpn_train = create_lgb_dataset(df_cpn[df_cpn.group=='train'])
ds_lgb_cpn_valid = create_lgb_dataset(df_cpn[df_cpn.group=='valid'])
ds_lgb_cpn_test = create_lgb_dataset(df_cpn[df_cpn.group=='test'])

#%% function for optimizing lgbm
def adjust_parameter_space(param_dict, to_cast_as_int, fixed_params):
    """adjust the format of some parameters

    Arguments:
        param_dict {[type]} -- dictionary of parameters
        to_cast_as_int {[type]} -- which shall be casted at int
        fixed_params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    for key in to_cast_as_int:
        param_dict[key] = int(param_dict[key])
        # add fixed parameters
    param_dict = {**param_dict, **fixed_params}
    return param_dict

# %% lightgbm optimizer 

lgb_k_fold = 10
lgb_early_stopping_rouds = 200
def lgb_model_optimizer(lgb_train, max_iterazioni, fixed_params, predictors, 
    predictors_categorical, lgb_parameter_space, to_cast_as_int, lgb_valid=None):
    """Optimizer lightgbm

    Arguments:
        lgb_train {lgb dataset} -- trainset
        max_iterazioni {int} -- max number of iterations
        fixed_params {dict} -- dictionary of fixed parameters
        predictors {list} -- list of predictors
        predictors_categorical {list} -- list of categoricals predictors (contained in predictors)
        lgb_parameter_space {hyperoptlist} -- hyperopt list of parameters
        to_cast_as_int {dict} -- parameters to be casted as integer

    Keyword Arguments:
        lgb_valid {lgb dataset} -- lightgbm dataset (default: {None})
    """    


    def lgb_objective(params, fixed_params=fixed_params, verbose=True):
        """Internal objective function for optimizing the poisson model

        Arguments:
            params {dict} -- params to run in iter

        Keyword Arguments:
            fixed_params {dict} -- fixed parameters to be optimized (default: {fixed_params})
            verbose {bool} -- log printing (default: {True})

        Returns:
            error -- the error of prediction
        """        


        # extract parameters from the nested domain based on the boosting type

        params = adjust_parameter_space(params, to_cast_as_int, fixed_params = fixed_params)
        if params['objective'] == 'poisson' :
            error_name = 'poisson-mean'
            metric = 'poisson'
        else :
            error_name = 'tweedie-mean'
            metric = 'tweedie'

        try:
            if lgb_valid is None:
                cv_result = lgb.cv(
                    params = params,
                    train_set = lgb_train,
                    metrics = metric,
                    categorical_feature = predictors_categorical,
                    nfold = lgb_k_fold,
                    stratified = False,
                    early_stopping_rounds = lgb_early_stopping_rouds
                )
                error = cv_result[error_name][-1]
            else:
                lgb_train_result = lgb.train(
                    params = params,
                    train_set = lgb_train,
                    #metrics = metric,
                    categorical_feature=predictors_categorical,
                    valid_sets=[lgb_valid],
                    early_stopping_rounds=lgb_early_stopping_rouds,
                    verbose_eval=False
                )
                error= lgb_train_result.best_score['valid_0'][metric]
        except:
            error = np.infty

        finally:
            lgb_objective.i += 1
        if verbose:
            print("Info: iteration {} error {:.4f}".format(lgb_objective.i, error))
        return error

    # iteration count initialized to 0
    lgb_objective.i = 0

    # function to find the best model parameters
    print('Start tuning with hyperopt, with {} iterations\n'.format(max_iterazioni))
    bestpars = fmin(fn=lgb_objective, space=lgb_parameter_space, algo=tpe.suggest, max_evals=max_iterazioni)
    print('End tuning, found best hyper-parameters \n')
    bestpars = adjust_parameter_space(bestpars, to_cast_as_int, fixed_params = fixed_params)
    return bestpars 

#%% hyperparameter configuration 

to_cast_as_int=['bagging_freq','num_leaves', 
'min_child_samples', 'max_depth', 'max_cat_to_onehot', 
'cat_smooth', 'cat_l2', 'max_cat_threshold', 'min_data_per_group' ]

fixed_params = {
    'num_boost_round': 4000,
    'verbose': 0,
    'seed': mystate,
    'num_threads': 0,
    'bagging_seed' : mystate,
    'objective': 'poisson'
    #for gpu training
    ,'device':'cpu'
#    ,'gpu_platform_id':0
    ,'max_bin' : 63
#    ,'gpu_device_id': 0
}

lgb_parameter_space = {
    'max_depth' : hp.quniform('max_depth', 3, 20, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.1)),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 10), #teoricamente 2^max_depth ma per valori grandi overfitta facilmente
    'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.001, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.0001), 1.5),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.0001), 1.5),
    'min_child_samples': hp.quniform('min_child_samples', 5, 100, 5),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.25, 1.0),
    'min_data_per_group' : hp.quniform('min_data_per_group', 50, 200, 50),
    'max_cat_threshold' : hp.quniform('max_cat_threshold', 16, 64, 16),
    'cat_l2' : hp.quniform('cat_l2', 10, 30, 10),
    'cat_smooth' : hp.quniform('cat_smooth', 10, 30, 10),
    'max_cat_to_onehot' : hp.quniform('max_cat_to_onehot', 2, 10, 2),   
    'bagging_freq' : hp.quniform('bagging_freq', 0, 300, 50)
}

#%% find the best hyperparameters
def main_lgb_model(argomenti):

    hyperopt_iters = argomenti.iterations
    save_models = argomenti.save_models

    start = time.time()

    
    best_params_lgb_mkt = lgb_model_optimizer(lgb_train=ds_lgb_mkt_train, 
    max_iterazioni = hyperopt_iters, fixed_params = fixed_params, predictors = predictors,
    predictors_categorical = categorical_preds, 
    lgb_parameter_space = lgb_parameter_space,
    to_cast_as_int = to_cast_as_int,lgb_valid=ds_lgb_mkt_valid)

    end = time.time()-start
    print(f"it took {end:.3f} to run a {hyperopt_iters}-cyle of hyperopt optimizations...")

    #%% retraining best model

    best_model_lgb_mkt =lgb_train_result = lgb.train(
                        params = best_params_lgb_mkt,
                        train_set = ds_lgb_mkt_train,
                        #metrics = metric,
                        categorical_feature=categorical_preds,
                        valid_sets=[ds_lgb_mkt_valid],
                        early_stopping_rounds=lgb_early_stopping_rouds,
                        verbose_eval=500
                    )
    if save_models is True:
        best_model_lgb_mkt.save_model(mkt_model_file)
    # %% fitting on validation
    temp_xvars = df_mkt[df_mkt.group=='valid'][predictors]
    temp_init_score = df_mkt[df_mkt.group=='valid']['log_exposure']
    temp_raw_score = best_model_lgb_mkt.predict(temp_xvars, raw_score=True)
    claims_pred = np.exp(temp_raw_score+temp_init_score)
    claims_true = df_mkt[df_mkt.group=='valid']['claims']
    exposure_valid = df_mkt[df_mkt.group=='valid']['exposure']
    analyze_results(y_pred=claims_pred, y_true=claims_true,exposure=exposure_valid)

    # %% fitting on test
    temp_xvars = df_mkt[df_mkt.group=='test'][predictors]
    temp_init_score = df_mkt[df_mkt.group=='test']['log_exposure']
    temp_raw_score = best_model_lgb_mkt.predict(temp_xvars, raw_score=True)
    claims_pred = np.exp(temp_raw_score+temp_init_score)
    claims_true = df_mkt[df_mkt.group=='test']['claims']
    exposure_test = df_mkt[df_mkt.group=='test']['exposure']
    analyze_results(y_pred=claims_pred, y_true=claims_true, exposure=exposure_test)


    # %% now trying to train a model only on cpn data
    start = time.time()
    cpn_hyperopt_iters = hyperopt_iters
    best_params_lgb_cpn = lgb_model_optimizer(lgb_train=ds_lgb_cpn_train, 
    max_iterazioni = cpn_hyperopt_iters, fixed_params = fixed_params, 
    predictors = predictors,
    predictors_categorical = categorical_preds, 
    lgb_parameter_space = lgb_parameter_space,
    to_cast_as_int = to_cast_as_int,lgb_valid=ds_lgb_cpn_valid)

    best_model_lgb_cpn =lgb_train_result = lgb.train(
                        params = best_params_lgb_cpn,
                        train_set = ds_lgb_cpn_train,
                        categorical_feature=categorical_preds,
                        valid_sets=[ds_lgb_cpn_valid],
                        early_stopping_rounds=lgb_early_stopping_rouds,
                        verbose_eval=500
                    )
    end = time.time()-start
    print(f"it took {end:.3f} to run a {hyperopt_iters}-cyle of hyperopt optimizations...")
    if save_models is True:
        best_model_lgb_cpn.save_model(cpn_model_file)



    #%% now applying transfer learning: calculating a priori claim experience with market data
    best_model_lgb_mkt = lgb.Booster(model_file=mkt_model_file)
    ## applying market model on company data
    raw_base_scores = best_model_lgb_mkt.predict(df_cpn[predictors], raw_score=True)
    a_priori_log_estimates = raw_base_scores+df_cpn['log_exposure']
    df_cpn['a_priori_log_claims']=a_priori_log_estimates

    ds_lgb_cpn_trf_train = create_lgb_dataset(df_cpn[df_cpn.group=='train'], init_score_col='a_priori_log_claims')
    ds_lgb_cpn_trf_valid = create_lgb_dataset(df_cpn[df_cpn.group=='valid'], init_score_col='a_priori_log_claims')
    ds_lgb_cpn_trf_test = create_lgb_dataset(df_cpn[df_cpn.group=='test'], init_score_col='a_priori_log_claims')
    # %% apply transfer learning
    start = time.time()
    trf_hyperopt_iters = hyperopt_iters
    best_params_lgb_trf = lgb_model_optimizer(lgb_train=ds_lgb_cpn_trf_train, 
    max_iterazioni = trf_hyperopt_iters, fixed_params = fixed_params, predictors = predictors,
    predictors_categorical = categorical_preds, 
    lgb_parameter_space = lgb_parameter_space,
    to_cast_as_int = to_cast_as_int,lgb_valid=ds_lgb_cpn_trf_valid)

    best_model_lgb_trf = lgb.train(
                        params = best_params_lgb_trf,
                        train_set = ds_lgb_cpn_trf_train,
                        #metrics = metric,
                        categorical_feature=categorical_preds,
                        valid_sets=[ds_lgb_cpn_trf_valid],
                        early_stopping_rounds=lgb_early_stopping_rouds,
                        verbose_eval=500
                    )
    end = time.time()-start
    print(f"it took {end:.3f} to run a {hyperopt_iters}-cyle of hyperopt optimizations...")
    if save_models is True:
        best_model_lgb_trf.save_model(trf_model_file)


    # %% predict on company data (validation)
    temp_data = df_cpn[df_cpn.group=='valid']
    y_pred_cpn = np.exp(best_model_lgb_cpn.predict(temp_data[predictors], raw_score=True)+temp_data.log_exposure)
    y_pred_trf = np.exp(best_model_lgb_trf.predict(temp_data[predictors], raw_score=True)+temp_data.a_priori_log_claims)
    y_pred_mkt = np.exp(best_model_lgb_mkt.predict(temp_data[predictors], raw_score=True)+temp_data.log_exposure)
    y_true_cpn = temp_data.claims
    exposure_cnp_valid = temp_data.exposure

    analyze_results(y_true=y_true_cpn, y_pred=y_pred_cpn,exposure=exposure_cnp_valid)
    analyze_results(y_true=y_true_cpn, y_pred=y_pred_trf,exposure=exposure_cnp_valid)
    analyze_results(y_true=y_true_cpn, y_pred=y_pred_mkt,exposure=exposure_cnp_valid)



#%% scoring function
def scoring_lgbm_cpn_test():
    #reloading the models
    lgbm_logger.info('reloading LGBM models...')
    best_model_lgb_mkt = lgb.Booster(model_file=mkt_model_file)
    best_model_lgb_cpn = lgb.Booster(model_file=cpn_model_file) 
    best_model_lgb_trf = lgb.Booster(model_file=trf_model_file) 
    ## applying market model on company data (for transfer learning)
    lgbm_logger.info('scoring company data...')
    raw_base_scores = best_model_lgb_mkt.predict(df_cpn[predictors], raw_score=True)
    a_priori_log_estimates = raw_base_scores+df_cpn['log_exposure']
    df_cpn['a_priori_log_claims']=a_priori_log_estimates
    #%% exposure on test data (test)
    df_cpn_test = df_cpn[df_cpn.group=='test']

    y_pred_test_mkt = np.exp(best_model_lgb_mkt.predict(df_cpn_test[predictors], raw_score=True)+df_cpn_test.log_exposure) 
    y_pred_test_cpn = np.exp(best_model_lgb_cpn.predict(df_cpn_test[predictors], raw_score=True)+df_cpn_test.log_exposure)
    y_pred_test_trf = np.exp(best_model_lgb_trf.predict(df_cpn_test[predictors], raw_score=True)+df_cpn_test.a_priori_log_claims)
    
    y_true_cpn_test = df_cpn_test.claims
    exposure_cnp_test = df_cpn_test.exposure

    ap_mkt, gini_mkt=analyze_results(y_true=y_true_cpn_test, y_pred=y_pred_test_mkt,exposure=exposure_cnp_test)
    ap_cpn, gini_cpn=analyze_results(y_true=y_true_cpn_test, y_pred=y_pred_test_cpn,exposure=exposure_cnp_test)
    ap_trf, gini_trf=analyze_results(y_true=y_true_cpn_test, y_pred=y_pred_test_trf,exposure=exposure_cnp_test)

    lgbm_logger.info(f'Company test dataset: MKT model A/P ratio {ap_mkt:.3f}, Gini {gini_mkt:.3f}')
    lgbm_logger.info(f'Company test dataset: CPN model A/P ratio {ap_cpn:.3f}, Gini {gini_cpn:.3f}')
    lgbm_logger.info(f'Company test dataset: TRF model A/P ratio {ap_trf:.3f}, Gini {gini_trf:.3f}')

    df_cpn_test['claims_pred_bst_mkt'] = y_pred_test_mkt
    df_cpn_test['claims_pred_bst_cpn'] = y_pred_test_cpn
    df_cpn_test['claims_pred_bst_trf'] = y_pred_test_trf
    
    lgbm_logger.info('saving results on test set...')
    vars2keep = ['ID','exposure','claims','claims_pred_bst_mkt','claims_pred_bst_cpn','claims_pred_bst_trf']
    df_boost_out = df_cpn_test[vars2keep]
    df_boost_out.to_csv(os.path.join(outs_dir,'boosting_cpn_results.csv'),index=False, sep=";")
    





#%% main core
if __name__ == '__main__':
    my_parser = ArgumentParser()
    my_parser.add_argument("-i","--iterations",help="Number of iterations",type=int,default=100)
    my_parser.add_argument('--save', dest='save_models', action='store_true')
    my_parser.add_argument('--no-save', dest='save_models', action='store_false')
    my_parser.set_defaults(save_models=True)
    args = my_parser.parse_args()
    lgbm_logger.info(f"iterations {args.iterations}, save {args.save_models}")
    main_lgb_model(args)