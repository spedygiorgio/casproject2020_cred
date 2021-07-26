#%% import config all
from config_all import *
import pandas as pd
import numpy as np
import logging
from argparse import ArgumentParser
logging.basicConfig(level=logging.INFO,format='%(process)d-%(levelname)s-%(message)s')
lgbm_logger = logging.getLogger(__name__)
predictors = categorical_preds + continuous_preds
import lightgbm as lgb
#%% LGBM models
mkt_model_file = os.path.join(modelsdir,'lightgbm','mkt_lgb_model.txt')
cpn_model_file = os.path.join(modelsdir,'lightgbm','cpn_lgb_model.txt')
trf_model_file = os.path.join(modelsdir,'lightgbm','trf_lgb_model.txt')

#%% reloading modles
lgbm_MKT = lgb.Booster(model_file=mkt_model_file) 
lgbm_CPN = lgb.Booster(model_file=cpn_model_file) 
lgbm_TRF = lgb.Booster(model_file=trf_model_file) 

#%% LGBM MKT model
lgb.plot_importance(lgbm_MKT)
plt.title('LGBM MKT Model variable importance analysis')
plt.savefig(os.path.join(outs_dir,'lgbm_mkt_varimp.png'))
plt.show()
#%% LGBM CPN model
lgb.plot_importance(lgbm_CPN)
plt.title('LGBM CPN Model variable importance analysis')
plt.savefig(os.path.join(outs_dir,'lgbm_cpn_varimp.png'))
plt.show()


#%% LGBM TRF model
lgb.plot_importance(lgbm_TRF)
plt.title('LGBM TRF Model variable importance analysis')
plt.savefig(os.path.join(outs_dir,'lgbm_trf_varimp.png'))
plt.show()

# %%
