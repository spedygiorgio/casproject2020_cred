#%% load data
import pandas as pd
import numpy as np 
import os, sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config_all import *
#%% define logger
import logging
logging.basicConfig(level=logging.INFO,format='%(process)d-%(levelname)s-%(message)s')
prep_logger = logging.getLogger(__name__)
#%% load origina data 


dtypes_agriculture = { 'anno' : str,
             'cod_prod' : str,
             'cod_gar' : str,
             'cod_imp_protez': str,
             'compagnia' :str ,
             'codistat' : str, 
             'prodotto' : str,
             'istat_comune_montano': str,
             'istat_grado_urbanizzazione' : str,
             'ecoregione_divisione' : str,
             'ecoregione_provincia' : str,
             'ecoregione_sezione' : str,
             'istat_flag_litoraneita': str,
             'grandine_20_49': float,
             'grandine_50_69': float,
             'grandine_70_over': float,
             'qli_risarciti' : float,
             'qli_assic' : float       
}

#%% load data
mkt_df = pd.read_csv(os.path.join(dataraw_dir, 'data_joined_mkt.csv'), sep=";", encoding='latin-1', dtype=dtypes_agriculture)
cpn_df = pd.read_csv(os.path.join(dataraw_dir, 'data_joined_cpn.csv'), sep=";", encoding='latin-1', dtype=dtypes_agriculture)

#%% configuring predictors
predictors_continuous = ['densita_abitativa', 'reddito_imponibile_medio', 'enea_gradi_giorno', 
                         'altitudine_loc_abitata','temperature_massime', 'temperature_minime',
                        'vento_massimo', 'precipitazioni_massime', 'fulminazione', 'esai00', 
                        'grandine_20_49', 'grandine_50_69', 'grandine_70_over']

predictors_categorical = ['prodotto', 'cod_gar', 'cod_imp_protez', 'istat_comune_montano', 
                          'istat_grado_urbanizzazione', 'istat_flag_litoraneita','ecoregione_divisione', 'ecoregione_sezione']

# these are the new names
predictors_continuous_new = ['cont'+str(i+1) for i in range(len(predictors_continuous))]
predictors_categorical_new = ['cat'+str(i+1) for i in range(len(predictors_categorical))]


#%% split train and validation set

def train_valid(x):
    if x <= 0.8:
        return 'train'
    else: 
        return 'valid'

def recode_original_data(df, market_df=None):
    #load data
    prep_logger.info('basic renaming')
    df.drop(columns = ['compagnia'], inplace = True)
    rename_dict = {'qli_risarciti':'claims','qli_assic':'exposure','codistat':'zone_id', 'anno':'year'}
    df.rename(columns=rename_dict, inplace=True)
    #filter Nan, zero exposures and y > 1 and creating log-exposure and ids
    prep_logger.info('filtering NA, exposures, and indexing')
    df = df.dropna() #excluding NA
    df = df[df['exposure'] > 0] #considering only exposures positive
    df = df[df.claims/df.exposure <= 1] #excluding claim frequencies over 100% (theoretically impossible)
    df = df.reset_index().rename(columns = {'index' : 'ID'}) #creating a fake ID variables
    #transforming some variables
    prep_logger.info("shifting year")
    df['year']=df['year'].astype(int)-10
    prep_logger.info("doubling exposures") #for anonymization
    df['exposure'] = 2 * df['exposure']
    df['log_exposure'] = np.log(df['exposure'])
    #altri cambiamenti
    ## Encoding the variable cod_imp_protez
    prep_logger.info('encoding protection implant and ecoregione sezione')
    cod_dict = {x: x if x in ['000', '200'] else 'Others' for x in np.unique(df['cod_imp_protez'].values).tolist()}
    df['cod_imp_protez'].replace(cod_dict, inplace = True)
    ## Encoding the variable Ecoregione sezione
    ecoregione_sezione_dict_rename = {'Porzione Italiana della Provincia Illirica': 'Alpina' ,
                 'Porzione Italiana della Provincia Ligure Provenzale': 'Tirrenica',
                 'Sezione Adriatica Centrale': 'Adriatica Centrale',
                 'Sezione Adriatica Meridionale': 'Adriatica Meridionale',
                 'Sezione Alpina Centro-Orientale': 'Alpina',
                 'Sezione Alpina Occidentale': 'Alpina', 
                 'Sezione Appenninica Centrale': 'Appenninica Centrale',
                 'Sezione Appenninica Meridionale': 'Appenninica Meridionale',
                 'Sezione Appenninica Settentrionale e Nord-Occidentale' : 'Appenninica Settentrionale e Nord-Occidentale',
                 'Sezione Padana': 'Padana',
                 'Sezione Sarda': 'Sezione Sarda',
                 'Sezione Siciliana': 'Siciliana',
                 'Sezione Tirrenica centro-settentrionale': 'Tirrenica',
                 'Sezione Tirrenica meridionale': 'Tirrenica' }
    df['ecoregione_sezione'].replace(ecoregione_sezione_dict_rename, inplace = True)
    # throw categories not in market dataset
    if  market_df is not None:
        ## dropping categorie in cpn data and not in market
        s1 = df.shape[0]
        for feature in predictors_categorical : 
            filter_ = market_df[feature].unique()
            df = df[df[feature].isin(filter_)]
            s2 = df.shape[0]
            prep_logger.info('Feature {feature} Deleted Rows : {nr}'.format(nr = (s1 - s2), feature=feature))
    #adding group var for train validation and test set
    prep_logger.info('adding train/test group id')
    df_test, temp = df[df.year==2008], df[df.year!=2008]
    ## fixing the test
    df_test['group']='test'
    ## the remaining is split by training and validation
    rnds_ids = np.random.uniform(size=temp.shape[0])
    group_var = list(map(train_valid, rnds_ids))
    temp['group']=group_var
    df = pd.concat([temp, df_test],axis=0)

    #keeping only the right columns
    prep_logger.info("keeping only the right columns")
    vars2take = ['log_exposure', 'exposure','claims', 'ID', 'zone_id', 'year','group'] + predictors_continuous+predictors_categorical
    df = df[vars2take]
    return df

def renamer(df):
    prep_logger.info("renaming old predictors")
    renaming_dict_continuous = dict(zip(predictors_continuous, predictors_continuous_new))
    renaming_dict_categorical = dict(zip(predictors_categorical, predictors_categorical_new))
    rename_dict = {**renaming_dict_continuous, **renaming_dict_categorical}
    df.rename(columns=rename_dict, inplace=True)
    return df


#%% preprocessing
## initial stage
df_mkt = mkt_df.copy()
df_mkt = recode_original_data(df=df_mkt)
df_cpn = cpn_df.copy()
df_cpn = recode_original_data(df=df_cpn, market_df=df_mkt)
## add renaming
df_mkt = renamer(df=df_mkt)
df_cpn = renamer(df=df_cpn)
    

#%% standardizing
standardizers = {}
for variable in predictors_continuous_new:
    #checking missingess
    num_missing_mkt = df_mkt[variable].isnull().sum()
    num_missing_cpy = df_cpn[variable].isnull().sum()
    logging.info(f"{variable} predictor: there are {num_missing_mkt} missing in MKT and {num_missing_cpy} in CPN")
    #creating the scaler
    logging.info(f"creating the scaler for variable {variable} on market data")
    myscaler = StandardScaler()
    #calibrating the variable
    df_mkt[variable] = myscaler.fit_transform(df_mkt[variable].values.reshape(-1,1))
    logging.info(f"applying the scaler for variable {variable} on company data")
    #applying on testset
    df_cpn[variable] = myscaler.transform(df_cpn[variable].values.reshape(-1,1))
    #finishing
    standardizers[variable] = myscaler

# %% label encoding
labelencoders = {}
for variable in predictors_categorical_new+['zone_id']:
    #checking missingess
    num_missing_mkt = df_mkt[variable].isnull().sum()
    num_missing_cpy = df_cpn[variable].isnull().sum()
    logging.info(f"{variable} predictor: there are {num_missing_mkt} missing in MKT and {num_missing_cpy} in CPN")
    #creating the scaler
    logging.info(f"creating the LabelEncoder for variable {variable}")
    mylencoder = LabelEncoder()
    #calibrating the variable
    df_mkt[variable] = mylencoder.fit_transform(df_mkt[variable].values)
    #applying on testset
    df_cpn[variable] = mylencoder.transform(df_cpn[variable].values)
    #finishing
    labelencoders[variable] = mylencoder
    num_missing_cpy = df_cpn[variable].isnull().sum()
    logging.info(f"After LEncoding on {variable} there are {num_missing_cpy} missing in CPN")



# %% saving data
df_mkt.to_csv(os.path.join(dataprocessed_dir, 'mkt_anonymized_data.csv'), sep=";", decimal=".", index=False)
df_cpn.to_csv(os.path.join(dataprocessed_dir, 'cpn_anonymized_data.csv'), sep=";", decimal=".", index=False)


# %%
