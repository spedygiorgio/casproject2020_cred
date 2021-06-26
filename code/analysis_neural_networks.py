#%% import config all
from config_all import *
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format='%(process)d-%(levelname)s-%(message)s')
dl_logger = logging.getLogger(__name__)

predictors = categorical_preds + continuous_preds
random.seed(mystate)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
nepochs = 1000
#%%tensorflow keras
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, concatenate, BatchNormalization, Activation, Dropout, Add, add, Lambda
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Zeros, Constant
from tensorflow import feature_column
from tensorflow.keras import regularizers
# %% load data
dtypes_dict = {'claims':int, 'zone_id':str}
df_mkt=pd.read_csv(os.path.join(dataprocessed_dir, 'mkt_anonymized_data.csv'), sep=";", decimal=".", dtype=dtypes_dict)
df_cpn=pd.read_csv(os.path.join(dataprocessed_dir, 'cpn_anonymized_data.csv'), sep=";", decimal=".", dtype=dtypes_dict)
train_avg_freq = np.sum(df_mkt.claims)/np.sum(df_mkt.exposure)

#%% model saving
weights_mkt_file = os.path.join(modelsdir,'deeplearning', 'market_model_weights.h5')
weights_cpn_file = os.path.join(modelsdir,'deeplearning', 'company_model_weights.h5')
weights_trf_file = os.path.join(modelsdir,'deeplearning', 'transfer_model_weights.h5')
network_graph = os.path.join(modelsdir,'deeplearning', 'network_model_graph.json')
network_graph_company = os.path.join(modelsdir,'deeplearning', 'network_model_graph.json')
#%% definint the size of embedding layers
embedding_layer_size = {}
for variable in categorical_preds:
    distinct_categories = len(list(df_mkt[variable].unique()))
    embedding_dimension = int(max(np.floor(distinct_categories**0.333),1))
    embedding_layer_size[variable] = embedding_dimension
    #print(f"{variable}, distinct categories {distinct_categories}, embedding dimension {embedding_dimension}")

#%% helper function to create TF datasetss
def df_to_tfds(dataframe, predictors, target, batch_size, buffer_size = None, shuffle=True ):
    """Create TF datasets

    Args:
        dataframe (pd.DataFrame): Pandas datasets
        predictors ([type]): predictors in the same sequence of inputs
        target ([type]): [description]
        batch_size ([type]): [description]
        buffer_size ([type], optional): [description]. Defaults to None.
        shuffle (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    df = dataframe.copy()
    labels = df.pop(target)
    df = df[predictors]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size= buffer_size)
    ds = ds.batch(batch_size)
    return ds


# %% model definition (inputs)
def get_deep_learning_model(exposure_col='exposure'):
    """Defines the DL model for frequency modeling

    Args:
        exposure_col (str, optional): the exposure column name. Defaults to 'exposure'.

    Returns:
        TF Model: A tensorflow model
    """    
    ## categorical data
    #categoricals columns
    feature_columns = [] # list of encoded
    feature_layer_inputs = {} #input dicts

    #allocate space for categorical inputs (encoded as integers)
    for header in categorical_preds :
        feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name= header, dtype=tf.int16)

    #allocate space for each categorical column (adding just the embedded features)
    ## cat1
    cat1_input = feature_column.categorical_column_with_vocabulary_list( key='cat1', vocabulary_list=list(df_mkt.cat1.unique()) ) 
    cat1_embedding = feature_column.embedding_column(cat1_input, dimension = embedding_layer_size['cat1'])
    feature_columns.append(cat1_embedding)
    ## cat2
    cat2_input = feature_column.categorical_column_with_vocabulary_list( key='cat2', vocabulary_list=list(df_mkt.cat2.unique()) ) 
    cat2_embedding = feature_column.embedding_column(cat2_input, dimension = embedding_layer_size['cat2'])
    feature_columns.append(cat2_embedding)
    ## cat3
    cat3_input = feature_column.categorical_column_with_vocabulary_list( key='cat3', vocabulary_list=list(df_mkt.cat3.unique()) ) 
    cat3_embedding = feature_column.embedding_column(cat3_input, dimension = embedding_layer_size['cat3'])
    feature_columns.append(cat3_embedding)
    ## cat4
    cat4_input = feature_column.categorical_column_with_vocabulary_list( key='cat4', vocabulary_list=list(df_mkt.cat4.unique()) ) 
    cat4_embedding = feature_column.embedding_column(cat4_input, dimension = embedding_layer_size['cat4'])
    feature_columns.append(cat4_embedding)
    ## cat5
    cat5_input = feature_column.categorical_column_with_vocabulary_list( key='cat5', vocabulary_list=list(df_mkt.cat5.unique()) ) 
    cat5_embedding = feature_column.embedding_column(cat5_input, dimension = embedding_layer_size['cat5'])
    feature_columns.append(cat5_embedding)
    ## cat6
    cat6_input = feature_column.categorical_column_with_vocabulary_list( key='cat6', vocabulary_list=list(df_mkt.cat6.unique()) ) 
    cat6_embedding = feature_column.embedding_column(cat6_input, dimension = embedding_layer_size['cat6'])
    feature_columns.append(cat6_embedding)
    ## cat7
    cat7_input = feature_column.categorical_column_with_vocabulary_list( key='cat7', vocabulary_list=list(df_mkt.cat7.unique()) ) 
    cat7_embedding = feature_column.embedding_column(cat7_input, dimension = embedding_layer_size['cat7'])
    feature_columns.append(cat7_embedding)
    ## cat8
    cat8_input = feature_column.categorical_column_with_vocabulary_list( key='cat8', vocabulary_list=list(df_mkt.cat8.unique()) ) 
    cat8_embedding = feature_column.embedding_column(cat8_input, dimension = embedding_layer_size['cat8'])
    feature_columns.append(cat8_embedding)


    #numerical features
    for variable in continuous_preds :
        feature_columns.append(feature_column.numeric_column(variable))
        feature_layer_inputs[variable] = tf.keras.Input(shape=(1,), name=variable)

    #creo il feature layers e gli input layer
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    input_layer = feature_layer(feature_layer_inputs)

    # offset modeling

    ## defining column for exposure
    feature_columns2 = [] 
    feature_layer_inputs2 = {}

    feature_columns2.append(feature_column.numeric_column(exposure_col))
    input_tensor_exposure = Input(shape=(1,), name=exposure_col) 
    feature_layer_inputs2[exposure_col] = input_tensor_exposure

    ## function for taking log
    def log_function(x):
        return K.log(x)

    feature_layer2 = tf.keras.layers.DenseFeatures(feature_columns2)
    input_layer2 = feature_layer2(feature_layer_inputs2)
    input_tensor_log_exposure = Lambda(log_function)(input_layer2)



    # %% model definition (core structure)

    dropout_ratio = 0.2
    l2_coeff = 0.01
    graph = Dense(256, activation='relu', name='dense_1', kernel_regularizer=regularizers.l2(l2_coeff))(input_layer) 
    graph = Dropout(dropout_ratio)(graph)
    graph = Dense(512, activation='relu', name='dense_2', kernel_regularizer=regularizers.l2(l2_coeff))(graph) 
    graph = Dropout(dropout_ratio)(graph)
    graph = Dense(256, activation='relu', name='dense_3', kernel_regularizer=regularizers.l2(l2_coeff))(graph) 
    graph = Dropout(dropout_ratio)(graph)


    frequency_network = Dense(units=1, activation='linear', name='frequency_network', trainable=True,
        kernel_initializer=Zeros(), bias_initializer=Constant(value= np.log(train_avg_freq) ) )(graph)
    #offset modeling
    add_layer = Add()([frequency_network, input_tensor_log_exposure])
    frequency_output = Dense(1, activation= K.exp, name= 'frequency_out', trainable=False, bias_initializer=Zeros(),kernel_initializer=Constant(value=1))(add_layer)

    #%% model compiling
    dl_model = Model(inputs = [v for v in feature_layer_inputs.values()] + [v for v in feature_layer_inputs2.values()], outputs=frequency_output)
    #dl_model.summary()
    return dl_model

#%% data retrieval
#common pars
xvars = categorical_preds + continuous_preds + ['exposure']
my_batch_size = 1024
my_buffer_size = 20000



#%% create the tensorflow datasets fpr the market data
df_mkt_train = df_mkt[df_mkt.group=='train'].copy()
df_mkt_valid = df_mkt[df_mkt.group=='valid'].copy()
df_mkt_test = df_mkt[df_mkt.group=='test'].copy()

df_mkt_tf_train = df_to_tfds(dataframe=df_mkt_train, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=True)
df_mkt_tf_valid = df_to_tfds(dataframe=df_mkt_valid, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=False)
df_mkt_tf_test = df_to_tfds(dataframe=df_mkt_test, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=False)

#%% create the tensorflow datasets for the company data
df_cpn_train = df_cpn[df_cpn.group=='train'].copy()
df_cpn_valid = df_cpn[df_cpn.group=='valid'].copy()
df_cpn_test = df_cpn[df_cpn.group=='test'].copy()

df_cpn_tf_train = df_to_tfds(dataframe=df_cpn_train, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=True)
df_cpn_tf_valid = df_to_tfds(dataframe=df_cpn_valid, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=False)
df_cpn_tf_test = df_to_tfds(dataframe=df_cpn_test, predictors=xvars,target='claims',batch_size=my_batch_size, buffer_size=my_buffer_size, shuffle=False)


#%% creating the makret model
def fitting_models(train_mkt=True, save_mkt=True, train_cpn=True, save_cpn=True, train_trf=True, save_trf=True):
    
    # for feature_batch, label_batch in df_mkt_tf_valid.take(1):
    #     print('Every feature:', list(feature_batch.keys()))
    #     print('A batch of cat2:', feature_batch['cat2'])
    #     print('A batch of targets:', label_batch )

    #%% finding the best learning late
    #lr_schedule = LearningRateScheduler( lambda epoch : 1e-8 * 10 ** (epoch / 20))
    # optimizer = Adam(lr=1e-8, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
    # market_dl_model.compile(optimizer=optimizer, loss={'frequency_out': 'poisson'})
    # history = market_dl_model.fit(df_mkt_tf_train, epochs = 100, callbacks = [lr_schedule] )
    # lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    # plt.semilogx(lrs, history.history["loss"])
    # plt.axis([1e-8, 1e-3,-11000, -14500])
    #lr_hat = 1e-05
    #%% running the market model
    if train_mkt:
        inizio = time.time()
        market_dl_model = get_deep_learning_model(exposure_col='exposure')
        lr_hat = 1e-05
        optimizer = Adam(lr=lr_hat, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
        early_stopping_mkt = EarlyStopping(monitor='val_loss', min_delta=0, patience = 50, verbose=1, mode='auto', restore_best_weights= True)
        shrinking_mkt = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 30, verbose=1, mode='auto')
        checkpointing_mkt = ModelCheckpoint(weights_mkt_file, monitor='val_loss', verbose=1, save_best_only=True)
        market_dl_model.compile(optimizer=optimizer, loss={'frequency_out': 'poisson'})
        history_mtk_model = market_dl_model.fit(df_mkt_tf_train, validation_data = df_mkt_tf_valid, epochs = nepochs, callbacks = [early_stopping_mkt, shrinking_mkt,checkpointing_mkt] )
        durata = time.time() - inizio 
        dl_logger.info(f"ho impiegato {durata:.2f} secondi per addestrare la rete sui dati di mercato...")
        if save_mkt:
            # %% saving graph and weights
            # save graph 
            model_json = market_dl_model.to_json()
            with open(network_graph, "w") as json_file:
                json_file.write(model_json)
            market_dl_model.save_weights(weights_mkt_file)

    #%% reload model from weight
    market_dl_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    market_dl_model_reloaded.load_weights(weights_mkt_file)

    #%% performing predictions
    ## on MTK data
    df_mkt_valid['claims_pred_dl_mkt'] =  market_dl_model_reloaded.predict(df_mkt_tf_valid)
    df_mkt_test['claims_pred_dl_mkt'] =  market_dl_model_reloaded.predict(df_mkt_tf_test)
    ## on CPN data
    df_cpn_valid['claims_pred_dl_mkt'] =  market_dl_model_reloaded.predict(df_cpn_tf_valid)
    df_cpn_test['claims_pred_dl_mkt'] =  market_dl_model_reloaded.predict(df_cpn_tf_test)


    #%%performance assessment on MTK
    ap_vld, gini_vld = analyze_results(y_true=df_mkt_valid['claims'], y_pred=df_mkt_valid['claims_pred_dl_mkt'], exposure=df_mkt_valid['exposure'], verbose=False)
    ap_tst, gini_tst = analyze_results(y_true=df_mkt_test['claims'], y_pred=df_mkt_test['claims_pred_dl_mkt'], exposure=df_mkt_test['exposure'], verbose=False)
    dl_logger.info(f"MKT model, MKT valid data: A/P ratio {ap_vld:.3f}, Gini {gini_vld:.3f}...")
    dl_logger.info(f"MKT model, MKT test data: A/P ratio {ap_tst:.3f}, Gini {gini_tst:.3f}...")


    #%%performance assessment on CPN
    ap_vld, gini_vld = analyze_results(y_true=df_cpn_valid['claims'], y_pred=df_cpn_valid['claims_pred_dl_mkt'], exposure=df_cpn_valid['exposure'], verbose=False)
    ap_tst, gini_tst = analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_mkt'], exposure=df_cpn_test['exposure'], verbose=False)
    dl_logger.info(f"MKT model, CPN valid data: A/P ratio {ap_vld:.3f}, Gini {gini_vld:.3f}...")
    dl_logger.info(f"MKT model, CPN test data: A/P ratio {ap_tst:.3f}, Gini {gini_tst:.3f}...")



    # %% create the company model
    if train_cpn:
        company_dl_model = get_deep_learning_model(exposure_col='exposure')
        # %% training the model on full company data
        lr_hat = 1e-05
        optimizer_cpn = Adam(lr=lr_hat, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
        early_stopping_cpn = EarlyStopping(monitor='val_loss', min_delta=0, patience = 50, verbose=1, mode='auto', restore_best_weights= True)
        shrinking_cpn = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 30, verbose=1, mode='auto')
        checkpointing_cpn = ModelCheckpoint(weights_cpn_file, monitor='val_loss', verbose=1, save_best_only=True)
        company_dl_model.compile(optimizer=optimizer_cpn, loss={'frequency_out': 'poisson'})
        history_cpn_model = company_dl_model.fit(df_cpn_tf_train, validation_data = df_cpn_tf_valid, epochs = nepochs, callbacks = [early_stopping_cpn, shrinking_cpn,checkpointing_cpn] )
        #%% save model company
        if save_cpn:
            company_dl_model.save_weights(weights_cpn_file)

    #%% reload model
    company_dl_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    company_dl_model_reloaded.load_weights(weights_cpn_file)
    # %% applying predictions
    df_cpn_valid['claims_pred_dl_cpn'] =  company_dl_model_reloaded.predict(df_cpn_tf_valid)
    df_cpn_test['claims_pred_dl_cpn']=  company_dl_model_reloaded.predict(df_cpn_tf_test)
    
    ap_vld, gini_vld = analyze_results(y_true=df_cpn_valid['claims'], y_pred=df_cpn_valid['claims_pred_dl_cpn'], exposure=df_cpn_valid['exposure'])
    ap_tst, gini_tst =analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_cpn'], exposure=df_cpn_test['exposure'])
    dl_logger.info(f"CPN model, CPN valid data: A/P ratio {ap_vld:.3f}, Gini {gini_vld:.3f}...")
    dl_logger.info(f"CPN model, CPN test data: A/P ratio {ap_tst:.3f}, Gini {gini_tst:.3f}...")

    # %% transfer learning model
    if train_trf:
        company_trf_model = get_deep_learning_model(exposure_col='exposure')
        company_trf_model.load_weights(weights_mkt_file) #initializing with market weights
        lr_hat = 1e-05
        optimizer_trf = Adam(lr=lr_hat, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
        early_stopping_trf = EarlyStopping(monitor='val_loss', min_delta=0, patience = 50, verbose=1, mode='auto', restore_best_weights= True)
        shrinking_trf = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 30, verbose=1, mode='auto')
        checkpointing_trf = ModelCheckpoint(weights_trf_file, monitor='val_loss', verbose=1, save_best_only=True)
        company_trf_model.compile(optimizer=optimizer_trf, loss={'frequency_out': 'poisson'})
        history_trf_model = company_trf_model.fit(df_cpn_tf_train, validation_data = df_cpn_tf_valid, epochs = nepochs, callbacks = [early_stopping_trf, shrinking_trf,checkpointing_trf] )
        if save_trf:
        #%% save transnfer learning modle
            company_trf_model.save_weights(weights_trf_file)
    #%% reload model
    company_trf_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    company_trf_model_reloaded.load_weights(weights_trf_file)
    # %% applying predictions
    df_cpn_valid['claims_pred_dl_trf'] =  company_trf_model_reloaded.predict(df_cpn_tf_valid)
    df_cpn_test['claims_pred_dl_trf']=  company_trf_model_reloaded.predict(df_cpn_tf_test)
    
    ap_vld, gini_vld = analyze_results(y_true=df_cpn_valid['claims'], y_pred=df_cpn_valid['claims_pred_dl_trf'], exposure=df_cpn_valid['exposure'], verbose=False)
    ap_tst, gini_tst =analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_trf'], exposure=df_cpn_test['exposure'], verbose=False)
    dl_logger.info(f"TRF model, CPN valid data: A/P ratio {ap_vld:.3f}, Gini {gini_vld:.3f}...")
    dl_logger.info(f"TRF model, CPN test data: A/P ratio {ap_tst:.3f}, Gini {gini_tst:.3f}...")

#%% analyze results 
def scoring_dl_cpn_test():
    #%% models reloading
    # market
    dl_logger.info(f'reloading DL models...')
    market_dl_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    market_dl_model_reloaded.load_weights(weights_mkt_file)
    # company own
    company_dl_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    company_dl_model_reloaded.load_weights(weights_cpn_file)
    # company with TL
    company_trf_model_reloaded = get_deep_learning_model(exposure_col='exposure')
    company_trf_model_reloaded.load_weights(weights_trf_file) 
    #predict on test
    dl_logger.info(f'scoring DL models...')
    df_cpn_test['claims_pred_dl_trf'] =  company_trf_model_reloaded.predict(df_cpn_tf_test)
    df_cpn_test['claims_pred_dl_cpn'] =  company_dl_model_reloaded.predict(df_cpn_tf_test)
    df_cpn_test['claims_pred_dl_mkt'] =  market_dl_model_reloaded.predict(df_cpn_tf_test)    
    dl_logger.info(f'assessing DL models...')
    ap_mkt, gini_mkt=analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_mkt'], exposure=df_cpn_test['exposure'], verbose=False)
    ap_cpn, gini_cpn=analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_cpn'], exposure=df_cpn_test['exposure'], verbose=False)
    ap_trf, gini_trf=analyze_results(y_true=df_cpn_test['claims'], y_pred=df_cpn_test['claims_pred_dl_trf'], exposure=df_cpn_test['exposure'], verbose=False)

    dl_logger.info(f'Company test dataset: MKT model A/P ratio {ap_mkt:.3f}, Gini {gini_mkt:.3f}')
    dl_logger.info(f'Company test dataset: CPN model A/P ratio {ap_cpn:.3f}, Gini {gini_cpn:.3f}')
    dl_logger.info(f'Company test dataset: TRF model A/P ratio {ap_trf:.3f}, Gini {gini_trf:.3f}')

    vars2keep = ['ID','exposure','claims','claims_pred_dl_trf','claims_pred_dl_cpn','claims_pred_dl_mkt']
    df_dl_cpn_out = df_cpn_test[vars2keep]
    df_dl_cpn_out.to_csv(os.path.join(outs_dir,'deeplearning_cpn_results.csv'),index=False, sep=";")


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