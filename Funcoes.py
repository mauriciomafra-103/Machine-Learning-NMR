# Bibliotecas mais comuns
import pandas as pd
import numpy as np
import os, time
#import cv2 as cv
import random
import statistics
import openpyxl
import sys

# Bibliotecas para leitura e processamentos dos dados
import statsmodels.api as sm
import pickle


# Bibliotecas para criação de gráficos
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import randint

# Bibliotecas para o Projeto RMN
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, GammaRegressor, TheilSenRegressor, ElasticNet, HuberRegressor, BayesianRidge, PoissonRegressor
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score


# Implementação e treinamento da rede CNN
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import optim



random.seed(0)
## Abertura dos dados de Ressonância Magnética Nuclear 
def AbrirDados (diretorio_pasta, arquivo_niumag, arquivo_laboratorio, inicio_conversao, pontos_inversao,
                 Pasta_Salvamento = None, Salvar = False, Data = None, T2_niumag_gm = False,
                 T2_niumag_av = False, Fracoes_T2 = False, Dados_porosidade_Transverso = False):
    niumag = str(diretorio_pasta) + str(arquivo_niumag)
    laboratorio = str(diretorio_pasta) + str(arquivo_laboratorio)
    dados_niumag = pd.read_excel(niumag).drop('File Name', axis=1)
    dados_lab = pd.read_excel(laboratorio)

    inicio = inicio_conversao-2
    final = inicio+pontos_inversao


    amostras = []
    tempo_distribuicao = []
    distribuicao_t2 = []
    t2gm_niumag = []
    t2av_niumag = []
    area_distribuicao = []
    fitting_erro = []
    fracao_argila = []
    porosidade_i = []
    poço = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []

    for i in np.arange(int(len(dados_niumag.columns)/7)):
        df = dados_niumag.T.reset_index().drop('index', axis = 1).T
        nome = dados_niumag.columns[i*7][:9]
        tempo = df[i*7+3][inicio:final]
        distribuicao = df[i*7+4][inicio:final]
        gm = float(df[i*7+2][1][7:-4])
        av = float(df[i*7+2][2][7:-4])
        area = float(df[i*7][2][18:])
        fit_erro = float(df[i*7][0][-5:])
        argila = sum(distribuicao[:53])/sum(distribuicao)
        p = nome[:4]
        amostras.append(nome)
        poço.append(p)
        tempo_distribuicao.append(list(tempo))
        distribuicao_t2.append(list(distribuicao))
        t2gm_niumag.append(gm)
        t2av_niumag.append(av)
        area_distribuicao.append(area)
        fitting_erro.append(fit_erro)
        fracao_argila.append(argila)

    codi_lab = preprocessing.LabelEncoder()
    categoria_lito = codi_lab.fit_transform(dados_lab['Litofacies'])
    onehot = OneHotEncoder()
    ohe = pd.DataFrame(onehot.fit_transform(dados_lab[['Litofacies']]).toarray())
    ohe.columns = onehot.categories_


    for i in np.arange(len(distribuicao_t2)):
        t2_transpose = pd.DataFrame([distribuicao_t2[i]]).T
        scaler = pd.DataFrame(MaxAbsScaler().fit_transform(t2_transpose))
        scaler_sum_phi = float(dados_lab['Porosidade RMN'][i])/float(scaler.sum())
        phi_i = []
        for j in np.arange(len(scaler)):
            p = float(scaler[0][j]*scaler_sum_phi)
            phi_i.append(p)
        porosidade_i.append(list(phi_i))
    media_ponderada_log = []
    for i in np.arange(len(porosidade_i)):
        phi_i = porosidade_i[i]
        tempo_log = np.log(tempo_distribuicao[i])
        produto_porosidade_t2_log = pd.DataFrame(phi_i*tempo_log)
        sum_num = np.sum(produto_porosidade_t2_log)
        sum_den = np.sum(phi_i)
        razao_t2 = float(np.exp(sum_num/sum_den))
        media_ponderada_log.append((razao_t2))

    dados = pd.DataFrame({'Amostra': amostras})
    dados['Poço'] = poço
    dados['Litofacies'] = dados_lab['Litofacies']
    dados['Categoria Litofacies'] = categoria_lito
    dados['Bioturbiditos'] = ohe['Bioturbated']
    dados['Dolowackstone'] = ohe['Dolowackstone']
    dados['Grainstone'] = ohe['Grainstone']
    dados['Brechado'] = ohe['Brechado']
    dados['Tempo Distribuicao'] = pd.Series(tempo_distribuicao)
    dados['Distribuicao T2'] = pd.Series(distribuicao_t2)
    dados['Porosidade i'] = pd.Series(porosidade_i)
    dados['Porosidade Gas'] = dados_lab['Porosidade Gas']/100
    dados['Porosidade RMN'] = dados_lab['Porosidade RMN']/100
    dados['Permeabilidade Gas'] = dados_lab['Permeabilidade Gas']
    dados['Fracao Argila'] =  fracao_argila
    dados['Fitting Error'] = fitting_erro
    dados['T2 Ponderado Log'] = media_ponderada_log

    if T2_niumag_gm == True:
        dados['T2 Geometrico Niumag'] = t2gm_niumag

    if T2_niumag_av == True:
        dados['T2 Medio Niumag'] = t2av_niumag


    if Fracoes_T2 == True:
        for i in np.arange(len(porosidade_i)):
            phi_i = pd.Series(porosidade_i[i])
            porosidade = np.sum(porosidade_i[i])
            a1 = phi_i[:74].sum()
            a2 = phi_i[74:84].sum()
            a3 = phi_i[84:92].sum()
            a4 = phi_i[92:].sum()
            
            phimicro = float(a1/porosidade)           
            phimeso  = float(a2/porosidade)               
            phimacro = float(a3/porosidade)               
            phisuper = float(a4/porosidade)

            if phimicro <= 0.0001:
                phimicro = 0.0001
            if phimeso <= 0.0001:
                phimeso = 0.0001
            if phimacro <= 0.0001:
                phimacro = 0.0001
            if phisuper <= 0.0001:
                phisuper = 0.0001 
            
            s1.append(phimicro)
            s2.append(phimeso)
            s3.append(phimacro)
            s4.append(phisuper)
        
      
        
        dados['S1'] = s1
        dados['S2'] = s2
        dados['S3'] = s3
        dados['S4'] = s4

    if Dados_porosidade_Transverso == True:
        dataframe_porosidade = dados['Porosidade i']
        array_tempo_distribuicao = dados['Tempo Distribuicao']
        array_amostras = dados ['Amostra']
        df = pd.DataFrame([[0 for col in range(len(array_tempo_distribuicao[0]))] for row in range(len(array_amostras))])
        colunas = []
        for i in range(len(array_amostras)):
            for j in np.arange(len(array_tempo_distribuicao[0])):
                por = dataframe_porosidade[i][j]
                string = 'T2 ' + str(array_tempo_distribuicao[i][j])
                colunas.append(string)
                df[j][i] = por
        df.columns = colunas[0:128]
        dados = pd.concat([dados, df], axis = 1)

    if Salvar == True:
        local_salvamento = Pasta_Salvamento + arquivo_niumag[:10] + Data + '.xlsx'
        dados.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe

    return dados

# Dados para construção de RidgeLine
def DadosRidgeLine(dados, Pasta_salvamento, nome):


    colunas = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
           'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064', 
           'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
           'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
           'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
           'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',  
           'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494', 
           'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612', 
           'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
           'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
           'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
           'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
           'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
           'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
           'T2 8969.245',  'T2 10000']

    porosidade_i = []
    tempo_distribuicao = []
    for i in np.arange(len(dados)):
        phi_i = []
        tempo = []
        for j in np.arange(len(colunas)):
            phi_i.append(dados[colunas[j]][i])
            tempo.append(float(colunas[j][3:]))
        porosidade_i.append(phi_i)
        tempo_distribuicao.append(tempo)
    
    dados['Porosidade i'] = porosidade_i
    dados['Tempo Distribuicao'] = tempo_distribuicao
    
    lista_tempo = []
    lista_amostra = []
    lista_t2 = []
    lista_litofacie = []
    lista_poço = []
    for i in np.arange(len(dados)):
        for j in np.arange(len(dados['Tempo Distribuicao'][0])):
            lista_amostra.append(dados['Amostra'][i])
            lista_tempo.append(dados['Tempo Distribuicao'][i][j])
            lista_t2.append(dados['Porosidade i'][i][j])
            lista_litofacie.append(dados['Categoria Litofacies'][i])
            lista_poço.append(dados['Poço'][i])

    df = pd.DataFrame({'Amostra': lista_amostra,
                       'Poço': lista_poço,
                       'Tempo': lista_tempo,
                       'T2': lista_t2,
                       'Litofacie': lista_litofacie})
    local_salvamento = Pasta_salvamento + 'Dados_RidgeLine_' + str(nome) + '.xlsx'
    df.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe
    
    return df

## Escolha dos dados para aplicação no modelo SDR
def ProcessamentoDadosSDR (Dataframe, T2_escolhido):
    dados = pd.DataFrame({
        'Amostra': Dataframe['Amostra'],
        'Litofacies': Dataframe['Litofacies'],
        'T2': Dataframe[str(T2_escolhido)],
        'Porosidade RMN': Dataframe['Porosidade RMN'],
        'Porosidade Gas': Dataframe['Porosidade Gas'],
        'Permeabilidade Gas': Dataframe['Permeabilidade Gas']
    })
    return dados

def RegressaoSDR (Dataframe_SDR):
    # Regressão via OLS
    t2 = Dataframe_SDR['T2']
    phi = Dataframe_SDR['Porosidade RMN']
    permeabilidade = Dataframe_SDR['Permeabilidade Gas']
    dados_calculo = pd.DataFrame({'Log k': np.log(permeabilidade),
                                'Log φ': np.log(phi),
                                'Log T2': np.log(t2)})
    dados_calculo = sm.add_constant(dados_calculo)
    atributos = dados_calculo[['const', 'Log φ', 'Log T2']]
    rotulos = dados_calculo[['Log k']]
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'c', 'R2'],
        'Valor': [np.exp(reg_ols_log.params[0]),
                  reg_ols_log.params[1],
                  reg_ols_log.params[2],
                  reg_ols_log.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes['Valor']['a']
    b = coeficientes['Valor']['b']
    c = coeficientes['Valor']['c']
    k = (a*(phi**b)*(t2**c))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)

    return reg_ols_log, coeficientes, pd.concat([Dataframe_SDR, dados], axis = 1), sigma


## Escolha dos dados para aplicação no novo modelo SDR
def ProcessamentoDadosNovoSDR (Dataframe):
    dados = pd.DataFrame({'Amostra': Dataframe['Amostra'],
                          'Litofacies': Dataframe['Litofacies'],
                          'Permeabilidade Gas': Dataframe['Permeabilidade Gas'],
                          'Porosidade Gas': Dataframe['Porosidade Gas'],
                          'Porosidade RMN': Dataframe['Porosidade RMN'],
                          'S1': Dataframe['S1'],
                          'S2': Dataframe['S2'],
                          'S3': Dataframe['S3'],
                          'S4': Dataframe['S4']}).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)

    return dados

def RegressaoNovoSDR (Dataframe_Novo_SDR):
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log(Dataframe_Novo_SDR['Permeabilidade Gas']),
    'Log φ': np.log(Dataframe_Novo_SDR['Porosidade RMN']),
    'S1 log': (-1)*(np.log(Dataframe_Novo_SDR['S1'])),
    'S2 log': (-1)*(np.log(Dataframe_Novo_SDR['S2'])),
    'S3 log': np.log(Dataframe_Novo_SDR['S3']),
    'S4 log': np.log(Dataframe_Novo_SDR['S4'])})
    dados_calculo = sm.add_constant(dados_calculo_log)

    atributos = dados_calculo[['const', 'Log φ', 'S3 log', 'S4 log', 'S1 log', 'S2 log']]
    rotulos = dados_calculo['Log k']
    reg_novo = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
          'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'f', 'R2'],
          'Valor': [np.exp(reg_novo.params[0]),
                    reg_novo.params[1],
                    reg_novo.params[2],
                    reg_novo.params[3],
                    reg_novo.params[4],
                    reg_novo.params[5],
                    reg_novo.rsquared]
          }).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes_novo['Valor']['a']
    b = coeficientes_novo['Valor']['b']
    c = coeficientes_novo['Valor']['c']
    d = coeficientes_novo['Valor']['d']
    e = coeficientes_novo['Valor']['e']
    f = coeficientes_novo['Valor']['f']
    phi = Dataframe_Novo_SDR['Porosidade RMN']
    s1 = Dataframe_Novo_SDR['S1']
    s2 = Dataframe_Novo_SDR['S2']
    s3 = Dataframe_Novo_SDR['S3']
    s4 = Dataframe_Novo_SDR['S4']
    k = a*(phi**b)*(s3**c)*(s4**d)/((s1**e)*(s2**f))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(Dataframe_Novo_SDR['Permeabilidade Gas'])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)



    return reg_novo, coeficientes_novo, pd.concat([Dataframe_Novo_SDR, dados], axis = 1), sigma

def previsaoMachineLearning (Modelo_Treinado, Dados_usados_Treino):
    dados_fit = Dados_usados_Treino.copy()
    dados_copia = dados_fit.copy().drop(['Permeabilidade Gas', 'Categoria Litofacies', 'Amostra', 'Poço'], axis = 1)
    previsao = Modelo_Treinado.predict(dados_copia)
    dados_fit['Permeabilidade Prevista'] = previsao

    #Erro Sigma
    k_p = np.log10(previsao)
    k_g = np.log10(dados_fit['Permeabilidade Gas'])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)


    return dados_fit, sigma

def fitRandomForestRegressor(X, y, Versao = 1.0, n_jobs = 64, Pasta_Salvamento = None,
                             n_estimators = [100],
                             criterion = ['squared_error'],
                             max_depth = [None],
                             min_samples_split = [2],
                             min_samples_leaf = [1],
                             min_weight_fraction_leaf = [0.0],
                             max_features = [1.0],
                             max_leaf_nodes = [None],
                             min_impurity_decrease = [0.0],
                             bootstrap = [True],
                             oob_score = [False],
                             warm_start = [False],
                             ccp_alpha = [0.0],
                             max_samples = [None]):

    hiper_parametros = {
        "n_estimators" : n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split" : min_samples_split,
        'min_samples_leaf' : min_samples_leaf,
        'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'max_features': max_features,
        'max_leaf_nodes': max_leaf_nodes,
        'min_impurity_decrease': min_impurity_decrease,
        'bootstrap': bootstrap,
        'oob_score': oob_score,
        'ccp_alpha': ccp_alpha,
        'max_samples': max_samples
        }

    rfr = RandomForestRegressor(random_state = 0)

    rfr_GSCV = GridSearchCV(estimator = rfr,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    rfr_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_Random_Forest_Regressor' + '.sav'
    pickle.dump(rfr_GSCV, open(filename, 'wb'))

    return rfr_GSCV

def fitMultiLayerPercepetronRegressor (X, y, Pasta_Salvamento = None, Versao = 1.0, n_jobs = 64,
                                       hidden_layer_sizes= [(100,)],
                                       activation= ['relu'],
                                       solver= ['adam'],
                                       alpha= [0.0001],
                                       learning_rate= ['constant'],
                                       learning_rate_init= [0.001],
                                       power_t= [0.5],
                                       max_iter = 200,
                                       tol= [0.0001],
                                       warm_start= [False],
                                       momentum= [0.9],
                                       nesterovs_momentum= [True],
                                       early_stopping= [False],
                                       validation_fraction= [0.1],
                                       beta_1= [0.9], 
                                       beta_2= [0.999], 
                                       epsilon= [1e-08], 
                                       n_iter_no_change= [10]):

    camada = str(hidden_layer_sizes)
    hiper_parametros = {
        'hidden_layer_sizes': hidden_layer_sizes,
        "activation": activation,
        'solver': solver,
        "alpha": alpha,
        'solver': solver,
        'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init,
        'power_t': power_t,
        'tol': tol,
        'warm_start': warm_start,
        'momentum': momentum,
        'nesterovs_momentum': nesterovs_momentum,
        'validation_fraction': validation_fraction,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'epsilon': epsilon,
        'n_iter_no_change': n_iter_no_change
        }

    mlp = MLPRegressor(random_state = 0, max_iter = max_iter)

    mlp_GSCV = GridSearchCV(estimator = mlp,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)

    mlp_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_MLPRegressor_Camada_' + camada + '.sav'
    pickle.dump(mlp_GSCV, open(filename, 'wb'))

    return mlp_GSCV

def fitGradienteBoosting(X, y, Pasta_Salvamento = None, Versao = 1.0, n_jobs = 64,
                         loss = ['squared_error'],
                         learning_rate = [0.1],
                         n_estimators = [100],
                         subsample = [1.0],
                         criterion = ['friedman_mse'],
                         min_samples_split = [2],
                         min_samples_leaf = [1],
                         min_weight_fraction_leaf = [0.0],
                         max_depth = [3],
                         min_impurity_decrease = [0.0],
                         init = [None],
                         max_features = [None],
                         alpha = [0.9], 
                         max_leaf_nodes = [None],
                         warm_start = [False],
                         validation_fraction = [0.1],
                         n_iter_no_change = [None],
                         tol = [0.0001],
                         ccp_alpha = [0.0]):

    hiper_parametros = {
        "loss" : loss,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        'subsample': subsample,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_depth": max_depth,
        "min_impurity_decrease": min_impurity_decrease,
        "init": init,
        "max_features": max_features,
        "alpha": alpha,
        "max_leaf_nodes": max_leaf_nodes,
        "warm_start": warm_start,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
        "tol": tol,
        "ccp_alpha": ccp_alpha
        }

    gbr = GradientBoostingRegressor(random_state = 0)

    gbr_GSCV = GridSearchCV(estimator = gbr,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)
    
    gbr_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_GBRegressor' + '.sav'
    pickle.dump(gbr_GSCV, open(filename, 'wb'))
    
    return gbr_GSCV


def fitHistGradientBoostingRegressor(X, y, Pasta_Salvamento = None, Versao = 1.0, n_jobs = 64,
                                     loss = ['squared_error'],
                                     quantile = [None],
                                     learning_rate =[0.1],
                                     max_iter = [100],
                                     max_leaf_nodes=[31],
                                     max_depth = [None],
                                     min_samples_leaf = [20],
                                     l2_regularization = [0.0],
                                     max_bins = [255],
                                     categorical_features = [None],
                                     monotonic_cst = [None],
                                     interaction_cst = [None],
                                     warm_start = [False],
                                     early_stopping = ['auto'],
                                     scoring = ['loss'],
                                     validation_fraction = [0.1],
                                     n_iter_no_change = [10],
                                     tol = [1e-07]):
    hiper_parametros = {
        "loss" : loss,
        'quantile' : quantile,
        'learning_rate' : learning_rate,
        'max_iter' : max_iter,
        'max_leaf_nodes' : max_leaf_nodes,
        'max_depth' : max_depth,
        'min_samples_leaf' : min_samples_leaf,
        'l2_regularization' : l2_regularization,
        'max_bins' : max_bins,
        'categorical_features' : categorical_features,
        'monotonic_cst' : monotonic_cst,
        'interaction_cst' : interaction_cst,
        'warm_start' : warm_start,
        'early_stopping' : early_stopping,
        'scoring' : scoring,
        'validation_fraction' : validation_fraction,
        'n_iter_no_change' : n_iter_no_change,
        'tol' : tol
    }


    hgb = HistGradientBoostingRegressor(random_state = 0)

    hgb_GSCV = GridSearchCV(estimator = hgb,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)

    hgb_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_HGBRegressor' + '.sav'
    pickle.dump(hgb_GSCV, open(filename, 'wb'))
    
    return hgb_GSCV



def fitSGDRegressor(X, y, Pasta_Salvamento = None, Versao = 1.0, n_jobs = 64,
                    loss = ['squared_error', 'huber'],
                    penalty = ['l2'],
                    alpha = [0.0001, 0.001],
                    fit_intercept = [True, False],
                    tol = [0.001, 0.01],
                    learning_rate = ['constant', 'optimal', 'adaptive'],
                    eta0= [0.01],
                    average = [5, 10]):

    hiper_parametros = {
        "loss": loss,
        'penalty': penalty,
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'tol': tol,
        'learning_rate': learning_rate,
        'eta0': eta0,
        'average': average
    }

    SGD = SGDRegressor(random_state = 0, max_iter = 65536, shuffle=True)

    sgd_GSCV = GridSearchCV(estimator = SGD,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)
    
    sgd_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_SGDRegressor' + '.sav'
    pickle.dump(sgd_GSCV, open(filename, 'wb'))
    
    return sgd_GSCV




def fitGammaRegressor(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                      alpha = [1.0, 0.1, 10],
                      fit_intercept = [True, False],
                      solver = ['lbfgs', 'newton-cholesky'],
                      max_iter = 16384,
                      tol = [0.0001, 0.001],
                      warm_start = [False, True]):

    hiper_parametros = {
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'solver': solver,
        'tol': tol,
        'warm_start': warm_start
        }

    gamma = GammaRegressor(max_iter = max_iter)

    gamma_GSCV = GridSearchCV(estimator = gamma,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    gamma_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_GammaRegressor' + '.sav'
    pickle.dump(gamma_GSCV, open(filename, 'wb'))

    return gamma_GSCV


def fitTheilSenRegressor(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                         fit_intercept=[True, False],
                         max_subpopulation = [10000.0, 1000.0, 100.0],
                         n_subsamples = [None],
                         max_iter = 16384,
                         tol = [0.001, 0.0001]):

    hiper_parametros = {
        'fit_intercept': fit_intercept,
        'max_subpopulation': max_subpopulation,
        'n_subsamples': n_subsamples,
        'tol': tol
        }

    tsr = TheilSenRegressor(random_state = 0, max_iter = max_iter)

    tsr_GSCV = GridSearchCV(estimator = tsr,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    tsr_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_TheilSenRegressor' + '.sav'
    pickle.dump(tsr_GSCV, open(filename, 'wb'))

    return tsr_GSCV


def fitElasticNet(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                   alpha = [1.0],
                   l1_ratio = [0.5],
                   fit_intercept = [True],
                   precompute = [False],
                   max_iter = 1000,
                   copy_X = [True],
                   tol = [0.0001],
                   warm_start = [False],
                   positive = [False],
                   selection = ['cyclic']):

    hiper_parametros = {
        'fit_intercept': fit_intercept,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'precompute': precompute,
        'copy_X': copy_X,
        'positive': positive,
        'selection': selection,
        'warm_start': warm_start,
        'tol': tol
        }

    eln = ElasticNet(random_state = 0, max_iter = max_iter)

    eln_GSCV = GridSearchCV(estimator = eln,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    eln_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_ElasticNet' + '.sav'
    pickle.dump(eln_GSCV, open(filename, 'wb'))

    return eln_GSCV


def fitHuberRegressor(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                      epsilon = [1.35],
                      max_iter = [100],
                      alpha = [0.0001],
                      warm_start = [False],
                      fit_intercept = [True],
                      tol = [1e-05]):

    hiper_parametros = {
        'fit_intercept': fit_intercept,
        'alpha': alpha,
        'epsilon': epsilon,
        'warm_start': warm_start,
        'tol': tol
        }

    hur = HuberRegressor(max_iter = max_iter)

    hur_GSCV = GridSearchCV(estimator = hur,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs)


    hur_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_HuberRegressor' + '.sav'
    pickle.dump(hur_GSCV, open(filename, 'wb'))

    return hur_GSCV


def fitPoissonRegressor(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                        alpha = [1.0],
                        fit_intercept = [True],
                        solver = ['lbfgs'],
                        max_iter = 100,
                        tol = [0.0001],
                        warm_start = [False]):

    hiper_parametros = {
        'fit_intercept': fit_intercept,
        'alpha': alpha,
        'solver': solver,
        'warm_start': warm_start,
        'tol': tol
        }

    poi = PoissonRegressor(max_iter = max_iter)

    poi_GSCV = GridSearchCV(estimator = poi,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    poi_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_PoissonRegressor' + '.sav'
    pickle.dump(poi_GSCV, open(filename, 'wb'))

    return poi_GSCV



def fitBayesianRidge(X, y, Versao = 1.0, Pasta_Salvamento = None, n_jobs = 64,
                     max_iter=300,
                     tol=[0.001],
                     alpha_1=[1e-06],
                     alpha_2=[1e-06],
                     lambda_1=[1e-06],
                     lambda_2=[1e-06],
                     alpha_init=[None],
                     lambda_init=[1],
                     compute_score=[False],
                     fit_intercept=[True],
                     copy_X=[True]):

    hiper_parametros = {
        'fit_intercept': fit_intercept,
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'alpha_init': alpha_init,
        'lambda_init': lambda_init,
        'compute_score': compute_score,
        'copy_X': copy_X,
        'tol': tol
        }

    brr = BayesianRidge(max_iter = max_iter)

    brr_GSCV = GridSearchCV(estimator = brr,
                            param_grid = hiper_parametros,
                            n_jobs = n_jobs,
                            return_train_score = True)


    brr_GSCV.fit(X, y)

    filename = Pasta_Salvamento + 'Versão_' + str(Versao) + '_BayesianRidge' + '.sav'
    pickle.dump(brr_GSCV, open(filename, 'wb'))

    return brr_GSCV





####################### Visualizações ##################################
def VisualizarPorosidade(Dados, Pasta_Salvamento = None, Modelo = None,
                          Litofacies = None, Salvar = False):
    titulo = 'Ajuste da porosidade Gás com RMN'                                        # Nomes do Gráfico
    eixo_x = 'Porosidade Gás (%)'
    eixo_y = 'Porosidade RMN (%)'
    legenda = ['Resultado Esperado', 'Porosidade RMN']
    
    
    reta = pd.DataFrame({'x' : np.arange(30),                                             # Determinando Reta de ajuste
                         'y' : np.arange(30)})
    
    fig, ax = plt.subplots(figsize = (6,4))                                              # Criando os subplots
    
    sns.scatterplot(x = Dados['Porosidade Gas']*100,
                    y = Dados['Porosidade RMN']*100,
                    hue = Dados['Litofacies'],
                    palette = 'Spectral')
    
    sns.lineplot(data = reta,
                x = 'x',
                y = 'y')
    
    ax.set_xlabel(eixo_x)                                                                 # Determinando os nomes
    ax.set_ylabel(eixo_y)
    ax.set(xlim=(0, 30),
           ylim=(0, 30),
           title = titulo)
    
    if Salvar == True:
        plt.savefig(Pasta_Salvamento + titulo + '.png', format='png')
    
    plt.show()

def VisualizarPredicoes (Dados, Pasta_Salvamento = None, Modelo = None,
                          Litofacies = None, Salvar = False, Sigma = False, Valor_Sigma = 3.64):
    titulo = 'Predições da Permeabilidade\n Modelo ' + str(Modelo)
    eixo_x = 'Log Permeabilidade Gás (mD)'
    eixo_y = 'Log Permeabilidade RMN (mD)'
    reta = pd.DataFrame({'x' : np.arange(1000),
                         'y' : np.arange(1000)})
    plt.subplots(figsize = (6,4))
    sns.scatterplot(data = Dados,
                    x = 'Permeabilidade Gas',
                    y = 'Permeabilidade Prevista',
                    hue = Litofacies,
                    palette = 'Spectral')
    sns.lineplot(data = reta,
                 x = 'x',
                 y = 'y')
    if Sigma == True:
        plt.plot(reta['x'], reta['y'] * Valor_Sigma, "b-.", linewidth=1)
        plt.plot(reta['x'], reta['y'] / Valor_Sigma, "b-.", linewidth=1, label = f'+/- \u03C3: {Valor_Sigma:.2f}')




    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.title(titulo)
    plt.xlim(0.00001,1000)
    plt.ylim(0.00001,1000)
    plt.legend(loc="upper left", fontsize=10)



    plt.xscale('log')
    plt.yscale('log')

    if Salvar == True:
        plt.savefig(Pasta_Salvamento + titulo + '.png', format='png')

    plt.show()

def VisualizarDistribuicaoT2 (Dados, Pasta_Salvamento, CBW = False, Anotacao = False, Salvar = False):

    for i in np.arange(0, (len(Dados)-1), 2):
        amostra1 = Dados['Amostra'][i]
        amostra2 = Dados['Amostra'][i+1]
        titulo1 = 'Curva de Distribuição T2 amostra: ' + amostra1
        titulo2 = 'Curva de Distribuição T2 amostra: ' + amostra2
        eixo_x = 'Tempo (ms)'
        eixo_y = 'Amplitude do sinal'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,6))


        x1 = np.array(list(Dados['Tempo Distribuicao'][i]))
        y1 = np.array(list(Dados['Porosidade i'][i]))
        ax1.plot(x1,y1)
        ax1.set_xlabel(eixo_x)
        ax1.set_ylabel(eixo_y)
        ax1.set(title = titulo1)
        ax1.set_xscale('log')

        if CBW == True:
            ax1.fill_between(x1, y1, where = x1 < 3.2, alpha = 0.3)
            ax1.text(0.5,y1[30]/2, 'CBW')


        if Anotacao == True:

            ax1.annotate('T2 Geométrico Niumag', xy=(Dados['T2 Geometrico Niumag'][i], 0.5), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Geometrico Niumag'][i], color='lightgray')
            ax1.annotate('T2 Ponderado Log', xy=(Dados['T2 Ponderado Log'][i], 0.7), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Ponderado Log'][i], color='lightgray')
            ax1.annotate('T2 Médio Niumag', xy=(Dados['T2 Medio Niumag'][i], 0.9), xycoords=("data", "axes fraction"))
            ax1.axvline(x=Dados['T2 Medio Niumag'][i], color='lightgray')


        x2 = np.array(list(Dados['Tempo Distribuicao'][i+1]))
        y2 = np.array(list(Dados['Porosidade i'][i+1]))
        ax2.plot(x2, y2)
        ax2.set_xlabel(eixo_x)
        ax2.set_ylabel(eixo_y)
        ax2.set(title = titulo2)
        ax2.set_xscale('log')

        if Anotacao == True:
            ax2.annotate('T2 Geométrico Niumag', xy=(Dados['T2 Geometrico Niumag'][i+1], 0.5), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Geometrico Niumag'][i+1], color='lightgray')
            ax2.annotate('T2 Ponderado Log', xy=(Dados['T2 Ponderado Log'][i+1], 0.7), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Ponderado Log'][i+1], color='lightgray')
            ax2.annotate('T2 Médio Niumag', xy=(Dados['T2 Medio Niumag'][i+1], 0.9), xycoords=("data", "axes fraction"))
            ax2.axvline(x=Dados['T2 Medio Niumag'][i+1], color='lightgray')

        if CBW == True:
            ax2.text(0.5,y2[30]/2, 'CBW')
            ax2.fill_between(x2, y2, where = x1 < 3, alpha = 0.3)

        if Salvar == True:
            plt.savefig(Pasta_Salvamento + amostra1 + amostra2 + '.png', format='png')                           # Salvar imagem

        plt.show()
        

def VisualizarPorosidadePermeabilidade(Dados, Pasta_Salvamento = None, Modelo = None, Salvar = False):

    titulo = 'Porosidade x Permeabilidade Gás:\n Modelo ' + str(Modelo)
    eixo_x = 'Porosidade Gás (%)'
    eixo_y = 'Permeabilidade RMN log (%)'
    legenda = ['Modelo ' + str(Modelo),
               'Dados a Gás']

    plt.subplots(figsize = (6,4))                                                                                  # Criando os subplots

    sns.scatterplot(x = Dados['Porosidade RMN']*100, y = Dados['Permeabilidade Prevista'])
    sns.scatterplot(x = Dados['Porosidade Gas']*100, y = Dados['Permeabilidade Gas'])


    plt.xlabel(eixo_x)                                                                                              # Determinando os nomes
    plt.ylabel(eixo_y)
    plt.title(titulo)
    plt.xlim(0,30)
    plt.ylim(0.0001,100)
    plt.legend(legenda)
    plt.yscale('log')

    if Salvar == True:
        plt.savefig(Pasta_Salvamento + titulo + '.png', format='png')

    plt.show()
    
def VisualizarSigmoide (Dados_FZI, Pasta_Salvamento = None, Salvar = False, Modelo = None, Litofacies = None, lim_x = 1.6, lim_y = 2):
    titulo = 'Função Sigmoidal para Diferentes Tipos de Sistemas de Poros: Modelo ' + str(Modelo)
    eixo_x = 'r, Polar arm'
    eixo_y = 'teta, Polar Angle'

    plt.subplots(figsize = (6,4))

    sns.scatterplot(data = Dados_FZI,
                      x = 'Polar arm',
                      y = 'Polar angle',
                      hue = Litofacies)

    sns.lineplot(data = Dados_FZI,
                   x = 'Pontos Sigmoid',
                   y = 'Sigmoid')


    plt.xlabel(eixo_x)                                                                                              # Determinando os nomes
    plt.ylabel(eixo_y)
    plt.title(titulo)


    plt.xlim(0,lim_x)
    plt.ylim(0,lim_y)

    if Salvar == True:
        plt.savefig(Pasta_Salvamento + titulo + '.png', format='png')

    plt.show()

def VisualizarFZI (Dados_FZI, Pasta_Salvamento = None, Salvar = False, Modelo = None, Litofacies = None):
    titulo = 'Unidades de Fluxo Hidráulico - FZI: Modelo ' + str(Modelo)
    eixo_x = 'Phi_z'
    eixo_y = 'RQI'

    plt.subplots(figsize = (6,4))

    sns.scatterplot(data = Dados_FZI,
                    x = 'phi_z',
                    y = 'Polar angle',
                    hue = Litofacies)

    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.title(titulo)
    plt.xlim(0.001, 1)
    plt.ylim(0.001, 1000)
    plt.xscale('log')
    plt.yscale('log')

    if Salvar == True:
        plt.savefig(Pasta_Salvamento + titulo + '.png', format='png')


    plt.show()

def VisualizarRidgeLine(dados_totais, dados_ridge, permeabilidade_ensemble, permeabilidade_deep, permeabilidade_sdr, permeabilidade_han):
    for i in np.arange(len(dados_ridge['Poço'].unique())):
        df_copia = dados_ridge.copy()

        df_dados = dados_totais.loc[dados_totais['Poço'] == dados_totais['Poço'].unique()[i]].reset_index().drop('index', axis = 1)
        dados_plot = df_copia.loc[df_copia['Poço'] == df_copia['Poço'].unique()[i]].reset_index().drop('index', axis = 1)    

        array = np.arange(len(dados_plot.Amostra.unique()))
        fig = plt.figure(figsize = (2,10))
        cores = ['rosybrown', 'm', 'gray', 'c', 'y', 'k', 'w']

        for j in np.arange(len(array)):
            x = dados_plot['Tempo'][j*128:j*128+128]
            y = dados_plot['T2'][j*128:j*128+128]
            cor = dados_plot['Litofacie'][j*128]
            poço = dados_plot['Poço'][0]
            espacamento = j*-0.02                   # Espaçamento para poços com muitos dados
            #espacamento = j*-0.2/len(array)        # Espaçamento para poços com poucos dados
            
            if j == array[0]:
                ax = str(array[j])
                ax = fig.add_axes([0, espacamento, 1, 0.05])
                ax.fill(x, y, alpha = 0.7, color = cores[cor], linewidth=2)
                ax.fill_between(x, y, where = x < 3.2, alpha = 0.7, color = 'lime')
                ax.set_xscale('log')
                ax.set_frame_on(False)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                secx = ax.secondary_xaxis(1.2)
                secx.set_xlabel('ms')
                secx.set_xticks([0.01, 1, 100, 10000])
                plt.title(f'Espectro T2 \n Poço {poço}', fontsize=12, loc = 'center', y = 3)
            else:
                ax = str(array[j])
                ax = fig.add_axes([0, espacamento, 1, 0.05])
                ax.fill(x, y, alpha = 0.7, color = cores[cor], linewidth=2)
                ax.fill_between(x, y, where = x < 3.2, alpha = 0.7, color = 'lime')
                ax.set_xscale('log')
                ax.set_frame_on(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        tam = -1*espacamento+0.01
        ax = fig.add_axes([-3.6, espacamento, 1, tam])        # Porosidade Gás
        ax1 = fig.add_axes([-3.6, espacamento, 1, tam])       # Porosidade RMN
        ax2 = fig.add_axes([-2.4, espacamento, 1, tam])       # Permeabilidade Gás
        ax3 = fig.add_axes([-2.4, espacamento, 1, tam])       # Permeabilidade Prevista Ensemble
        ax4 = fig.add_axes([-2.4, espacamento, 1, tam])       # Permeabilidade Prevista Deep Learning
        ax5 = fig.add_axes([-1.2, espacamento, 1, tam])       # Permeabilidade Gás
        ax6 = fig.add_axes([-1.2, espacamento, 1, tam])       # Permeabilidade Prevista SDR
        ax7 = fig.add_axes([-1.2, espacamento, 1, tam])       # Permeabilidade Prevista Han
        
        poço = df_dados['Poço'][0]


        ax.plot(df_dados['Porosidade Gas']*100, df_dados['Amostra'], marker='o', color = 'r',)
        ax.set_frame_on(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.invert_yaxis()
        secx = ax.secondary_xaxis(1.25)
        ax.set_xlim(0, 40)
        ax.grid(True)
        secx.set_xticks([0, 10, 20, 30, 40])
        secx.set_xlabel('Porosidade Gás (%)', color = 'r')

        ax1.plot(df_dados['Porosidade RMN']*100, df_dados['Amostra'], marker='o', color = 'k')
        ax1.set_frame_on(False)
        ax1.get_xaxis().set_visible(False)
        ax1.invert_yaxis()
        secx1 = ax1.secondary_xaxis(1.55)
        ax1.set_xlim(0, 40)
        secx1.set_xticks([0, 10, 20, 30, 40])
        secx1.set_xlabel('Porosidade RMN (%)', color = 'k')
        ax1.set_title(f'Dados de Porosidade\n Gás e RMN \n Poço {poço}', fontsize=12, loc = 'center', y = 2.25)

        ax2.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax2.set_frame_on(False)
        ax2.set_xscale('log')
        ax2.invert_yaxis()
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        secx2 = ax2.secondary_xaxis(1.25)
        ax2.set_xlim(0.001, 1000)
        secx2.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx2.set_xlabel('Permeabilidade Gás (mD)', color = 'r')
        ax2.set_title(f'Dados de Permeabilidade\n Gás, \n{permeabilidade_ensemble} \ne {permeabilidade_deep}\n Poço {poço}', fontsize=12, loc = 'center', y = 2.25)

        ax3.plot(df_dados[permeabilidade_ensemble], df_dados['Amostra'], marker='*', color = 'k')
        ax3.set_frame_on(False)
        ax3.set_xscale('log')
        ax3.invert_yaxis()
        ax3.get_yaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        secx3 = ax3.secondary_xaxis(1.55)
        ax3.set_xlim(0.001, 1000)
        secx3.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx3.set_xlabel(f'Permeabilidade \n {permeabilidade_ensemble} (mD)', color = 'k')
        
        ax4.plot(df_dados[permeabilidade_deep], df_dados['Amostra'], marker='x', color = 'b')
        ax4.set_frame_on(False)
        ax4.set_xscale('log')
        ax4.invert_yaxis()
        ax4.get_yaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        secx4 = ax4.secondary_xaxis(1.9)
        ax4.set_xlim(0.001, 1000)
        secx4.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx4.set_xlabel(f'Permeabilidade \n {permeabilidade_deep} (mD)', color = 'b')

        ax5.plot(df_dados['Permeabilidade Gas'], df_dados['Amostra'], marker='p', color = 'r')
        ax5.set_frame_on(False)
        ax5.set_xscale('log')
        ax5.invert_yaxis()
        ax5.get_yaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        secx5 = ax5.secondary_xaxis(1.25)
        ax5.set_xlim(0.001, 1000)
        secx5.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx5.set_xlabel('Permeabilidade Gás (mD)', color = 'r')
        ax5.set_title(f'Dados de Permeabilidade\n Gás, {permeabilidade_sdr} e {permeabilidade_han}\n Poço {poço}', fontsize=12, loc = 'center', y = 2.25)

        ax6.plot(df_dados[permeabilidade_sdr], df_dados['Amostra'], marker='*', color = 'k')
        ax6.set_frame_on(False)
        ax6.set_xscale('log')
        ax6.invert_yaxis()
        ax6.get_yaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        secx6 = ax6.secondary_xaxis(1.55)
        ax6.set_xlim(0.001, 1000)
        secx6.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx6.set_xlabel(f'Permeabilidade \n {permeabilidade_sdr} (mD)', color = 'k')

        ax7.plot(df_dados[permeabilidade_han], df_dados['Amostra'], marker='x', color = 'b')
        ax7.set_frame_on(False)
        ax7.set_xscale('log')
        ax7.invert_yaxis()
        ax7.get_yaxis().set_visible(False)
        ax7.get_xaxis().set_visible(False)
        secx7 = ax7.secondary_xaxis(1.9)
        ax7.set_xlim(0.001, 1000)
        secx7.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        secx7.set_xlabel(f'Permeabilidade \n {permeabilidade_han} (mD)', color = 'b')
        
        plt.show()

        



















## Funções CNN

def ConversaoDataFrame (diretorio_imagens, diretorio_dados, diretorio_perneabilidade):
    # Diretórios das imagens e dos dados
    diretorio_imagens = diretorio_imagens
    diretorio_dados = diretorio_dados
    diretorio_perneabilidade = diretorio_perneabilidade

    arquivos = sorted(os.listdir(diretorio_imagens))
    dados = sorted(os.listdir(diretorio_dados))

    dados_permeabilidade = pd.read_excel(diretorio_perneabilidade, sheet_name = 'Planilha1')
    dados_permeabilidade['Permeabilidade (mD)'][0]

    # Coletando cada uma das imagens
    imagem = []
    for i in np.arange(len(arquivos)):
        diretorio = diretorio_imagens + str(arquivos[i]) + '/'
        lista_imagens = sorted(os.listdir(diretorio))
        img = []
        for j in np.arange(len(lista_imagens)):
            diretorio_imgem = diretorio + lista_imagens[j]
            img.append(cv.imread(diretorio_imgem, cv.IMREAD_GRAYSCALE))
        imagem.append(img)

    amostras = len(imagem)        # Quantidade de amostras
    n_imagens = len(imagem[0])    # Quantidade de imagens que cada amostra possui

    # Coletando cada um dos Threshould's
    dados_threshould = []
    nome_amostra = []
    for i in np.arange(amostras):
        dir_dados = diretorio_dados + str(dados[i])
        arqu = pd.read_csv(dir_dados, sep = ':').T.reset_index().drop('index', axis = 1).T
        treshould = []
        for j in np.arange(n_imagens):
            treshould.append(int(arqu[1][j][1:-4]))
            nome_amostra.append(dados[i][:3])
        dados_threshould.append(treshould)


    # Obtendo os histogramas, a escala de cinza e a multiplicação
    escala_cinza = np.arange(255, -1, -1)
    histogramas = []
    multiplicacao_hist_cinza = []
    soma_multiplicacao = []

    for i in np.arange(amostras):
        threshould = dados_threshould[i]
        image = imagem[i]
        histograma = []
        multiplicacao = []
        soma = []
        for j in np.arange(n_imagens):
            histg = cv.calcHist([image[j]], [0], None, [256], [0,256])
            mult = histg[:threshould[j]].T[0]*escala_cinza[:threshould[j]]
            histograma.append(histg)
            multiplicacao.append(mult)
            soma.append(sum(mult))
        histogramas.append(histograma)
        multiplicacao_hist_cinza.append(multiplicacao)
        soma_multiplicacao.append(soma)


    # Calculando a permeabilidade em cada Slice
    permeabilidade_slice = []
    for i in np.arange(amostras):
        k_slice = []
        for j in np.arange(n_imagens):
            soma_amostra = sum(soma_multiplicacao[i])/n_imagens
            k_gas = dados_permeabilidade['Permeabilidade (mD)'][i]
            soma_slice = soma_multiplicacao[i][j]
            k_slice.append(k_gas*soma_slice/soma_amostra)
        permeabilidade_slice.append(k_slice)

    # Criando DataFrame com os dados da imagem e permeabilidade
    lista_imagem = []
    permeabilidade_input = []
    for i in np.arange(amostras):
        for j in np.arange(n_imagens):
            #reshape = imagem[i][j].reshape(1, pixel_imagem*pixel_imagem)
            lista_imagem.append(imagem[i][j])
            permeabilidade_input.append(permeabilidade_slice[i][j])

    dados_entrada = pd.DataFrame({'Amostra': nome_amostra,
                                'Imagem': lista_imagem,
                                'Permeabilidade': permeabilidade_input})

    return dados_entrada