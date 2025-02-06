#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning models
#
#  author: Francisco Perdigon Romero, Wesley Chorney, and Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================

import pickle
import time
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import wilcoxon
from scipy.signal import welch
from prettytable import PrettyTable

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import losses

import visualization as vs
from metrics import MAD, SSD, PRD, COS_SIM, RMSE, MAE, SNR
from dfilters import FIR_test_Dataset, IIR_test_Dataset
import dl_models as models

from backend import in_train_phase

# setting random seeds to constant value (doesn't guarantee 100% reproducibility though, some layers does not get affected by this).
tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)

#use LaTeX like font styling, takes considerably longer for plotting
plt.rcParams.update({
       'text.usetex':True,
       "font.family": "serif",        
       "font.serif": ["Times"],  
       "axes.labelsize": 10,   #75            # Font size for axes labels
       "axes.titlesize": 11,   #75            # Font size for titles
       "legend.fontsize": 8,  #70            # Font size for legend
       "xtick.labelsize": 8,  #70            # Font size for x-axis ticks
       "ytick.labelsize": 8,   #70           # Font size for y-axis ticks
       "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        })
           
    
###########################################
##### Results generation functions ########
###########################################
### Needed functions for loading data and plotting ###
def load_experiment(experiment_name, DATASET):
    '''This function loads the pkl files to a list of numpy arrays [X_test, y_test, y_pred].'''
    with open('results/'+DATASET+'/test_results_' + experiment_name + '_nv1.pkl', 'rb') as input:
        test_nv1 = pickle.load(input)
    with open('results/'+DATASET+'/test_results_' + experiment_name + '_nv2.pkl', 'rb') as input:
        test_nv2 = pickle.load(input)

    test_results = [np.concatenate((test_nv1[0], test_nv2[0])),
                    np.concatenate((test_nv1[1], test_nv2[1])),
                    np.concatenate((test_nv1[2], test_nv2[2]))]
    return test_results

def get_values_plot(values, start_index=1000, n=16):
    '''This function puts the signals to be plotted in the correct shape.'''
    if n==16:
        step = 2
    else:
        step = 1
    vals = values[start_index:(start_index+n):step,...]
    vals = vals.reshape(int(n/step*512))
    return vals

def make_figure(values, metrics_list, labels_list, label=None, start_index=1000, n=16, **kwargs):
    '''This function handles the plotting and the labelling text per figure.'''
    #make the legend info
    original = kwargs.get('original', None)
    title = kwargs.get('title', '')
    savename = kwargs.get('savename', '')
    
    if original is not None:
        original = get_values_plot(original, start_index=start_index, n=n)
    values = get_values_plot(values, start_index=start_index, n=n)
    legend_info = []
    for metric, label2 in zip(metrics_list, labels_list):
        leg_inf = r'${}\ =\ {}$'.format(label2, metric)
        legend_info.append(leg_inf)
    
    vs.ecg_plot(values, legend_info, label=label, original=original,
                title=title, savename=savename)
    return

def generate_figs(test_outputs, models_list, label_list, title_list, subject='', start_index=1000, n=16, **kwargs):
    '''This function generates and handles the figures for all methods.'''
    #first make one plot of baseline + noise added
    labels_list = ['SSD', 'MAD', 'PRD', 'COS\_SIM', 'SNR']
    plot_base_noise = False

    Dataset = kwargs.get('Dataset')
    if n==16: # for overlapping data, not recommended.
        step = 2
    else:
        step = 1
    for test_output, model, model_label, model_title in zip(test_outputs, models_list, label_list, title_list):
        if not plot_base_noise:
            #make baseline plots
            X_test, y_test, _ = test_output
            # calculate metrics
            y_test2 = y_test[start_index:(start_index+n):step,...]
            X_test2 = X_test[start_index:(start_index+n):step,...]
            ssd = round(SSD(y_test2, X_test2).sum(), 3)
            mad = round(MAD(y_test2, X_test2).max(), 3)
            prd = round(PRD(y_test2, X_test2).mean(), 3)
            cs = round(COS_SIM(y_test2, X_test2).mean(), 3)
            snr = round(SNR(y_test2, X_test2).mean(), 3)
            metric_list = [ssd, mad, prd, cs, snr]
            # make figures
            make_figure(y_test, [],[], label=None, start_index=start_index,
                        n=n, title=r'$Original\ ECG\ Signal$', savename=f'results/{Dataset}/ecg_orig_{subject}.png')
            make_figure(X_test, metric_list, labels_list, label=r'$Noisy\ ECG$',
                        start_index=start_index, n=n,
                        title=r'$Clean\ ECG\ v.s.\ Noisy\ ECG$', savename=f'results/{Dataset}/ecg_noise_{subject}.png', original=y_test)
                        
            plot_base_noise = True
        
        _, y_test, y_pred = test_output
        # calculate metrics
        y_test2 = y_test[start_index:(start_index+n):step,...]
        y_pred2 = y_pred[start_index:(start_index+n):step,...]
        if model == 'DAE' or model == 'CNN-DAE':
            y_pred2 = y_pred2.reshape((8, 512, 1))
        ssd = round(SSD(y_test2, y_pred2).sum(), 3)
        mad = round(MAD(y_test2, y_pred2).max(), 3)
        prd = round(PRD(y_test2, y_pred2).mean(), 3)
        cs = round(COS_SIM(y_test2, y_pred2).mean(), 3)
        snr = round(SNR(y_test2, y_pred2).mean(), 3)
        metric_list = [ssd, mad, prd, cs, snr]
        # make figures
        make_figure(y_pred, metric_list, labels_list, start_index=start_index,
                    n=n, title=r'$Filtered\ ECG\ using\ {}$'.format(model_title), label=model_label,
                    savename=f'results/{Dataset}/ecg_{model}_{subject}.png', original=y_test)
    return

def generate_ecg_plot_results(EXPERIMENTS, DATASET, overlapping=True):
    '''This function generates the ECG plots for visual comparison. Better to use non-overlapping data for this.'''
    # Further analysis
    with open('data/'+DATASET+'/Dataset_' + str(1) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)
        [_, _, _, _, rnd_test1,RMN_test1, name_test1] = Dataset
    with open('data/'+DATASET+'/Dataset_' + str(2) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)
        [_, _, _, _, rnd_test2,RMN_test2, name_test2] = Dataset
    
    rnd_test = np.concatenate([rnd_test1, rnd_test2])
    RMN_test = np.concatenate([RMN_test1, RMN_test2])
    name_test = name_test1 + name_test2
    
    # Load Result For Deep Learning Models
    test_CNN_DAE = load_experiment(EXPERIMENTS[0], DATASET)
    test_DRNN = load_experiment(EXPERIMENTS[1], DATASET)
    test_FCN_DAE = load_experiment(EXPERIMENTS[2], DATASET)
    test_Multibranch_LANLD = load_experiment(EXPERIMENTS[3], DATASET)
    test_ECADAE = load_experiment(EXPERIMENTS[4], DATASET)
    test_ADAE = load_experiment(EXPERIMENTS[5], DATASET)
    test_TCDAE = load_experiment(EXPERIMENTS[6], DATASET)
    test_gatedONN1 = load_experiment(EXPERIMENTS[7], DATASET)
    test_gatedONN2 = load_experiment(EXPERIMENTS[8], DATASET)
    test_gatedONN3 = load_experiment(EXPERIMENTS[9], DATASET)
    test_gatedONN4 = load_experiment(EXPERIMENTS[10], DATASET)
    
    # Load Result For FIR Filter
    with open('results/'+DATASET+'/test_results_FIR_nv1.pkl', 'rb') as input:
        test_FIR_nv1 = pickle.load(input)
    with open('results/'+DATASET+'/test_results_FIR_nv2.pkl', 'rb') as input:
        test_FIR_nv2 = pickle.load(input)
    test_FIR = [np.concatenate((test_FIR_nv1[0], test_FIR_nv2[0])),
                np.concatenate((test_FIR_nv1[1], test_FIR_nv2[1])),
                np.concatenate((test_FIR_nv1[2], test_FIR_nv2[2]))]
    
    # Load Result For IIR Filter
    with open('results/'+DATASET+'/test_results_IIR_nv1.pkl', 'rb') as input:
        test_IIR_nv1 = pickle.load(input)
    with open('results/'+DATASET+'/test_results_IIR_nv2.pkl', 'rb') as input:
        test_IIR_nv2 = pickle.load(input)
    test_IIR = [np.concatenate((test_IIR_nv1[0], test_IIR_nv2[0])),
                np.concatenate((test_IIR_nv1[1], test_IIR_nv2[1])),
                np.concatenate((test_IIR_nv1[2], test_IIR_nv2[2]))]

    #Organize the results
    test_outputs = [test_FIR, test_IIR, test_DRNN, test_CNN_DAE, 
                    test_Multibranch_LANLD, test_FCN_DAE, test_ECADAE, test_ADAE, test_TCDAE, 
                    test_gatedONN1, test_gatedONN2, test_gatedONN3, test_gatedONN4]
    #Model name for picutres saving name
    models_list = ['FIR Filter', 'IIR Filter', 'DRNN', 'CNN-DAE', 
                   'DeepFilter', 'FCN-DAE', 'ACDAE', 'CBAM-DAE', 'TCDAE', 
                   'FGDAE1', 'FGDAE2', 'FGDAE3', 'FGDAE4']
    #For labels in figures
    label_list = [r'$FIR\ Filter$', r'$IIR\ Filter$', r'$DRNN$', r'$CNN-DAE$', 
                  r'$DeepFilter$', r'$FCN-DAE$', r'$ACDAE$', r'$CBAM-DAE$', r'$TCDAE$', 
                  r'$FGDAE,\ q=1$', r'$FGDAE,\ q=2$',
                  r'$FGDAE,\ q=3$', r'$FGDAE,\ q=4$'] 
    #For figures titles
    title_list = ['FIR\ Filter', 'IIR\ Filter', 'DRNN', 'CNN-DAE', 
                  'DeepFilter', 'FCN-DAE', 'ACDAE', 'CBAM-DAE', 'TCDAE', 
                  'FGDAE\ (q=1)',
                  'FGDAE\ (q=2)',
                  'FGDAE\ (q=3)',
                  'FGDAE\ (q=4)'] 
    
    #Subjects: 
    names =['sel16795','sel123','sel233','sel302','sel307','sel820','sel853','sel16420','sele0106','sele0121','sel32','sel49','sel14046','sel15814']
    
    #Plotting for different identations
    if overlapping:
        n = 16
        step = 2
    else:
        n = 8
        step = 1    
    for name in names: #plot data for each subject
        print('Plots for subject: ',name)
        try:
            for identation in range(0,64,8): #move 8 segments so that you get new signal. you get 8 plots per subject, change if you want more.
                first_index = next((index for index, value in enumerate(name_test) if value == name), -1)+identation
                generate_figs(test_outputs, models_list, label_list, title_list, subject=name+'_'+str(identation), start_index=first_index, n=n, Dataset=DATASET)
        except:
            print('Next subject...')

def generate_results(EXPERIMENTS, DATASET):
    '''This function is for the generation of the results (e.g., tables and box plots).'''
    # Load Results for DL models
    test_CNN_DAE = load_experiment(EXPERIMENTS[0], DATASET)
    test_DRNN = load_experiment(EXPERIMENTS[1], DATASET)
    test_FCN_DAE = load_experiment(EXPERIMENTS[2], DATASET)
    test_Multibranch_LANLD = load_experiment(EXPERIMENTS[3], DATASET)
    test_ECADAE = load_experiment(EXPERIMENTS[4], DATASET)
    test_ADAE = load_experiment(EXPERIMENTS[5], DATASET)
    test_TCDAE = load_experiment(EXPERIMENTS[6], DATASET)
    test_gatedONN1 = load_experiment(EXPERIMENTS[7], DATASET)
    test_gatedONN2 = load_experiment(EXPERIMENTS[8], DATASET)
    test_gatedONN3 = load_experiment(EXPERIMENTS[9], DATASET)
    test_gatedONN4 = load_experiment(EXPERIMENTS[10], DATASET)
    
    # Load Result FIR Filter
    with open('results/'+DATASET+'/test_results_FIR_nv1.pkl', 'rb') as input:
        test_FIR_nv1 = pickle.load(input)
    with open('results/'+DATASET+'/test_results_FIR_nv2.pkl', 'rb') as input:
        test_FIR_nv2 = pickle.load(input)
    test_FIR = [np.concatenate((test_FIR_nv1[0], test_FIR_nv2[0])),
                np.concatenate((test_FIR_nv1[1], test_FIR_nv2[1])),
                np.concatenate((test_FIR_nv1[2], test_FIR_nv2[2]))]

    # Load Result IIR Filter
    with open('results/'+DATASET+'/test_results_IIR_nv1.pkl', 'rb') as input:
        test_IIR_nv1 = pickle.load(input)
    with open('results/'+DATASET+'/test_results_IIR_nv2.pkl', 'rb') as input:
        test_IIR_nv2 = pickle.load(input)
    test_IIR = [np.concatenate((test_IIR_nv1[0], test_IIR_nv2[0])),
                np.concatenate((test_IIR_nv1[1], test_IIR_nv2[1])),
                np.concatenate((test_IIR_nv1[2], test_IIR_nv2[2]))]

    ####### Calculate Error Metrics #######
    print('Calculating error metrics ...')

    # DL Metrics

    # Exp DRNN
    [X_test, y_test, y_pred] = test_DRNN
    SSD_values_DL_DRNN = SSD(y_test, y_pred)
    MAD_values_DL_DRNN = MAD(y_test, y_pred)
    PRD_values_DL_DRNN = PRD(y_test, y_pred)
    RMSE_values_DL_DRNN = RMSE(y_test, y_pred)
    MAE_values_DL_DRNN = MAE(y_test, y_pred)
    SNR_values_DL_DRNN = SNR(y_test, y_pred)
    COS_SIM_values_DL_DRNN = COS_SIM(y_test, y_pred)

    # Exp FCN-DAE
    [X_test, y_test, y_pred] = test_FCN_DAE
    SSD_values_DL_FCN_DAE = SSD(y_test, y_pred)
    MAD_values_DL_FCN_DAE = MAD(y_test, y_pred)
    PRD_values_DL_FCN_DAE = PRD(y_test, y_pred)
    RMSE_values_DL_FCN_DAE = RMSE(y_test, y_pred)
    MAE_values_DL_FCN_DAE = MAE(y_test, y_pred)
    SNR_values_DL_FCN_DAE = SNR(y_test, y_pred)
    COS_SIM_values_DL_FCN_DAE = COS_SIM(y_test, y_pred)

    # Exp CNN DAE
    [X_test, y_test, y_pred] = test_CNN_DAE
    SSD_values_DL_CNN_DAE = SSD(y_test, y_pred)
    MAD_values_DL_CNN_DAE = MAD(y_test, y_pred)
    PRD_values_DL_CNN_DAE = PRD(y_test, y_pred)
    RMSE_values_DL_CNN_DAE = RMSE(y_test, y_pred)
    MAE_values_DL_CNN_DAE = MAE(y_test, y_pred)
    SNR_values_DL_CNN_DAE = SNR(y_test, y_pred)
    COS_SIM_values_DL_CNN_DAE = COS_SIM(y_test, y_pred)

    # Exp Multibranch LANLD
    [X_test, y_test, y_pred] = test_Multibranch_LANLD
    SSD_values_DL_Multibranch_LANLD = SSD(y_test, y_pred)
    MAD_values_DL_Multibranch_LANLD = MAD(y_test, y_pred)
    PRD_values_DL_Multibranch_LANLD = PRD(y_test, y_pred)
    RMSE_values_DL_Multibranch_LANLD = RMSE(y_test, y_pred)
    MAE_values_DL_Multibranch_LANLD = MAE(y_test, y_pred)
    SNR_values_DL_Multibranch_LANLD = SNR(y_test, y_pred)
    COS_SIM_values_DL_Multibranch_LANLD = COS_SIM(y_test, y_pred)

    # Exp ECA DAE
    [X_test, y_test, y_pred] = test_ECADAE
    SSD_values_DL_ECADAE = SSD(y_test, y_pred)
    MAD_values_DL_ECADAE = MAD(y_test, y_pred)
    PRD_values_DL_ECADAE = PRD(y_test, y_pred)
    RMSE_values_DL_ECADAE = RMSE(y_test, y_pred)
    MAE_values_DL_ECADAE = MAE(y_test, y_pred)
    SNR_values_DL_ECADAE = SNR(y_test, y_pred)
    COS_SIM_values_DL_ECADAE = COS_SIM(y_test, y_pred)

    # Exp ADAE
    [X_test, y_test, y_pred] = test_ADAE
    SSD_values_DL_ADAE = SSD(y_test, y_pred)
    MAD_values_DL_ADAE = MAD(y_test, y_pred)
    PRD_values_DL_ADAE = PRD(y_test, y_pred)
    RMSE_values_DL_ADAE = RMSE(y_test, y_pred)
    MAE_values_DL_ADAE = MAE(y_test, y_pred)
    SNR_values_DL_ADAE = SNR(y_test, y_pred)
    COS_SIM_values_DL_ADAE = COS_SIM(y_test, y_pred)

    # Exp TCDAE
    [X_test, y_test, y_pred] = test_TCDAE
    SSD_values_DL_TCDAE = SSD(y_test, y_pred)
    MAD_values_DL_TCDAE = MAD(y_test, y_pred)
    PRD_values_DL_TCDAE = PRD(y_test, y_pred)
    RMSE_values_DL_TCDAE = RMSE(y_test, y_pred)
    MAE_values_DL_TCDAE = MAE(y_test, y_pred)
    SNR_values_DL_TCDAE = SNR(y_test, y_pred)
    COS_SIM_values_DL_TCDAE = COS_SIM(y_test, y_pred)

    # Exp FGDAE (q=1)
    [X_test, y_test, y_pred] = test_gatedONN1
    SSD_values_DL_gatedONN1 = SSD(y_test, y_pred)
    MAD_values_DL_gatedONN1 = MAD(y_test, y_pred)
    PRD_values_DL_gatedONN1 = PRD(y_test, y_pred)
    RMSE_values_DL_gatedONN1 = RMSE(y_test, y_pred)
    MAE_values_DL_gatedONN1 = MAE(y_test, y_pred)
    SNR_values_DL_gatedONN1 = SNR(y_test, y_pred)
    COS_SIM_values_DL_gatedONN1 = COS_SIM(y_test, y_pred)

    # Exp FGDAE (q=2)
    [X_test, y_test, y_pred] = test_gatedONN2
    SSD_values_DL_gatedONN2 = SSD(y_test, y_pred)
    MAD_values_DL_gatedONN2 = MAD(y_test, y_pred)
    PRD_values_DL_gatedONN2 = PRD(y_test, y_pred)
    RMSE_values_DL_gatedONN2 = RMSE(y_test, y_pred)
    MAE_values_DL_gatedONN2 = MAE(y_test, y_pred)
    SNR_values_DL_gatedONN2 = SNR(y_test, y_pred)
    COS_SIM_values_DL_gatedONN2 = COS_SIM(y_test, y_pred)
    
    # Exp FGDAE (q=3)
    [X_test, y_test, y_pred] = test_gatedONN3
    SSD_values_DL_gatedONN3 = SSD(y_test, y_pred)
    MAD_values_DL_gatedONN3 = MAD(y_test, y_pred)
    PRD_values_DL_gatedONN3 = PRD(y_test, y_pred)
    RMSE_values_DL_gatedONN3 = RMSE(y_test, y_pred)
    MAE_values_DL_gatedONN3 = MAE(y_test, y_pred)
    SNR_values_DL_gatedONN3 = SNR(y_test, y_pred)
    COS_SIM_values_DL_gatedONN3 = COS_SIM(y_test, y_pred)
    
    # Exp FGDAE (q=4)
    [X_test, y_test, y_pred] = test_gatedONN4
    SSD_values_DL_gatedONN4 = SSD(y_test, y_pred)
    MAD_values_DL_gatedONN4 = MAD(y_test, y_pred)
    PRD_values_DL_gatedONN4 = PRD(y_test, y_pred)
    RMSE_values_DL_gatedONN4 = RMSE(y_test, y_pred)
    MAE_values_DL_gatedONN4 = MAE(y_test, y_pred)
    SNR_values_DL_gatedONN4 = SNR(y_test, y_pred)
    COS_SIM_values_DL_gatedONN4 = COS_SIM(y_test, y_pred)

    
    # Digital Filters

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR
    SSD_values_FIR = SSD(y_test, y_filter)
    MAD_values_FIR = MAD(y_test, y_filter)
    PRD_values_FIR = PRD(y_test, y_filter)
    RMSE_values_FIR = RMSE(y_test, y_pred)
    MAE_values_FIR = MAE(y_test, y_pred)
    SNR_values_FIR = SNR(y_test, y_pred)
    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)

    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR
    SSD_values_IIR = SSD(y_test, y_filter)
    MAD_values_IIR = MAD(y_test, y_filter)
    PRD_values_IIR = PRD(y_test, y_filter)
    RMSE_values_IIR = RMSE(y_test, y_pred)
    MAE_values_IIR = MAE(y_test, y_pred)
    SNR_values_IIR = SNR(y_test, y_pred)
    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)

    ####### Results Visualization #######

    SSD_all = [
               SSD_values_FIR,
               SSD_values_IIR,
               SSD_values_DL_DRNN,
               SSD_values_DL_CNN_DAE,
               SSD_values_DL_Multibranch_LANLD,
               SSD_values_DL_FCN_DAE,
               SSD_values_DL_ECADAE,
               SSD_values_DL_ADAE,
               SSD_values_DL_TCDAE,
               SSD_values_DL_gatedONN1,
               SSD_values_DL_gatedONN2,
               SSD_values_DL_gatedONN3,
               SSD_values_DL_gatedONN4,
               ]

    MAD_all = [
               MAD_values_FIR,
               MAD_values_IIR,
               MAD_values_DL_DRNN,
               MAD_values_DL_CNN_DAE,
               MAD_values_DL_Multibranch_LANLD,
               MAD_values_DL_FCN_DAE,
               MAD_values_DL_ECADAE,
               MAD_values_DL_ADAE,
               MAD_values_DL_TCDAE,
               MAD_values_DL_gatedONN1,
               MAD_values_DL_gatedONN2,
               MAD_values_DL_gatedONN3,
               MAD_values_DL_gatedONN4,
               ]

    PRD_all = [
               PRD_values_FIR,
               PRD_values_IIR,
               PRD_values_DL_DRNN,
               PRD_values_DL_CNN_DAE,
               PRD_values_DL_Multibranch_LANLD,
               PRD_values_DL_FCN_DAE,
               PRD_values_DL_ECADAE,
               PRD_values_DL_ADAE,
               PRD_values_DL_TCDAE,
               PRD_values_DL_gatedONN1,
               PRD_values_DL_gatedONN2,
               PRD_values_DL_gatedONN3,
               PRD_values_DL_gatedONN4,
               ]

    COS_SIM_all = [
                   COS_SIM_values_FIR,
                   COS_SIM_values_IIR,
                   COS_SIM_values_DL_DRNN,
                   COS_SIM_values_DL_CNN_DAE,
                   COS_SIM_values_DL_Multibranch_LANLD,
                   COS_SIM_values_DL_FCN_DAE,
                   COS_SIM_values_DL_ECADAE,
                   COS_SIM_values_DL_ADAE,
                   COS_SIM_values_DL_TCDAE,
                   COS_SIM_values_DL_gatedONN1,
                   COS_SIM_values_DL_gatedONN2,
                   COS_SIM_values_DL_gatedONN3,
                   COS_SIM_values_DL_gatedONN4,
                   ]

    RMSE_all = [
                RMSE_values_FIR,
                RMSE_values_IIR,
                RMSE_values_DL_DRNN,
                RMSE_values_DL_CNN_DAE,
                RMSE_values_DL_Multibranch_LANLD,
                RMSE_values_DL_FCN_DAE,
                RMSE_values_DL_ECADAE,
                RMSE_values_DL_ADAE,
                RMSE_values_DL_TCDAE,
                RMSE_values_DL_gatedONN1,
                RMSE_values_DL_gatedONN2,
                RMSE_values_DL_gatedONN3,
                RMSE_values_DL_gatedONN4,
                ]
    
    MAE_all = [
               MAE_values_FIR,
               MAE_values_IIR,
               MAE_values_DL_DRNN,
               MAE_values_DL_CNN_DAE,
               MAE_values_DL_Multibranch_LANLD,
               MAE_values_DL_FCN_DAE,
               MAE_values_DL_ECADAE,
               MAE_values_DL_ADAE,
               MAE_values_DL_TCDAE,
               MAE_values_DL_gatedONN1,
               MAE_values_DL_gatedONN2,
               MAE_values_DL_gatedONN3,
               MAE_values_DL_gatedONN4,
               ]
    
    SNR_all = [
               SNR_values_FIR,
               SNR_values_IIR,
               SNR_values_DL_DRNN,
               SNR_values_DL_CNN_DAE,
               SNR_values_DL_Multibranch_LANLD,
               SNR_values_DL_FCN_DAE,
               SNR_values_DL_ECADAE,
               SNR_values_DL_ADAE,
               SNR_values_DL_TCDAE,
               SNR_values_DL_gatedONN1,
               SNR_values_DL_gatedONN2,
               SNR_values_DL_gatedONN3,
               SNR_values_DL_gatedONN4,
               ]
               
    FIR_metrics = [SSD_values_FIR.squeeze(),MAD_values_FIR.squeeze(),PRD_values_FIR.squeeze(),COS_SIM_values_FIR.squeeze(),RMSE_values_FIR.squeeze(),MAE_values_FIR.squeeze(),SNR_values_FIR.squeeze()]
    IIR_metrics = [SSD_values_IIR.squeeze(),MAD_values_IIR.squeeze(),PRD_values_IIR.squeeze(),COS_SIM_values_IIR.squeeze(),RMSE_values_IIR.squeeze(),MAE_values_IIR.squeeze(),SNR_values_IIR.squeeze()]
    DRNN_metrics = [SSD_values_DL_DRNN.squeeze(),MAD_values_DL_DRNN.squeeze(),PRD_values_DL_DRNN.squeeze(),COS_SIM_values_DL_DRNN.squeeze(),RMSE_values_DL_DRNN.squeeze(),MAE_values_DL_DRNN.squeeze(),SNR_values_DL_DRNN.squeeze()]
    CNN_DAE_metrics = [SSD_values_DL_CNN_DAE.squeeze(),MAD_values_DL_CNN_DAE.squeeze(),PRD_values_DL_CNN_DAE.squeeze(),COS_SIM_values_DL_CNN_DAE.squeeze(),RMSE_values_DL_CNN_DAE.squeeze(),MAE_values_DL_CNN_DAE.squeeze(),SNR_values_DL_CNN_DAE.squeeze()]
    Multibranch_LANLD_metrics = [SSD_values_DL_Multibranch_LANLD.squeeze(),MAD_values_DL_Multibranch_LANLD.squeeze(),PRD_values_DL_Multibranch_LANLD.squeeze(),COS_SIM_values_DL_Multibranch_LANLD.squeeze(),RMSE_values_DL_Multibranch_LANLD.squeeze(),MAE_values_DL_Multibranch_LANLD.squeeze(),SNR_values_DL_Multibranch_LANLD.squeeze()]
    FCN_DAE_metrics = [SSD_values_DL_FCN_DAE.squeeze(),MAD_values_DL_FCN_DAE.squeeze(),PRD_values_DL_FCN_DAE.squeeze(),COS_SIM_values_DL_FCN_DAE.squeeze(),RMSE_values_DL_FCN_DAE.squeeze(),MAE_values_DL_FCN_DAE.squeeze(),SNR_values_DL_FCN_DAE.squeeze()]
    ECADAE_metrics = [SSD_values_DL_ECADAE.squeeze(),MAD_values_DL_ECADAE.squeeze(),PRD_values_DL_ECADAE.squeeze(),COS_SIM_values_DL_ECADAE.squeeze(),RMSE_values_DL_ECADAE.squeeze(),MAE_values_DL_ECADAE.squeeze(),SNR_values_DL_ECADAE.squeeze()]
    ADAE_metrics = [SSD_values_DL_ADAE.squeeze(),MAD_values_DL_ADAE.squeeze(),PRD_values_DL_ADAE.squeeze(),COS_SIM_values_DL_ADAE.squeeze(),RMSE_values_DL_ADAE.squeeze(),MAE_values_DL_ADAE.squeeze(),SNR_values_DL_ADAE.squeeze()]
    TCDAE_metrics = [SSD_values_DL_TCDAE.squeeze(),MAD_values_DL_TCDAE.squeeze(),PRD_values_DL_TCDAE.squeeze(),COS_SIM_values_DL_TCDAE.squeeze(),RMSE_values_DL_TCDAE.squeeze(),MAE_values_DL_TCDAE.squeeze(),SNR_values_DL_TCDAE.squeeze()]
    gatedONN1_metrics = [SSD_values_DL_gatedONN1.squeeze(),MAD_values_DL_gatedONN1.squeeze(),PRD_values_DL_gatedONN1.squeeze(),COS_SIM_values_DL_gatedONN1.squeeze(),RMSE_values_DL_gatedONN1.squeeze(),MAE_values_DL_gatedONN1.squeeze(),SNR_values_DL_gatedONN1.squeeze()]
    gatedONN2_metrics = [SSD_values_DL_gatedONN2.squeeze(),MAD_values_DL_gatedONN2.squeeze(),PRD_values_DL_gatedONN2.squeeze(),COS_SIM_values_DL_gatedONN2.squeeze(),RMSE_values_DL_gatedONN2.squeeze(),MAE_values_DL_gatedONN2.squeeze(),SNR_values_DL_gatedONN2.squeeze()]
    gatedONN3_metrics = [SSD_values_DL_gatedONN3.squeeze(),MAD_values_DL_gatedONN3.squeeze(),PRD_values_DL_gatedONN3.squeeze(),COS_SIM_values_DL_gatedONN3.squeeze(),RMSE_values_DL_gatedONN3.squeeze(),MAE_values_DL_gatedONN3.squeeze(),SNR_values_DL_gatedONN3.squeeze()]
    gatedONN4_metrics = [SSD_values_DL_gatedONN4.squeeze(),MAD_values_DL_gatedONN4.squeeze(),PRD_values_DL_gatedONN4.squeeze(),COS_SIM_values_DL_gatedONN4.squeeze(),RMSE_values_DL_gatedONN4.squeeze(),MAE_values_DL_gatedONN4.squeeze(),SNR_values_DL_gatedONN4.squeeze()]
    
    # p-values
    # Perform a paired t-test across all combined error metrics
    errors_array = [FIR_metrics, IIR_metrics, DRNN_metrics, CNN_DAE_metrics, 
                    Multibranch_LANLD_metrics, FCN_DAE_metrics, ECADAE_metrics, ADAE_metrics, TCDAE_metrics, 
                    gatedONN1_metrics, gatedONN2_metrics, gatedONN3_metrics, gatedONN4_metrics]
    #overall significance
    # for q1 model
    p_values1 = []
    proposed_model_error = np.concatenate(gatedONN1_metrics) # the chosen model for evaluation.
    for other_metrics in errors_array:
        other_model_error = np.concatenate(other_metrics)
        if np.sum(proposed_model_error-other_model_error)==0:
            p_values1.append('-')
        else:
            w_statistic, p_value = wilcoxon(proposed_model_error, other_model_error)
            p_values1.append(p_value.squeeze())
        if p_value.squeeze() < 0.05: # in the FGDAE paper, we use alpha=0.05
            print("significant")
        else:
            print("not significant")
    print()
    # for q2 model
    p_values2 = []
    proposed_model_error = np.concatenate(gatedONN2_metrics) # the chosen model for evaluation.
    for other_metrics in errors_array:
        other_model_error = np.concatenate(other_metrics)
        if np.sum(proposed_model_error-other_model_error)==0:
            p_values2.append('-')
        else:
            w_statistic, p_value = wilcoxon(proposed_model_error, other_model_error)
            p_values2.append(p_value.squeeze())
        if p_value.squeeze() < 0.05: # in the FGDAE paper, we use alpha=0.05
            print("significant")
        else:
            print("not significant")
    print()
    
    #Exp_names only for table method names
    Exp_names = ['FIR Filter', 'IIR Filter', 'DRNN', 'CNN-DAE', 'DeepFilter', 'FCN-DAE', 'ACDAE', 'CBAM-DAE', 'TCDAE', 
                 'FGDAE, q=1 (ours)','FGDAE, q=2 (ours)','FGDAE, q=3 (ours)','FGDAE, q=4 (ours)']
    metrics = ['SSD', 'p-value1 (q=1)', 'p-value1 (q=2)', 'MAD', 'p-value2 (q=1)', 'p-value2 (q=2)', 'PRD', 'p-value3 (q=1)', 'p-value3 (q=2)', 'COS_SIM', 'p-value4 (q=1)', 'p-value4 (q=2)', 'RMSE', 'p-value5 (q=1)', 'p-value5 (q=2)', 'MAE', 'p-value6 (q=1)', 'p-value6 (q=2)', 'SNR', 'p-value7 (q=1)', 'p-value7 (q=2)', 'p-value (q=1)', 'p-value (q=2)']
    metric_values = [SSD_all, MAD_all, PRD_all, COS_SIM_all, RMSE_all, MAE_all, SNR_all, p_values1, p_values2]
    
    # Metrics table
    vs.generate_table(metrics, metric_values, Exp_names, pvalue=True)

    
    ################################################################################################################
    # Further analysis: we need the subject names, artifact mixing mask, and the noise scale for this.
    with open('data/'+DATASET+'/Dataset_' + str(1) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)
        [_, _, _, _, rnd_test1,RMN_test1, name_test1] = Dataset
    with open('data/'+DATASET+'/Dataset_' + str(2) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)
        [_, _, _, _, rnd_test2,RMN_test2, name_test2] = Dataset
    
    rnd_test = np.concatenate([rnd_test1, rnd_test2])
    RMN_test = np.concatenate([RMN_test1, RMN_test2])
    name_test = name_test1 + name_test2
    
    # noise-wise analysis (for each artifact combination, for each method)
    for idx_exp in range (len(Exp_names)):
        print("Evaluation for method: ", Exp_names[idx_exp])
        names =['000',
                '100',
                '010',
                '110',
                '001',
                '101',
                '011',
                '111',
                ]
        SSD_name =     [None] * (len(names))
        MAD_name =     [None] * (len(names))
        PRD_name =     [None] * (len(names))
        RMSE_name =     [None] * (len(names))
        MAE_name =     [None] * (len(names))
        SNR_name =     [None] * (len(names))
        COS_SIM_name = [None] * (len(names))
        for subj_idx in range(len(names)):
            SSD_name[subj_idx] = []
            MAD_name[subj_idx] = []
            PRD_name[subj_idx] = []
            RMSE_name[subj_idx] = []
            MAE_name[subj_idx] = []
            SNR_name[subj_idx] = []
            COS_SIM_name[subj_idx] = []
            for idx in range(len(RMN_test)):
                # SSD
                oua = SSD_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    SSD_name[subj_idx].append(oua)
                # MAD
                oua = MAD_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    MAD_name[subj_idx].append(oua)
                # PRD
                oua = PRD_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    PRD_name[subj_idx].append(oua)
                # RMSE
                oua = RMSE_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    RMSE_name[subj_idx].append(oua)
                # MAE
                oua = MAE_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    MAE_name[subj_idx].append(oua)
                # SNR
                oua = SNR_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    SNR_name[subj_idx].append(oua)
                # COS_SIM
                oua = COS_SIM_all[idx_exp][idx]
                if np.array_equal(RMN_test[idx] , np.array(list(map(int, names[subj_idx])))):
                    COS_SIM_name[subj_idx].append(oua)
        SSD_name = np.asarray(SSD_name, dtype="object")
        MAD_name = np.asarray(MAD_name, dtype="object")
        PRD_name = np.asarray(PRD_name, dtype="object")
        RMSE_name = np.asarray(RMSE_name, dtype="object")
        MAE_name = np.asarray(MAE_name, dtype="object")
        SNR_name = np.asarray(SNR_name, dtype="object")
        COS_SIM_name = np.asarray(COS_SIM_name, dtype="object")
        tb = PrettyTable(border=False)
        tb.field_names = ['Subjects', ' & SSD', ' & MAD', ' & PRD', ' & Cosine Sim', ' & RMSE', ' & MAE', ' & SNR', '\\\\ \\hline \\hline']
        ind = 0
        for name in names:
            tb_row = []
            tb_row.append(name)
            m_mean = np.mean(SSD_name[ind])
            m_std = np.std(SSD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(MAD_name[ind])
            m_std = np.std(MAD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(PRD_name[ind])
            m_std = np.std(PRD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(COS_SIM_name[ind])
            m_std = np.std(COS_SIM_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(RMSE_name[ind])
            m_std = np.std(RMSE_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(MAE_name[ind])
            m_std = np.std(MAE_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(SNR_name[ind])
            m_std = np.std(SNR_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            tb_row.append('\\\\ \\hline')
            tb.add_row(tb_row)
            ind += 1
        print()
        print(tb)
        print()
           
    # subject-wise analysis 
    for idx_exp in range (len(Exp_names)):
        print("Evaluation for method: ", Exp_names[idx_exp])
        names =['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database
                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database
                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database
                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH
                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
        SSD_name =     [None] * (len(names))
        MAD_name =     [None] * (len(names))
        PRD_name =     [None] * (len(names))
        RMSE_name =     [None] * (len(names))
        MAE_name =     [None] * (len(names))
        SNR_name =     [None] * (len(names))
        COS_SIM_name = [None] * (len(names))
        for subj_idx in range(len(names)):
            SSD_name[subj_idx] = []
            MAD_name[subj_idx] = []
            PRD_name[subj_idx] = []
            RMSE_name[subj_idx] = []
            MAE_name[subj_idx] = []
            SNR_name[subj_idx] = []
            COS_SIM_name[subj_idx] = []
            for idx in range(len(name_test)):
                # SSD
                oua = SSD_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    SSD_name[subj_idx].append(oua)
                # MAD
                oua = MAD_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    MAD_name[subj_idx].append(oua)
                # PRD
                oua = PRD_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    PRD_name[subj_idx].append(oua)
                # RMSE
                oua = RMSE_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    RMSE_name[subj_idx].append(oua)
                # MAE
                oua = MAE_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    MAE_name[subj_idx].append(oua)
                # SNR
                oua = SNR_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    SNR_name[subj_idx].append(oua)
                # COS_SIM
                oua = COS_SIM_all[idx_exp][idx]
                if name_test[idx] == names[subj_idx]:
                    COS_SIM_name[subj_idx].append(oua)
        SSD_name = np.asarray(SSD_name, dtype="object")
        MAD_name = np.asarray(MAD_name, dtype="object")
        PRD_name = np.asarray(PRD_name, dtype="object")
        RMSE_name = np.asarray(RMSE_name, dtype="object")
        MAE_name = np.asarray(MAE_name, dtype="object")
        SNR_name = np.asarray(SNR_name, dtype="object")
        COS_SIM_name = np.asarray(COS_SIM_name, dtype="object")
        tb = PrettyTable(border=False)
        tb.field_names = ['Subjects', ' & SSD', ' & MAD', ' & PRD', ' & Cosine Sim', ' & RMSE', ' & MAE', ' & SNR', '\\\\ \\hline \\hline']
        ind = 0
        for name in names:
            tb_row = []
            tb_row.append(name)
            m_mean = np.mean(SSD_name[ind])
            m_std = np.std(SSD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(MAD_name[ind])
            m_std = np.std(MAD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(PRD_name[ind])
            m_std = np.std(PRD_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(COS_SIM_name[ind])
            m_std = np.std(COS_SIM_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(RMSE_name[ind])
            m_std = np.std(RMSE_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(MAE_name[ind])
            m_std = np.std(MAE_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            m_mean = np.mean(SNR_name[ind])
            m_std = np.std(SNR_name[ind])
            tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
            tb_row.append('\\\\ \\hline')
            tb.add_row(tb_row)
            ind += 1
        print()
        print(tb)
        print()
    
    # Segmentation by noise amplitude
    segm = [0.2, 0.6, 1.0, 1.5, 2.0]  # real number of segmentations is len(segmentations) - 1
    SSD_seg_all = []
    MAD_seg_all = []
    PRD_seg_all = []
    RMSE_seg_all = []
    MAE_seg_all = []
    SNR_seg_all = []
    COS_SIM_seg_all = []
    for idx_exp in range(len(Exp_names)):
        SSD_seg = [None] * (len(segm) - 1)
        MAD_seg = [None] * (len(segm) - 1)
        PRD_seg = [None] * (len(segm) - 1)
        RMSE_seg = [None] * (len(segm) - 1)
        MAE_seg = [None] * (len(segm) - 1)
        SNR_seg = [None] * (len(segm) - 1)
        COS_SIM_seg = [None] * (len(segm) - 1)
        for idx_seg in range(len(segm) - 1):
            SSD_seg[idx_seg] = []
            MAD_seg[idx_seg] = []
            PRD_seg[idx_seg] = []
            RMSE_seg[idx_seg] = []
            MAE_seg[idx_seg] = []
            SNR_seg[idx_seg] = []
            COS_SIM_seg[idx_seg] = []
            for idx in range(len(rnd_test)):
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)
                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)
                # PRD
                oua = PRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    PRD_seg[idx_seg].append(oua)
                # COS SIM
                oua = COS_SIM_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    COS_SIM_seg[idx_seg].append(oua)
                # RMSe
                oua = RMSE_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    RMSE_seg[idx_seg].append(oua)
                # MAE
                oua = MAE_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAE_seg[idx_seg].append(oua)
                # SNR
                oua = SNR_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SNR_seg[idx_seg].append(oua)
                
        # Processing the last index
        # SSD
        SSD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SSD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SSD_seg[-1].append(oua)
        SSD_seg_all.append(SSD_seg)  # [exp][seg][item]

        # MAD
        MAD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAD_seg[-1].append(oua)
        MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

        # PRD
        PRD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = PRD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                PRD_seg[-1].append(oua)
        PRD_seg_all.append(PRD_seg)  # [exp][seg][item]

        # COS SIM
        COS_SIM_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = COS_SIM_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                COS_SIM_seg[-1].append(oua)
        COS_SIM_seg_all.append(COS_SIM_seg)  # [exp][seg][item]

        # RMSE
        RMSE_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = RMSE_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                RMSE_seg[-1].append(oua)
        RMSE_seg_all.append(RMSE_seg)  # [exp][seg][item]

        # MAE
        MAE_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAE_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAE_seg[-1].append(oua)
        MAE_seg_all.append(MAE_seg)  # [exp][seg][item]

        # SNR
        SNR_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SNR_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SNR_seg[-1].append(oua)
        SNR_seg_all.append(SNR_seg)  # [exp][seg][item]

        
    # Printing Tables
    seg_table_column_name = []
    i = 1
    for idx_seg in range(len(segm) - 1):
        column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
        seg_table_column_name.append(column_name)
        seg_table_column_name.append(f'p-value (q=1) {i}')
        seg_table_column_name.append(f'p-value (q=2) {i}')
        i+=1

    # SSD Table
    SSD_seg_all = np.asarray(SSD_seg_all, dtype="object") 
    SSD_seg_all = np.swapaxes(SSD_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the SSD metric')
    vs.generate_table(seg_table_column_name, SSD_seg_all, Exp_names, pvalue=True)

    # MAD Table
    MAD_seg_all = np.asarray(MAD_seg_all, dtype="object")
    MAD_seg_all = np.swapaxes(MAD_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the MAD metric')
    vs.generate_table(seg_table_column_name, MAD_seg_all, Exp_names, pvalue=True)

    # PRD Table
    PRD_seg_all = np.asarray(PRD_seg_all, dtype="object")
    PRD_seg_all = np.swapaxes(PRD_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the PRD metric')
    vs.generate_table(seg_table_column_name, PRD_seg_all, Exp_names, pvalue=True)

    # COS SIM Table
    COS_SIM_seg_all = np.asarray(COS_SIM_seg_all, dtype="object")
    COS_SIM_seg_all = np.swapaxes(COS_SIM_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the COS SIM metric')
    vs.generate_table(seg_table_column_name, COS_SIM_seg_all, Exp_names, pvalue=True)

    # RMSE Table
    RMSE_seg_all = np.asarray(RMSE_seg_all, dtype="object")
    RMSE_seg_all = np.swapaxes(RMSE_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the RMSE metric')
    vs.generate_table(seg_table_column_name, RMSE_seg_all, Exp_names, pvalue=True)

    # MAE Table
    MAE_seg_all = np.asarray(MAE_seg_all, dtype="object")
    MAE_seg_all = np.swapaxes(MAE_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the MAE metric')
    vs.generate_table(seg_table_column_name, MAE_seg_all, Exp_names, pvalue=True)

    # SNR Table
    SNR_seg_all = np.asarray(SNR_seg_all, dtype="object")
    SNR_seg_all = np.swapaxes(SNR_seg_all, 0, 1)
    print('\n')
    print('Printing Table for different noise values on the SNR metric')
    vs.generate_table(seg_table_column_name, SNR_seg_all, Exp_names, pvalue=True)

    
    ##############################################################################################################
    # Metrics graphs
    log=False
    #Exp_labels only for figures method names
    Exp_labels = [r'$FIR\ Filter$', r'$IIR\ Filter$', r'$DRNN$', r'$CNN-DAE$', r'$DeepFilter$', r'$FCN-DAE$', r'$ACDAE$', r'$CBAM-DAE$', r'$TCDAE$', 
                  r'$FGDAE,\ q=1$',r'$FGDAE,\ q=2$',r'$FGDAE,\ q=3$',r'$FGDAE,\ q=4$']
    
    vs.generate_hboxplot(SSD_all, Exp_labels, r'$SSD$', log, 'ssd', Dataset=DATASET, set_x_axis_size=(0, 100.01))
    vs.generate_hboxplot(MAD_all, Exp_labels, r'$MAD$', log, 'mad', Dataset=DATASET, set_x_axis_size=(0, 2.01))
    vs.generate_hboxplot(PRD_all, Exp_labels, r'$PRD$', log, 'prd', Dataset=DATASET, set_x_axis_size=(0, 150.01))
    vs.generate_hboxplot(COS_SIM_all, Exp_labels, r'$CosSim$', log, 'cossim', Dataset=DATASET, set_x_axis_size=(0.4, 1.01))
    vs.generate_hboxplot(MAE_all, Exp_labels, r'$MAE$', log, 'mae', Dataset=DATASET, set_x_axis_size=(0.4, 1.01))
    vs.generate_hboxplot(RMSE_all, Exp_labels, r'$RMSE$', log, 'rmse', Dataset=DATASET, set_x_axis_size=(0.4, 1.01))
    vs.generate_hboxplot(SNR_all, Exp_labels, r'$SNR$', log, 'snr', Dataset=DATASET, set_x_axis_size=(0.4, 1.01))
    
    vs.generate_boxplot(SSD_all, Exp_labels, r'$SSD$', log, 'ssd', Dataset=DATASET, set_y_axis_size=(0, 100.01))
    vs.generate_boxplot(MAD_all, Exp_labels, r'$MAD$', log, 'mad', Dataset=DATASET, set_y_axis_size=(0, 2.01))
    vs.generate_boxplot(PRD_all, Exp_labels, r'$PRD$', log, 'prd', Dataset=DATASET, set_y_axis_size=(0, 150.01))
    vs.generate_boxplot(COS_SIM_all, Exp_labels, r'$CosSim$', log, 'cossim', Dataset=DATASET, set_y_axis_size=(0.4, 1.01))
    vs.generate_boxplot(MAE_all, Exp_labels, r'$MAE$', log, 'mae', Dataset=DATASET, set_y_axis_size=(0.4, 1.01))
    vs.generate_boxplot(RMSE_all, Exp_labels, r'$RMSE$', log, 'rmse', Dataset=DATASET, set_y_axis_size=(0.4, 1.01))
    vs.generate_boxplot(SNR_all, Exp_labels, r'$SNR$', log, 'snr', Dataset=DATASET, set_y_axis_size=(0.4, 1.01))
    
    
    ################################################################################################################
    # Load testing timing
    with open('results/'+DATASET+'/timing_nv1.pkl', 'rb') as input:
        timing_nv1 = pickle.load(input)
        [train_time_list_nv1, test_time_list_nv1] = timing_nv1
    with open('results/'+DATASET+'/timing_nv2.pkl', 'rb') as input:
        timing_nv2 = pickle.load(input)
        [train_time_list_nv2, test_time_list_nv2] = timing_nv2
    train_time_list = []
    test_time_list = []
    if isinstance(train_time_list_nv1, list):
        timing_names = None
        for i in range(len(train_time_list_nv1)):
            train_time_list.append(train_time_list_nv1[i] + train_time_list_nv2[i])
    
        for i in range(len(test_time_list_nv1)):
            test_time_list.append(test_time_list_nv1[i] + test_time_list_nv2[i])
    else:
        #assume it's a dict
        timing_names = []
        for key in train_time_list_nv1.keys():
            train_time_list.append(train_time_list_nv1[key] + train_time_list_nv2[key])
            test_time_list.append(test_time_list_nv1[key] + test_time_list_nv2[key])
            timing_names.append(key)
    timing = [train_time_list, test_time_list]
    # Timing table
    timing_var = ['training', 'test']
    vs.generate_table_time(timing_var, timing, Exp_names, gpu=True)

    return



    


###########################################
###### Train/Test main functions ##########
###########################################
def train_models(EXPERIMENTS, FILTERS, DATASET):
    '''This is the function where you train DL methods, and test all the methods you have.'''
    noise_versions = [1, 2]
    for noise_version in noise_versions: #For the two noise versions, check FGDAE paper for details.
        
        ## Load dataset
        with open('data/'+DATASET+'/Dataset_' + str(noise_version) + '.pkl', 'rb') as input:
            Dataset = pickle.load(input)

        train_time_dict = {}
        test_time_dict = {}

        ## Deep learning models
        for experiment in range(len(EXPERIMENTS)):
            if not os.path.isfile('results/'+DATASET+'/test_results_' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + '.pkl'): 
                # training
                print(EXPERIMENTS[experiment]+': Testing the model.')
                start_train = datetime.now()
                print('noise version ' , noise_version)
                train_dl(Dataset, EXPERIMENTS[experiment], ds=str(noise_version))
                end_train = datetime.now()
                train_time_dict[EXPERIMENTS[experiment]] = end_train - start_train
                print("Training Time: ", end_train - start_train)
                
                # testing
                start_test = datetime.now()
                [X_test, y_test, y_pred] = test_dl(Dataset, EXPERIMENTS[experiment], ds=str(noise_version))
                end_test = datetime.now()
                test_time_dict[EXPERIMENTS[experiment]] = end_test - start_test
                print("Testing Time: ", end_test - start_test)
                
                # Save Results
                test_results = [X_test, y_test, y_pred]
                with open('results/'+DATASET+'/test_results_' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(test_results, output)
                print('Results from experiment ' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + ' saved')

        ## Classical Filters (they take very long time to run)
        # FIR
        if "FIR" in FILTERS:
            if not os.path.isfile('results/'+DATASET+'/test_results_FIR_nv' + str(noise_version) + '.pkl'): 
                # testing
                print('FIR Filter: Testing the filter')
                start_test = datetime.now()
                [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
                end_test = datetime.now()
                train_time_dict['FIR Filter'] = 0
                test_time_dict['FIR Filter'] = end_test - start_test
                print("FIR Testing Time: ", end_test - start_test)
                test_results_FIR = [X_test_f, y_test_f, y_filter]

                # Save FIR filter results
                with open('results/'+DATASET+'/test_results_FIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(test_results_FIR, output)
                print('Results from experiment FIR filter nv ' + str(noise_version) + ' saved')

        # IIR
        if "IIR" in FILTERS:
            if not os.path.isfile('results/'+DATASET+'/test_results_IIR_nv' + str(noise_version) + '.pkl'): 
                # testing
                print('IIR Filter: Testing the filter')
                start_test = datetime.now()
                [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
                end_test = datetime.now()
                train_time_dict['IIR Filter'] = 0
                test_time_dict['IIR Filter'] = end_test - start_test
                print("IIR Testing Time: ", end_test - start_test)
                test_results_IIR = [X_test_f, y_test_f, y_filter]

                # Save IIR filter results
                with open('results/'+DATASET+'/test_results_IIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(test_results_IIR, output)
                print('Results from experiment IIR filter nv ' + str(noise_version) + ' saved')

        ## Saving timing list
        timing = [train_time_dict, test_time_dict]
        with open('results/'+DATASET+'/timing_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(timing, output)
        print('Timing nv ' + str(noise_version) + ' saved')
    
    return


def test_models(EXPERIMENTS, FILTERS, DATASET):
    '''This function is only for testing all the methods you have (i.e., estimating clean signals and generating testing time tables.)'''
    noise_versions = [1, 2]
    for noise_version in noise_versions:
        
        ## Load dataset
        with open('data/'+DATASET+'/Dataset_' + str(noise_version) + '.pkl', 'rb') as input:
            Dataset = pickle.load(input)

        test_time_dict = {}

        ## Deep learning models
        for experiment in range(len(EXPERIMENTS)):
            if not os.path.isfile('results/'+DATASET+'/test_results_' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + '.pkl'): 
                # testing
                print(EXPERIMENTS[experiment]+': Testing the model.')
                start_test = datetime.now()
                [X_test, y_test, y_pred] = test_dl(Dataset, EXPERIMENTS[experiment], ds=str(noise_version))
                end_test = datetime.now()
                test_time_dict[EXPERIMENTS[experiment]] = end_test - start_test
                print("Testing Time: ", end_test - start_test)
                
                # Save Results (Overwrites any existing file).
                test_results = [X_test, y_test, y_pred]
                with open('results/'+DATASET+'/test_results_' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + '.pkl', 'wb') as output:
                    pickle.dump(test_results, output)
                print('Results from experiment ' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + ' saved\n')

        
        ## Classical Filters (they take very long time to run)
        # FIR
        if "FIR" in FILTERS:
            if not os.path.isfile('results/'+DATASET+'/test_results_FIR_nv' + str(noise_version) + '.pkl'): 
                # testing
                print('FIR Filter: Testing the filter')
                start_test = datetime.now()
                [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
                end_test = datetime.now()
                test_time_dict['FIR Filter'] = end_test - start_test
                print("FIR Testing Time: ", end_test - start_test)
                test_results_FIR = [X_test_f, y_test_f, y_filter]

                # Save FIR filter results (Overwrites any existing file).
                with open('results/'+DATASET+'/test_results_FIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  
                    pickle.dump(test_results_FIR, output)
                print('Results from experiment FIR filter nv ' + str(noise_version) + ' saved\n')

        # IIR
        if "IIR" in FILTERS:
            if not os.path.isfile('results/'+DATASET+'/test_results_IIR_nv' + str(noise_version) + '.pkl'): 
                # testing
                print('IIR Filter: Testing the filter')
                start_test = datetime.now()
                [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
                end_test = datetime.now()
                test_time_dict['IIR Filter'] = end_test - start_test
                print("IIR Testing Time: ", end_test - start_test)
                test_results_IIR = [X_test_f, y_test_f, y_filter]

                # Save IIR filter results (Overwrites any existing file).
                with open('results/'+DATASET+'/test_results_IIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  
                    pickle.dump(test_results_IIR, output)
                print('Results from experiment IIR filter nv ' + str(noise_version) + ' saved\n')

        ## Saving timing list (Overwrites any existing file).
        timing = [test_time_dict]
        with open('results/'+DATASET+'/timing_nv' + str(noise_version) + '.pkl', 'wb') as output: 
            pickle.dump(timing, output)
        print('Timing nv ' + str(noise_version) + ' saved')
    
    return
    
    
###########################################
##### Train/Test basic functions ##########
###########################################

def train_dl(Dataset, experiment, signal_size=512, ds=''):
    '''Train specified DL model.'''
    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))

    # ==================
    # LOAD THE DATA
    # ==================
    [X_train, y_train, _, _, _, _, _] = Dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=False, random_state=1)

    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'DRNN':
        # DRNN
        model = models.DRNN_denoising(signal_size=signal_size)
        model_label = 'DRNN'

    if experiment == 'Multibranch LANLD':
        # Inception-like linear and non linear dilated
        model = models.deep_filter_model_I_LANL_dilated(signal_size=signal_size)
        model_label = 'DeepFilter'

    if experiment == 'Vanilla DAE':
        model = models.VanillaAutoencoder(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'Vanilla_DAE'

    if experiment == 'CNN-DAE':
        model = models.CNN_DAE(signal_size=signal_size)
        model_label = 'CNN_DAE'

    if experiment == 'FCN-DAE':
        # FCN_DAE
        model = models.FCN_DAE(signal_size=signal_size)
        model_label = 'FCN_DAE'

    if experiment == 'ECA Skip DAE':
        # ACDAE
        model = models.ECASkipDAE(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'ACDAE'

    if experiment == 'Attention Skip DAE':
        # CDAE-BAM
        model = models.AttentionSkipDAE2(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'CDAE_BAM'

    if experiment == 'Transformer_DAE':
        # TCDAE
        model = models.Transformer_DAE(signal_size=signal_size)
        model_label = 'TCDAE'
        
    if experiment == 'Proposed_gatedONN1':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=1)
        model_label = 'Proposed_gatedONN1'
        
    if experiment == 'Proposed_gatedONN2':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=2)
        model_label = 'Proposed_gatedONN2'
        
    if experiment == 'Proposed_gatedONN3':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=3)
        model_label = 'Proposed_gatedONN3'
        
    if experiment == 'Proposed_gatedONN4':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=4)
        model_label = 'Proposed_gatedONN4'
        
    
    print('\n ' + model_label + '\n ')

    model.summary()

    # ==================
    # TRAINING SETUP
    # ==================
    # Hyper-Parameters
    epochs = int(1e5)  # 100000
    batch_size = 64
    lr = 1e-3
    minimum_lr = 1e-10
    
    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = tf.keras.losses.MeanSquaredError

    elif experiment == 'FCN-DAE' or experiment == 'CNN-DAE' or experiment == 'ECA Skip DAE' or experiment == 'Vanilla DAE':
        criterion = ssd_loss
    
    elif experiment == 'Transformer_DAE' :
        criterion = combined_huber_freq_loss

    elif experiment == 'Proposed_gatedONN1' or 'Proposed_gatedONN2' or 'Proposed_gatedONN3' or 'Proposed_gatedONN4':
        criterion = lambda y_true, y_pred: combined_ssd_mad_morph_L1norm_loss(y_true, y_pred, model)

    else:
        criterion = combined_ssd_mad_loss

    #compile model
    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.losses.MeanSquaredError, tf.keras.losses.MAE, ssd_loss, mad_loss])
    
    # Keras Callbacks
    # checkpoint
    model_filepath = 'models/'+ model_label + f'{ds}_best.weights.h5'
    

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  
                                 save_weights_only=True,
                                )
    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',  
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=0)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",  
                               min_delta=0.05,
                               mode='min',  
                               patience=10,
                               verbose=0)

    tb_log_dir = 'runs/' + model_label

    tboard = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False,# write_grads=False,
                         write_images=False, embeddings_freq=0,
                        # embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs

    # GPU
    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              callbacks=[early_stop,
                         reduce_lr,
                         checkpoint,
                         tboard])
    
    K.clear_session()


def test_dl(Dataset, experiment, signal_size=512, ds=''):
    '''Test specified DL model.'''
    
    print('Deep Learning pipeline: Testing the model')

    batch_size = 32
    lr = 1e-3

    # ==================
    # LOAD THE DATA
    # ==================
    [_, _, X_test, y_test, rnd_test, RMN_test, name_test] = Dataset
    
    # ==================
    # LOAD THE DL MODEL
    # ==================
    if experiment == 'DRNN':
        # DRNN
        model = models.DRNN_denoising(signal_size=signal_size)
        model_label = 'DRNN'

    if experiment == 'Multibranch LANLD':
        # DeepFilter
        model = models.deep_filter_model_I_LANL_dilated(signal_size=signal_size)
        model_label = 'DeepFilter'

    if experiment == 'Vanilla DAE':
        model = models.VanillaAutoencoder(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'Vanilla_DAE'

    if experiment == 'CNN-DAE':
        model = models.CNN_DAE(signal_size=signal_size)
        model_label = 'CNN_DAE'

    if experiment == 'FCN-DAE':
        # FCN_DAE
        model = models.FCN_DAE(signal_size=signal_size)
        model_label = 'FCN_DAE'

    if experiment == 'ECA Skip DAE':
        # ACDAE
        model = models.ECASkipDAE(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'ACDAE'

    if experiment == 'Attention Skip DAE':
        # CDAE-BAM
        model = models.AttentionSkipDAE2(signal_size=signal_size)
        model.build(input_shape=(None, signal_size, 1))
        model_label = 'CDAE_BAM'

    if experiment == 'Transformer_DAE':
        # TCDAE
        model = models.Transformer_DAE(signal_size=signal_size)
        model_label = 'TCDAE'

    if experiment == 'Proposed_gatedONN1':
        # Proposed gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=1)
        model_label = 'Proposed_gatedONN1'
        
    if experiment == 'Proposed_gatedONN2':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=2)
        model_label = 'Proposed_gatedONN2'
        
    if experiment == 'Proposed_gatedONN3':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=3)
        model_label = 'Proposed_gatedONN3'
        
    if experiment == 'Proposed_gatedONN4':
        # gated self-ONN DAE (FGDAE)
        model = models.GatedONNDAE(signal_size=signal_size, q=4)
        model_label = 'Proposed_gatedONN4'
        
    
    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = keras.losses.MeanSquaredError

    elif experiment == 'FCN-DAE' or experiment == 'CNN-DAE' or experiment == 'ECA Skip DAE' or experiment == 'Vanilla DAE':
        criterion = ssd_loss
    
    elif experiment == 'Transformer_DAE' :
        criterion = combined_huber_freq_loss

    elif experiment == 'Proposed_gatedONN1' or 'Proposed_gatedONN2' or 'Proposed_gatedONN3' or 'Proposed_gatedONN4':
        criterion = lambda y_true, y_pred: combined_ssd_mad_morph_L1norm_loss(y_true, y_pred, model)

    else:
        criterion = combined_ssd_mad_loss

    # ==================
    # COMPILE THE DL MODEL
    # ==================
    # checkpoint
    model_filepath = 'models/'+ model_label + f'{ds}_best.weights.h5'
    # load weights
    model.load_weights(model_filepath) 
    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.losses.MeanSquaredError, tf.keras.losses.MAE, ssd_loss, mad_loss])
    
    #############################
    # Test score
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)

    K.clear_session()

    return [X_test, y_test, y_pred]
    
    

###########################################
########### Loss functions ################
###########################################
# Custom loss with L1 normalization and other improvements (customized for FGDAE)
def combined_ssd_mad_morph_L1norm_loss(y_true, y_pred, model):
    # combined SSD and MAD
    mad = mad_loss(y_true, y_pred)
    ssd = ssd_loss(y_true, y_pred)
    base_loss = mad * 50 + ssd
    # morphology loss
    morph = cos_sim_loss(y_true, y_pred)
    gdl = gradient_difference_loss(y_true, y_pred)
    # L1 normalization of encoder
    l1_loss = l1_regularization(y_true, y_pred, model)
    return base_loss + l1_loss + morph + gdl

# L1 regularization loss of the encoder layers (customized for FGDAE)
def l1_regularization(y_true, y_pred, model):
    lambda_l1=0.01
    l1_loss = 0.0
    encoder_layers = ['gated_onn_encoder_block', 'gated_encoder_block']  # possible encoder modules
    for layer in model.layers:
        if any(encoder_name in layer.name for encoder_name in encoder_layers):
            for weight in layer.trainable_weights:  # Iterate over layer's trainable weights
                l1_loss += tf.reduce_sum(tf.abs(weight))  # L1 regularization (sum of absolute values)
    l1_loss *= lambda_l1
    return l1_loss
    
# Function to calculate the gradient difference loss (G-Loss) (customized for FGDAE)
def gradient_difference_loss(y_true, y_pred):
    grad_true1 = y_true[:, 1:] - y_true[:, :-1]
    grad_pred1 = y_pred[:, 1:] - y_pred[:, :-1]
    return K.max(tf.abs(grad_true1 - grad_pred1)) 
    
# similarity loss using correlation between the reference and estimated (customized for FGDAE)
def cos_sim_loss(y_true, y_pred):
    lambda_morph=10
    similarity_min = K.min(tfp.stats.correlation(y_true, y_pred,sample_axis=1))
    return (1 - similarity_min) * lambda_morph
    
# Custom loss SSD
def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)

# Custom loss SAD
def sad_loss(y_true, y_pred):
    return K.sum(K.abs(K.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true), axis=-2)

# Combined loss SSD + MSE
def combined_ssd_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-2) * 500 + K.sum(K.square(y_true - y_pred), axis=-2)

# Combined loss SSD + MAD
def combined_ssd_mad_loss(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred), axis=-2) * 50 + K.sum(K.square(y_true - y_pred), axis=-2)


# Huber+ frequency (For TCDAE)
def hann_window(length):
    n = tf.range(length)
    n = tf.cast(n, tf.float32)  # Convert y to float
    window = 0.5 - 0.5 * tf.cos((2.0 * tf.constant(np.pi) * n) / tf.cast((length - 1), tf.float32))
    return window
def rfftfreq(n, d=1.0):
    return np.fft.rfftfreq(n, d)
def periodogram(signal, sample_rate, window='hann', nfft=None, scaling='density'):
    # Apply the window function
    window_func = hann_window(tf.shape(signal)[0])
    windowed_signal = signal * window_func
    # Compute the Discrete Fourier Transform (DFT)
    dft = tf.signal.fft(tf.cast(windowed_signal, tf.complex64))
    # Compute the squared magnitude of the DFT
    power_spectrum = tf.square(tf.abs(dft))
    # Normalize the power spectrum
    if scaling == 'density':
        power_spectrum /= sample_rate
    elif scaling == 'spectrum':
        power_spectrum /= tf.reduce_sum(window_func)**2
    elif scaling == 'magnitude':
        power_spectrum = tf.math.sqrt(power_spectrum)
    # Compute the frequencies
    frequencies = rfftfreq(512, 1/sample_rate)# sigLen changed to 512 --> complex input to float
    frequencies_tensor = tf.convert_to_tensor(frequencies, dtype=tf.float32)
    return frequencies_tensor, power_spectrum
def combined_huber_freq_loss(y_true,y_pred):
    delta = 0.05   ##0.5
    frequencies_orig, power_spectrum_orig = periodogram(y_true,360)
    frequencies_denoised, power_spectrum_denoised = periodogram(y_pred,360)
    similarity = tf.reduce_mean(tfp.stats.correlation(power_spectrum_orig, power_spectrum_denoised,sample_axis=1))
    frequency_weights = tf.math.exp(1 - abs(similarity))
    squared_loss = tf.square(y_true - y_pred)
    linear_loss = delta * (tf.abs(y_true - y_pred) - 0.5 * delta)
    weighted_loss = frequency_weights * tf.where(tf.abs(y_true - y_pred) <= delta, squared_loss, linear_loss)
    loss = tf.reduce_mean(weighted_loss)
    return loss+keras.losses.cosine_similarity(y_pred,y_true,axis=-2)

