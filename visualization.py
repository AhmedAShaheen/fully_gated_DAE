#============================================================
#
#  Deep Learning BLW Filtering
#  Data Visualization
#
#  author: Francisco Perdigon Romero, and Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy import stats
from scipy.stats import wilcoxon


np.random.seed(1234)

def generate_violinplots(np_data, description, ylabel, log):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)
    pd_df = pd_df.applymap(lambda x: x[0].squeeze() if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=pd_df, palette="Set3", bw=.2, cut=1, linewidth=1)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel=r'$Models\ /\ Methods$', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    plt.show()
    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def generate_barplot(np_data, description, ylabel, log, metric, Dataset):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)
    pd_df = pd_df.applymap(lambda x: x[0].squeeze() if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.barplot(data=pd_df)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel=r'$Models\ /\ Methods$', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    plt.tight_layout()
    savename=f'results/{Dataset}/barplot_{metric}.png'
    plt.savefig(savename, dpi=450)
    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def generate_hbarplot(np_data, description, ylabel, log, metric, Dataset):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)
    pd_df = pd_df.applymap(lambda x: x[0].squeeze() if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    # Set up the matplotlib figure
    f, ax = plt.subplots()
    sns.set(style="whitegrid")
    ax = sns.barplot(data=pd_df, orient="h")
    if log:
        ax.set_yscale("log")
    ax.set(xlabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    plt.tight_layout()
    savename=f'results/{Dataset}/hbarplot_{metric}.png'
    plt.savefig(savename, dpi=450)
    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def generate_boxplot(np_data, description, ylabel, log, metric, Dataset, set_y_axis_size=None):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)
    pd_df = pd_df.applymap(lambda x: x[0].squeeze() if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 8))
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=pd_df)
    plt.xticks(rotation=90)
    if log:
        ax.set_yscale("log")
    if set_y_axis_size != None:
        ax.set_ylim(set_y_axis_size)
    ax.set(ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    plt.tight_layout()
    savename=f'results/{Dataset}/boxplot_{metric}.png'
    plt.savefig(savename, dpi=450)
    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def generate_hboxplot(np_data, description, ylabel, log, metric, Dataset, set_x_axis_size=None):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)
    pd_df = pd_df.applymap(lambda x: x[0].squeeze() if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    # Set up the matplotlib figure
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(8, 5))
    ax = sns.boxplot(data=pd_df, orient="h", width=0.4)
    if log:
        ax.set_xscale("log")
    if set_x_axis_size != None:
        ax.set_xlim(set_x_axis_size)
    ax.set(xlabel=ylabel)
    ax = sns.despine(left=True, bottom=True)
    plt.tight_layout()
    savename=f'results/{Dataset}/hboxplot_{metric}.png'
    plt.savefig(savename, dpi=450)
    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def ecg_view(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg_blw, 'k', label='ECG + BLW')
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.grid(True)
    plt.ylabel('au')
    plt.xlabel('samples')
    leg = ax.legend()
    if signal_name != None and beat_no != None:
        plt.title('Signal ' + str(signal_name) + 'beat ' + str(beat_no))
    else:
        plt.title('ECG signal for comparison')
    plt.show()

def ecg_view_diff(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.plot(ecg - ecg_dl, color='#0099ff', lw=3, label='Difference ECG - DL Filter')
    plt.plot(ecg - ecg_f, color='#cb828d', lw=3, label='Difference ECG - IIR Filter')
    plt.grid(True)
    plt.ylabel('Amplitude (au)')
    plt.xlabel('samples')
    leg = ax.legend()
    if signal_name != None and beat_no != None:
        plt.title('Signal ' + str(signal_name) + 'beat ' + str(beat_no))
    else:
        plt.title('ECG signal for comparison')
    plt.show()

def generate_table(metrics, metric_values, Exp_names, pvalue=False):
    # Print tabular results in the console, in a pretty way (latex code of the table)
    tb = PrettyTable(border=False)
    ind = 0
    tb.field_names = ['Method/Model'] + metrics + ['\\\\ \\hline']
    for exp_name in Exp_names:
        tb_row = []
        tb_row.append(exp_name)
        for metric in metric_values:   # metric_values[metric][model][beat]
            if isinstance(metric[ind], np.float64) or isinstance(metric[ind], str):
                tb_row.append('& ${}$'.format(metric[ind]))
            else:
                m_mean = np.mean(metric[ind])
                m_std = np.std(metric[ind])
                tb_row.append('& ${:.3f}'.format(m_mean) + ' \pm ' + '{:.3f}$'.format(m_std))
                if pvalue == True: 
                    if np.sum(np.array(metric[10])-np.array(metric[ind]))==0: # 10 is the index of the proposed model
                        tb_row.append('& $-$')
                    else:
                        w_statistic, p_value1 = wilcoxon(metric[10], metric[ind])
                        tb_row.append('& ${}$'.format(p_value1.squeeze()))
                    if np.sum(np.array(metric[11])-np.array(metric[ind]))==0: # 11 is the index of the proposed model
                        tb_row.append('& $-$')
                    else:
                        w_statistic, p_value2 = wilcoxon(metric[11], metric[ind])
                        tb_row.append('& ${}$'.format(p_value2.squeeze()))
        tb_row.append('\\\\ \\hline')
        tb.add_row(tb_row)
        ind += 1
    print()
    print(tb)
    print()
    

def generate_table_time(column_names, all_values, Exp_names, gpu=True):
    # Print tabular results in the console, in a pretty way (latex code of the table)
    tb = PrettyTable()
    ind = 0
    if gpu:
        device = 'GPU'
    else:
        device = 'CPU'
    for exp_name in Exp_names:
        tb.field_names = ['Method/Model'] + [column_names[0] + '(' + device + ') h:m:s:ms'] + [
            column_names[1] + '(' + device + ') h:m:s:ms']
        tb_row = []
        tb_row.append(exp_name)
        tb_row.append(all_values[0][ind])
        tb_row.append(all_values[1][ind])
        tb.add_row(tb_row)
        ind += 1
    print(tb)
    if gpu:
        print('* For FIR and IIR Filters is CPU since scipy filters are CPU based implementations')

def ecg_plot(values, legend_info, label=None, **kwargs):
    original_ecg = kwargs.get('original', None)
    fig, ax = plt.subplots(figsize=(6, 4))
    if original_ecg is not None:
        plt.plot(original_ecg, label=r'$Original\ ECG$')
    plt.plot(values, label=label)
    for info in legend_info:
        plt.plot([],[], ' ', label=info)
    plt.grid(True)
    plt.ylabel(r'$Amplitude\ (au)$')
    plt.xlabel(r'$samples$')
    plt.xlim(0,4095) #assuming 8 segments of size 512 samples.
    if label is not None:
        plt.legend(loc='upper right')
    plt.title(kwargs.get('title', ''))
    plt.tight_layout()
    savename = kwargs.get('savename', '')
    if savename:
        plt.savefig(savename, dpi=450)
    return