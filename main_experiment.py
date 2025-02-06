#============================================================
#
#  Artifact reduction from ECG
#  Main experiment in the FGDAE paper
#
#  author: Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================


import matplotlib.pyplot as plt
from train_test import train_models, test_models, generate_results, generate_ecg_plot_results
import argparse
import warnings 
warnings.filterwarnings('ignore')

#use LaTeX like font styling, takes considerably longer for plotting
plt.rcParams.update({
        'text.usetex':True,
        'font.family':'Helvetica',
        'font.size':18
        })

#labels of the used methods
FILTERS = [
           'FIR', 
           'IIR',
           ]

EXPERIMENTS = [
              'CNN-DAE',            # CNN-DAE
              'DRNN',               # DRNN
              'FCN-DAE',            # FCN-DAE
              'Multibranch LANLD',  # DeepFilter
              'ECA Skip DAE',       # ACDAE
              'Attention Skip DAE', # CDAE-BAM
              'Transformer_DAE',    # TCDAE
              'Proposed_gatedONN1', # FGDAE (q=1)
              'Proposed_gatedONN2', # FGDAE (q=2)
              'Proposed_gatedONN3', # FGDAE (q=3)
              'Proposed_gatedONN4', # FGDAE (q=4)
              ]

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Unimodal ECG Denoising Benchmark")
    parser.add_argument("--Data", type=str, default="RMN1", help='Data type referring to the noise formation protocol. Options: RMN1, RMN2,BW, EM, MA, MN')
    parser.add_argument("--Mode", type=str, default="Train", help='Mode of the desired operation. Options: Train, Test, Eval')
    args = parser.parse_args()
    
    overlapping = True
    if args.Data == "BW":
        DATASET = '0_BW'                   #0_BW for baseline wander data (overlapping)
    if args.Data == "MA":
        DATASET = '1_MA'                   #1_MA for muscle artifact data (overlapping)
    if args.Data == "EM":
        DATASET = '2_EM'                   #2_EM for electrode motion data (overlapping)
    if args.Data == "MN":
        DATASET = '3_MN'                   #3_MN for totally mixied noise data (overlapping)
    if args.Data == "RMN1":
        DATASET = '4_RMN'                  #4_RMN for random mixed noise data (overlapping)
    if args.Data == "RMN2":
        DATASET = '5_RMN_nonoverlapping'   #5_RMN_nonoverlapping  for random mixed noise data (non-overlapping), mainly for plotting.
        overlapping=False
        
    if args.Mode == "Train":
        train_models(EXPERIMENTS, FILTERS, DATASET)
    if args.Mode == "Test":
        test_models(EXPERIMENTS, FILTERS, DATASET)
    if args.Mode == "Eval":
        generate_results(EXPERIMENTS, DATASET)
    if args.Mode == "Plot":
        generate_ecg_plot_results(EXPERIMENTS, DATASET, overlapping=overlapping)