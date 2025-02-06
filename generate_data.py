#============================================================
#
#  Deep Learning Artifact Filtering
#  Data Generation
#
#  author: Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================

import pickle
import argparse
import numpy as np
from Data_Preparation.data_preparation_overlapped import Data_Preparation

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Data generation for unimodal ECG denoising benchmark")
    parser.add_argument("--Data", type=str, default="RMN1", help='Data type referring to the noise formation protocol. Options: RMN1, RMN2,BW, EM, MA, MN')
    args = parser.parse_args()
    
    if args.Data == "BW":
        DATASET = '0_BW'      #0_BW for baseline wander data (overlapping)
        mode = args.Data
        overlapping = True
    if args.Data == "MA":
        DATASET = '1_MA'      #1_MA for muscle artifact data (overlapping)
        mode = args.Data
        overlapping = True
    if args.Data == "EM":
        DATASET = '2_EM'      #2_EM for electrode motion data (overlapping)
        mode = args.Data
        overlapping = True
    if args.Data == "MN":
        DATASET = '3_MN'      #3_MN for totally mixied noise data (overlapping)
        mode = args.Data
        overlapping = True
    if args.Data == "RMN1":
        DATASET = '4_RMN'      #4_RMN for random mixed noise data (overlapping)
        mode = 'RMN'
        overlapping = True
    if args.Data == "RMN2":
        DATASET = '5_RMN_nonoverlapping'      #5_RMN_nonoverlapping  for random mixed noise data (non-overlapping)
        mode = 'RMN'
        overlapping = False
        
    for n_type in [1,2]:    
        # using scale from 0.2 to 2 (reference has zero mean)
        Dataset = Data_Preparation(n_type, mode=mode, overlapping=overlapping)
        with open('./data/'+DATASET+'/Dataset_'+str(n_type)+'.pkl', 'wb') as output:
            pickle.dump(Dataset, output)