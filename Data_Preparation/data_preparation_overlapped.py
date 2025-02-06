#============================================================
#
#  Data preparation for Random-Mixed Noise 
#
#  author: Francisco Perdigon Romero, Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================

import re
import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB

def Data_Preparation(noise_version=1, mode='RMN', overlapping=True):

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level across runs
    # this enhance reproducibility but does not guarantee 100%.
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseBWL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    ma_signals = np.array(ma_signals)
    em_signals = np.array(em_signals)
    
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]

    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]
    
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]


    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test_1 = bw_noise_channel2_b
        noise_test_2 = ma_noise_channel2_b
        noise_test_3 = em_noise_channel2_b
        noise_train_1 = bw_noise_channel1_a
        noise_train_2 = ma_noise_channel1_a
        noise_train_3 = em_noise_channel1_a
    elif noise_version == 2:
        noise_test_1 = bw_noise_channel1_b
        noise_test_2 = ma_noise_channel1_b
        noise_test_3 = em_noise_channel1_b
        noise_train_1 = bw_noise_channel2_a
        noise_train_2 = ma_noise_channel2_a
        noise_train_3 = em_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################

    
    # QTDatabese signals Dataset splitting. Considering the following link
    # https://www.physionet.org/physiobank/database/qtdb/doc/node3.html
    #  Distribution of the 105 records according to the original Database.
    #  | MIT-BIH | MIT-BIH |   MIT-BIH  |  MIT-BIH  | ESC | MIT-BIH | Sudden |
    #  | Arrhyt. |  ST DB  | Sup. Vent. | Long Term | STT | NSR DB	| Death  |
    #  |   15    |   6	   |     13     |     4     | 33  |  10	    |  24    |
    #
    # The two random signals of each pathology will be keep for testing set.
    # The following list was used
    # https://www.physionet.org/physiobank/database/qtdb/doc/node4.html
    # Selected test signal amount (14) represent ~13 % of the total

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
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

    
    samples = 512
    
    # concatenate the signal segments
    beats_train0 = []
    beats_test0 = []
    name_test0 = []
    qtdb_keys = list(qtdb.keys())
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        test_name = re.search('(?<=\\\).*', signal_name).group(0)
        sig = np.concatenate(qtdb[signal_name])
        if test_name in test_set:
            beats_test0.append(sig)
            name_test0.append(test_name)
        else:
            beats_train0.append(sig)
    
    
    # sample the signal with overlapping window
    if overlapping:
        overlap = samples//2
    else:
        overlap = samples
    beats_train = []
    beats_test = []
    name_test = []
    for signal in beats_train0:
        for i in range(0, len(signal)-samples+1, overlap):
            beats_train.append(signal[i:i+samples]-np.mean(signal[i:i+samples])) #Force each segment to have zero mean
    for signal, name in zip(beats_test0, name_test0):
        for i in range(0, len(signal)-samples+1, overlap):
            beats_test.append(signal[i:i+samples]-np.mean(signal[i:i+samples]))  #Force each segment to have zero mean
            name_test.append(name)
    
    
    # noise scales 
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    
    # Adding noise to train
    sn_train = []
    noise_index = 0
    
    if mode == 'BW':
        mask_train = np.ones((len(beats_train),3)) * np.array([1,0,0])# mixed noise mask
    elif mode == 'MA':
        mask_train = np.ones((len(beats_train),3)) * np.array([0,1,0])# mixed noise mask
    elif mode == 'EM':
        mask_train = np.ones((len(beats_train),3)) * np.array([0,0,1])# mixed noise mask
    elif mode == 'MN':
        mask_train = np.ones((len(beats_train),3))# mixed noise mask
    elif mode == 'RMN':
        mask_train = np.random.randint(0, 2, size=(len(beats_train),3))# random mixed noise mask
    else:
        raise ValueError("mode argument must be one of: 'BW', 'MA', 'EM', 'MN', or 'RMN'")
    
    print('Training data shape: ' + str(rnd_train.shape))
    for i in range(len(beats_train)):
        noise_train = mask_train[i,0]*noise_train_1 + mask_train[i,1]*noise_train_2 + mask_train[i,2]*noise_train_3 
        if np.sum(mask_train[i,:]) != 0:
            noise = noise_train[noise_index:noise_index + samples]
            beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
            noise_max_value = np.max(noise) - np.min(noise)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_train[i] / Ase
            
            signal_noise = beats_train[i] + alpha * noise
        else:
            signal_noise = beats_train[i]
            rnd_train[i] = 0
            
        sn_train.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_train) - samples):
            noise_index = 0
            

    # Adding noise to test
    sn_test = []
    noise_index = 0
    
    if mode == 'BW':
        mask_test = np.ones((len(beats_test),3)) * np.array([1,0,0])# mixed noise mask
    elif mode == 'MA':
        mask_test = np.ones((len(beats_test),3)) * np.array([0,1,0])# mixed noise mask
    elif mode == 'EM':
        mask_test = np.ones((len(beats_test),3)) * np.array([0,0,1])# mixed noise mask
    elif mode == 'MN':
        mask_test = np.ones((len(beats_test),3))# mixed noise mask
    elif mode == 'RMN':
        mask_test = np.random.randint(0, 2, size=(len(beats_test),3))# random mixed noise mask
    else:
        raise ValueError("mode argument must be one of: 'BW', 'MA', 'EM', 'MN', or 'RMN'")
    
    print('Test data shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise_test = mask_test[i,0]*noise_test_1 + mask_test[i,1]*noise_test_2 + mask_test[i,2]*noise_test_3 
        if np.sum(mask_test[i,:]) != 0:
            noise = noise_test[noise_index:noise_index + samples]
            beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
            noise_max_value = np.max(noise) - np.min(noise)
            Ase = noise_max_value / beat_max_value
            alpha = rnd_test[i] / Ase
            
            signal_noise = beats_test[i] + alpha * noise
        else:
            signal_noise = beats_test[i]
            rnd_test[i] = 0
            
        sn_test.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_test) - samples):
            noise_index = 0

    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    Dataset = [X_train, y_train, X_test, y_test, rnd_test, mask_test, name_test]

    print("Training data shape: ", X_train.shape)
    print("Testing data shape: ", X_test.shape)
    print('Dataset ready to use.')
    
    return Dataset