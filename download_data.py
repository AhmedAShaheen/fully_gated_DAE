#============================================================
#
#  Artifact reduction from ECG
#  Download datasets
#
#  author: Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================

import os
import zipfile
import argparse
import numpy as np
from Data_Preparation.data_preparation import Data_Preparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unimodal ECG Denoising Benchmark")
    parser.add_argument("--sys", type=str, default="linux", help='Operating system used to generate the data. Options: linux and windows')
    args = parser.parse_args()

	os.system('echo "Downloading data from Physionet servers ..."')
	os.system('python -m wget https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip')
	os.system('python -m wget https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip')
	os.system('mkdir data')

	os.system('echo "Extracting data ..."')
	with zipfile.ZipFile('qt-database-1.0.0.zip', 'r') as zip_ref:
		zip_ref.extractall('./data')
		zip_ref.close()

	with zipfile.ZipFile('mit-bih-noise-stress-test-database-1.0.0.zip', 'r') as zip_ref:
		zip_ref.extractall('./data')
		zip_ref.close()

	os.system('echo "Removing zip files..."')
	if args.Data == "windows"
        os.system('del qt-database-1.0.0.zip')
        os.system('del mit-bih-noise-stress-test-database-1.0.0.zip')
    else: 
        os.system('rm qt-database-1.0.0.zip')
        os.system('rm mit-bih-noise-stress-test-database-1.0.0.zip')
	os.system('echo "Data download and moving done."')