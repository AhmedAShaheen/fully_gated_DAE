# Fully-Gated Denoising Auto-Encoder for Artifact Reduction in ECG Signals

You can access our paper here: [https://www.mdpi.com/1424-8220/25/3/801/htm](https://www.mdpi.com/1424-8220/25/3/801/htm)


## Citation
Please use this BibTeX entry for citing our work:

```bibtex
@Article{s25030801,
AUTHOR = {Shaheen, Ahmed and Ye, Liang and Karunaratne, Chrishni and Seppänen, Tapio},
TITLE = {Fully-Gated Denoising Auto-Encoder for Artifact Reduction in ECG Signals},
JOURNAL = {Sensors},
VOLUME = {25},
YEAR = {2025},
NUMBER = {3},
ARTICLE-NUMBER = {801},
URL = {https://www.mdpi.com/1424-8220/25/3/801},
ISSN = {1424-8220},
DOI = {10.3390/s25030801}
}
```

Download this git repository and run locally on Linux (or WSL if using Windows) for GPU support
```bash
git clone https://github.com/AhmedAShaheen/fully_gated_DAE.git
```


## Generating Datasets
To generate the main dataset used in training and testing, use the following command:  
```bash
python generate_data.py --Data "RMN1"
```

To generate the "non-overlapping" dataset used in ECG plotting, use the following command:
```bash
python generate_data.py --Data "RMN2"
```


## Training and testing
The models' weights obtained from our runs are already available in folder `Models`. 
However, if you need to re-train the deep learning models used in the experiment, use the `main_experiment.py` file and set the correct parameters. For example to run the same experiment in the paper, use the following python code in cmd.
```bash
python main_experiment.py --Data "RMN1" --Mode Train>> ./results/results.txt
```

If you run the previous command, the testing is already included. However, if you need to re-run the testing (e.g., to re-generate the results) you can use `main_experiment.py` as follows: 
- For overlapping data (The main dataset in the experiment):
```bash
python main_experiment.py --Data "RMN1" --Mode Test>> ./results/results.txt
```
- For non-overlapping data (The dataset used for ECG plotting):
```bash
python main_experiment.py --Data "RMN2" --Mode Test>> ./results/results.txt
```


## Generating results and plots
The results, tables, and plots are not generated automatically in the testing process.

To generate result tables and boxplots use command: 
```bash
python3 main_experiment.py --Data "RMN1" --Mode Eval>> ./results/results.txt
```

To generate ECG plots, use the following command:
```bash
python3 main_experiment.py --Data "RMN2" --Mode Eval>> ./results/results.txt
```

You can remove the `>> ./results/results.txt` if you don't want the output of the codes to be written (appended not overwritten) directly to `results.txt` file. However, it is useful to keep it.