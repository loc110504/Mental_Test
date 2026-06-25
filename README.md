# Try code AgentMental

## Quick Start

Clone this project locally
```bash
git clone https://github.com/MACLAB-HFUT/AgentMental.git
```

Create a new virtual environment, e.g. with conda or miniconda 
```bash
~$ conda create -n agentmental python>=3.10
```

Activate the environment:
```bash
~$ conda activate agentmental
```

Install the required packages:
```bash
~$ pip install -r requirements.txt
```

Replace your model, base_url and api_key with your actual model, base_url and API keys in files ```OAI_CONFIG_LIST, generate_response.py, memory.py```.

Run
```bash
~$ python main.py
```

## Data

Download the dataset from an authorized page or a manifest of direct links
```bash
~$ python data/data_download.py --url "<AUTHORIZED_DATASET_PAGE_URL>"
```

or
```bash
~$ python data/data_download.py --manifest data/download_urls.txt
```

Extract transcripts for a split after the archives are downloaded
```bash
~$ python data/extract_data.py \
    --split-csv /path/to/test_split.csv \
    --source-dir data/daic_woz_dataset \
    --extract-dir data/extracted_test \
    --transcript-dir data/test_transcripts
```

Convert the split into the JSON format used by the project
```bash
~$ python data/data_process.py \
    --score-csv /path/to/test_split.csv \
    --transcript-dir data/test_transcripts \
    --output-dir data/processed_test_daic_woz
```


Llamma-3.1-8b-instruct:
=== Depression Classification Evaluation ===
   Accuracy  Macro_F1  F1_Control  F1_Depression  Kappa
0     0.686     0.676       0.732          0.621  0.366

=== Item-level Evaluation ===
                         Item    MAE  Accuracy  Macro_F1
0            Loss of Interest  0.743     0.457     0.276
1              Depressed Mood  0.914     0.286     0.217
2              Sleep Problems  0.886     0.400     0.240
3       Fatigue or Low Energy  0.829     0.286     0.230
4  Appetite or Weight Changes  0.714     0.400     0.291
5              Low Self-Worth  0.914     0.429     0.314
6  Concentration Difficulties  1.029     0.343     0.202
7         Psychomotor Changes  0.743     0.543     0.223

LLama-3.3-70b-instruct
=== Depression Classification Evaluation ===
   Accuracy  Macro_F1  F1_Control  F1_Depression  Kappa
0     0.857     0.844       0.889            0.8  0.689

=== Item-level Evaluation ===
                         Item    MAE  Accuracy  Macro_F1
0            Loss of Interest  0.371     0.714     0.515
1              Depressed Mood  0.429     0.600     0.569
2              Sleep Problems  0.600     0.571     0.496
3       Fatigue or Low Energy  0.657     0.429     0.427
4  Appetite or Weight Changes  0.943     0.400     0.189
5              Low Self-Worth  0.714     0.400     0.363
6  Concentration Difficulties  0.714     0.429     0.387
7         Psychomotor Changes  0.429     0.629     0.476