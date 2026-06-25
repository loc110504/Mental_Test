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
