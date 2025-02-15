# Advanced-Software-Security

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `torch`
- `transformers`
- `pandas`
- `faiss-cpu`
- `sentence-transformers`
- `openpyxl`

Install dependencies using:
```sh
pip install torch transformers pandas faiss-cpu sentence-transformers openpyxl
```

## Usage
### Running the Model
To execute the script, run:
```sh
python main.py
```

### Input Data
When initialized, the ReEval class reads input data from config.json
When running first time, the program creates `config.json` template to your working directory, to avoid this you can add file `config.json` yourself with following data:
```python
{
    "input_file": "C:\\path\\to\\Scenarios.xlsx",
    "knowledge_file": "knowledge_base.xlsx",
    "output_file": "processed_scenarios.csv",
    "max_tokens_per_scenario": 128
}
```
To max_tokens_per_scenario please use preferred value

### Output
The script generates `processed_scenarios.csv`, which includes:
- **Scenario ID**
- **Threat ID**
- **Vulnerability ID**
- **Remediation Strategy**
- **Remediation Type** (Mandatory or Nice to Have - NTH)

### Example Output
```csv
Scenario ID, Threat ID, Vulnerability ID, Remediation Strategy, Remediation Type
1, T001, V001, "Implement strong authentication mechanisms", Mandatory
1, T001, V001, "Conduct regular security audits", Nice to Have (NTH)
```

---

## Results & Evaluation
- The model processes scenarios in batches.
- Results are stored in `results.csv`.
- **Performance metrics** such as accuracy, actionability, speed, and comprehensiveness are analyzed.


