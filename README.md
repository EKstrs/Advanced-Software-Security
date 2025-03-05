# Advanced-Software-Security

**Video Explanation: https://www.youtube.com/watch?v=yROLDTA8hkQ**

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
- **Knowledge Base**: `knowledge_base.xlsx`
- **Scenario Data**: `Scenarios.xlsx`

Modify the file paths in `main()` if necessary:
```python
knowledge_file = "path/to/knowledge_base.xlsx"
file_path = "path/to/Scenarios.xlsx"
```

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


