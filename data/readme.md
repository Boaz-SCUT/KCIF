This project uses the following medical datasets and standards:

# Data Sources

- **MIMIC-III & MIMIC-IV**  
Medical Information Mart for Intensive Care, containing de-identified clinical data from intensive care patients

- **SNOMED CT**  
Systematized Nomenclature of Medicine Clinical Terms, providing standardized medical concepts and terminology systems

# Data Acquisition Instructions

⚠️ **Important Notice**: Due to copyright and privacy protection requirements, this project does not include original data files.

**Acquisition Steps**:

1. **MIMIC Dataset**
 - Visit the official [PhysioNet](https://physionet.org/) website
 - Complete the required training courses and data use agreements
 - Apply for and obtain data access permissions

2. **SNOMED CT**
 - Visit the official [SNOMED International](https://www.snomed.org/) website
 - Register an account and apply for free usage license
 - Download the terminology set for the corresponding language

After downloading the data, please place the files in the following directory structure:

```
data/
├── mimic-iii/
│   ├── ADMISSIONS.csv
│   ├── PATIENTS.csv
│   └── ...
├── mimic-iv/
│   ├── admissions.csv
│   ├── patients.csv
│   └── ...
└── snomed/
```
