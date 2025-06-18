from torch.utils.data import Dataset, DataLoader
import json
import torch
import numpy as np

code_to_id = np.load('./data/icd2idx.npy', allow_pickle=True).item()
genders_to_id = {"M": 0, "F": 1}

def preprocess_data(patients):
    data = []
    for patient in patients:
        patient_data = []
        for visit in patient:
            visit_data = []
            for code in visit:
                visit_data.append(code_to_id[code])
            patient_data.append(torch.tensor(visit_data, dtype=torch.long))
        data.append(patient_data)
    return data

class PatientDataset(Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.read_example()
    def read_example(self):
        diagnosis = []
        labels = []
        genders = []
        ages = []
        with open(self.input_file, 'r') as f:
            for line in f.readlines():
                json_dic = json.loads(line)
                Diag_label = json_dic["labels"]["diag_label"]
                record_icd = json_dic["medical_records"]["record_icd"]
                gender = json_dic["other_info"]["gender"]
                age = json_dic["other_info"]["age"]
                race = json_dic["other_info"]["ethnicity"]
                diagnosis.append(record_icd)
                labels.append(Diag_label)
                genders.append(genders_to_id[gender])

                if age[0]>100:
                    ages.append(100)
                else:
                    ages.append(age[0])

            self.data = preprocess_data(diagnosis)
            genders = torch.tensor(genders, dtype=torch.long)
            ages = torch.tensor(ages, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.float)
            self.labels = labels
            self.ages = ages
            self.genders = genders

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.genders[idx], self.ages[idx], self.labels[idx]

# if __name__ == '__main__':
#     file_name = '../data/mimic/statement/dev.statement.jsonl'
#     d = PatientDataset(file_name)
#     d.read_example()