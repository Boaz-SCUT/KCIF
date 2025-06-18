import json
import torch

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    processed_data = {}
    for patient_id, episodes in data.items():
        patient_episodes = []
        for episode in episodes:
            if not isinstance(episode[0], list):
                episode = [episode]
            episode_tensor = torch.tensor(episode, dtype=torch.float32)
            if episode_tensor.dim() == 1:
                episode_tensor = episode_tensor.unsqueeze(0)
            patient_episodes.append(episode_tensor)
        processed_data[patient_id] = patient_episodes
    return processed_data


def get_patient_embedding(model, patient_data, subject_id):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if subject_id not in patient_data:
        raise ValueError(f"Subject ID {subject_id} not found in the data.")
    patient_episodes = patient_data[subject_id]
    # Ensure patient_episodes are on the same device as the model
    patient_episodes = [episode.to(device) for episode in patient_episodes]  # {{ edit_1 }}
    patient_embedding = model(patient_episodes)
    return patient_embedding


# # 加载数据
train_data = load_data(f'data/mimic/train_processed_patient_data.json')
dev_data = load_data(f'data/mimic/dev_processed_patient_data.json')
test_data = load_data(f'data/mimic/test_processed_patient_data.json')
all_data = {**preprocess_data(train_data), **preprocess_data(dev_data), **preprocess_data(test_data)}


