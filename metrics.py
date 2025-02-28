import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from modeling.transformer import TransformerTime
from modeling.units import adjust_input_hita
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import warnings
from modeling.modeling_seqG import get_seqG_embedding
from utils.obtain_lab_data import all_data
warnings.filterwarnings("ignore", category=UserWarning)

def get_patient_embedding(model, patient_data, subject_id):
        if subject_id not in patient_data:
            # Return a default embedding of shape [1, 128] if subject_id is not found
            return torch.randn(1, 128)
        patient_episodes = patient_data[subject_id]
        patient_episodes = [episode for episode in patient_episodes]
        patient_embedding = model(patient_episodes)
        # Ensure the output is of shape [1, 128]
        return patient_embedding.view(1, 128)
    
def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        # if len(t)==0:
        #     len_t = len(t)+100000000
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len_t)
            try:
                r[i] += len(it) / len(t)
            except:
                r[i] += len(it) / 10000000
    return a / len(y_true_hot), r / len(y_true_hot)


def calculate_occurred(historical, y, preds, ks):
    # y_occurred = np.sum(np.logical_and(historical, y), axis=-1)
    # y_prec = np.mean(y_occurred / np.sum(y, axis=-1))
    r1 = np.zeros((len(ks),))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        # n_k = np.minimum(n, k)
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        # pred_occurred = np.sum(np.logical_and(historical, pred_k), axis=-1)
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2

def evaluate_hf(eval_set, model,lab_model,seqGmodel, name='eval' ):
    model.eval()
    # labels = dataset.HF_labels
    outputs = []
    preds = []
    labels = []
    loss_fn = torch.nn.BCELoss()

    with torch.no_grad():
        
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            
            Hf_label = Hf_labels.float()
            
            valid_indices = [i for i, qid in enumerate(qids) if str(qid) in all_data]
            if not valid_indices:
                continue  
            
            missing_subjects = len(qids) - len(valid_indices)

            filtered_qids = [qids[i] for i in valid_indices]
            filtered_Hf_labels = Hf_label[valid_indices]
            filtered_simPatients = None
            filtered_main_codes = main_codes[valid_indices]
            filtered_sub_codes1 = sub_codes1[valid_indices]
            filtered_sub_codes2 = sub_codes2[valid_indices]
            filtered_input_data = [x[valid_indices] if isinstance(x, torch.Tensor) else [x[i] for i in valid_indices] for x in input_data]

            lab_embeddings = []
            seqG_embeddings = []
            edge_info = {}  
            for qid in filtered_qids:
                seqG_embe, edge_index, edge_weight = get_seqG_embedding(qid, seqGmodel) # [128]
                edge_info[qid] = {
                    'edge_index': edge_index,
                    'edge_weight': edge_weight
                }
                subject_id = str(qid)
                lab_embedding = get_patient_embedding(lab_model, all_data, subject_id)
                lab_embeddings.append(lab_embedding.squeeze())
                seqG_embeddings.append(seqG_embe)
            lab_embeddings = torch.stack(lab_embeddings, dim=0)
            seqG_embeddings = torch.stack(seqG_embeddings, dim=0)
            labels.append(filtered_Hf_labels.cpu().numpy())
            contrastive_loss,logits, attn, (edge_idx, edge_weight), visit_att, self_att = model(
                'test', lab_embeddings, seqG_embeddings,
                filtered_simPatients, filtered_main_codes, filtered_sub_codes1, filtered_sub_codes2, *filtered_input_data,
                return_P_emb=False, return_emb=False, simp_emb=None, use_graph=False)

            output = logits.detach().cpu().numpy()
            pred = (output > 0.5).astype(int)
            preds.append(pred)
            outputs.append(output)
        outputs = np.concatenate(outputs)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        auc = roc_auc_score(labels, outputs)
        f1_score_ = f1_score(labels, preds)
        return f1_score_, auc


def write_file(file_name, input_text):
    with open(file_name, "a") as file:
        file.write(input_text)

import numpy
def convert_tensor_to_list(obj):
    """递归地将tensor转换为list"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_list(item) for item in obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    return obj

def evaluate_codes(eval_set, model,lab_model,seqGmodel, name='eval'):
    model.eval()    
    preds = []
    labels = []
    with torch.no_grad():
        for qids, main_diagnoses, simPatients, Hf_labels, Diag_labels, main_codes, sub_codes1, sub_codes2, *input_data in tqdm(
                eval_set, desc=name):
            valid_indices = [i for i, qid in enumerate(qids) if str(qid) in all_data]
            if not valid_indices:
                continue  

            filtered_qids = [qids[i] for i in valid_indices]
            filtered_Diag_labels = Diag_labels[valid_indices]
            filtered_simPatients = None
            filtered_main_codes = main_codes[valid_indices]
            filtered_sub_codes1 = sub_codes1[valid_indices]
            filtered_sub_codes2 = sub_codes2[valid_indices]
            filtered_input_data = [x[valid_indices] if isinstance(x, torch.Tensor) else [x[i] for i in valid_indices] for x in input_data]

            lab_embeddings = []
            seqG_embeddings = []
            edge_info = {}  
            all_results = []
            for qid in filtered_qids:
                seqG_embe, edge_index, edge_weight = get_seqG_embedding(qid, seqGmodel) # [128]
                edge_info[qid] = {
                    'edge_index': convert_tensor_to_list(edge_index),
                    'edge_weight': convert_tensor_to_list(edge_weight)
                }
                subject_id = str(qid)
                lab_embedding = get_patient_embedding(lab_model, all_data, subject_id)
                lab_embeddings.append(lab_embedding.squeeze())
                seqG_embeddings.append(seqG_embe)
            lab_embeddings = torch.stack(lab_embeddings, dim=0)
            seqG_embeddings = torch.stack(seqG_embeddings, dim=0)
            
            filtered_Diag_labels = filtered_Diag_labels.float()
            labels.append(filtered_Diag_labels)
            contrastive_loss,logits, attn, (edge_idx, edge_weight), visit_att, self_att = model(
                'test', lab_embeddings, seqG_embeddings,
                filtered_simPatients, filtered_main_codes, filtered_sub_codes1, filtered_sub_codes2, *filtered_input_data,
                return_P_emb=False, return_emb=False, simp_emb=None, use_graph=False)
            
            pred = torch.argsort(logits, dim=-1, descending=True)
            preds.append(pred)
            
            for i, qid in enumerate(filtered_qids):
                result = {
                    'qid': str(qid),
                    'edge_info': edge_info[qid],
                    'pred': pred[i].detach().cpu().numpy().tolist(),
                    'label': filtered_Diag_labels[i].detach().cpu().numpy().tolist()
                }
                all_results.append(result)

        preds = torch.vstack(preds).detach().cpu().numpy()
        labels = torch.vstack(labels).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
        
        import json
        output_file = f'results_{name}.json'
        
        try:
            with open(output_file, 'w') as f:
                json.dump(convert_tensor_to_list(all_results), f, indent=2)
            print(f"Results saved to {output_file}")
            
        except TypeError as e:
            print(f"序列化错误: {e}")
            for result in all_results:
                for key, value in result.items():
                    print(f"Key: {key}, Type: {type(value)}")
            
        return f1_score, recall,