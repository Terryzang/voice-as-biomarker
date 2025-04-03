from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Read the Excel file
data_path = f'E:\\.xlsx'
data = pd.read_excel(data_path)

# Load the MiniRBT-H256 model and tokenizer
bert_path = 'E:\\MiniRBT_h256_pt'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path, return_dict=True).to(device)

# Store all extracted features
features_list = []

# Process data row by row
for index, row in data.iterrows():
    text = row['text']
    number = row['number']
    self_esteem = row['self-esteem']

    # Tokenize and move to GPU/CPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute CLS embedding
    with torch.no_grad():
        outputs = bert(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # Flatten to shape (256,)

    # Combine all information
    features_list.append([number] + cls_embedding.tolist() + [self_esteem])

    # Progress monitor
    if (index + 1) % 20 == 0:
        print(f"Processed  {index + 1} / {len(data)} texts")

# Convert to DataFrame
columns = ["number"] + [f"dim_{i+1}" for i in range(cls_embedding.shape[0])] + ["self-esteem"]
features_df = pd.DataFrame(features_list, columns=columns)

# Save to CSV
output_path = f'E:\\text_bert256_feature.csv'
features_df.to_csv(output_path, index=False)
