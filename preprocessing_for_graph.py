import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/dataset.csv') #Thi file contains structured and unstructured data as columns.
#The unstructured data is present under column name 'clean_text'

#Demographics are binary encoded
TAB_COLS = ['patient_gender',
       'patient_race', 'patient_ethnicity', 'Patient_age', 'Charlson',
        '0_Congestive Heart Failure',
       '1_Cardiac Arrhythmias', '2_Valvular Disease',
       '3_Pulmonary Circulation Disorders', '4_Peripheral Vascular Disorders',
       '5_Hypertension, Uncomplicated', '6_Hypertension, Complicated',
       '7_Paralysis', '8_Other Neurological Disorders',
       '9_Chronic Pulmonary Disease', '10_Diabetes, Uncomplicated',
       '11_Diabetes, Complicated', '12_Hypothyroidism', '13_Renal Failure',
       '14_Liver Disease', '15_Peptic Ulcer Disease Excluding Bleeding',
       '16_AIDS/HIV', '17_Lymphoma', '18_Metastatic Cancer',
       '19_Solid Tumor Without Metastasis',
       '20_Rheumatoid Arthritis/Collagen Vascular Diseases', '21_Coagulopathy',
       '22_Obesity', '23_Weight Loss', '24_Fluid And Electrolyte Disorders',
       '25_Blood Loss Anemia', '26_Deficiency Anemia', '27_Alcohol Abuse',
       '28_Drug Abuse', '29_Psychoses', '30_Depression', 'find_E',
       'has_emergency','prev_deli_flag']  # replace with your column names


tabular_matrix = df[TAB_COLS].astype(float).values  # shape: [N, D]

dataset_name = 'delirium_fusion'

# Create directories if not exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/corpus', exist_ok=True)
os.makedirs('data/tabular', exist_ok=True)

# Reset index so rows have IDs 0,1,2,...
df = df.reset_index(drop=True)

# Format: id<TAB>split<TAB>label
meta_lines = []

for i, row in df.iterrows():
    meta_line = f"{i}\t{row['split_column']}\t{row['label']}"
    meta_lines.append(meta_line)

meta_data_str = "\n".join(meta_lines)

with open(f"data/{dataset_name}.txt", "w", encoding="utf-8") as f:
    f.write(meta_data_str)

# Each line is one clinical note
corpus_str = "\n".join(df["clean_text"].astype(str).tolist())

with open(f"data/corpus/{dataset_name}.txt", "w", encoding="utf-8") as f:
    f.write(corpus_str)

with open(f"data/tabular/{dataset_name}.tab", "w") as f:
    for row in tabular_matrix:
        row_str = " ".join([str(x) for x in row])
        f.write(row_str + "\n")

print("Preprocessing complete!")
