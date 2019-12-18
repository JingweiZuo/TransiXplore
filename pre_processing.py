#this file handles the pre-processing for the data TransiXplore, to adapt the standard format for MTS processing

# Input file format:
# - train_meta: the meta data of MTS containing the labels and some extra informations (mean, mean_error, etc.)
# - train_pred: the MTS data without labelling, following the format (unnamed, object_id, time, single_value * 12 dimensions)

# Output file format:
# -(object_id, class, channel, length, channel_TS_data), separated into training/testing set

import pandas as pd
import numpy as np

dict = "/Users/Jingwei/Downloads/plasticc_part_data_201912/"
dict_meta = dict + "plasticc_train_meta.csv"
dict_pred = dict + "plasticc_train_pred.csv"
dict_pred_part = dict + "train_pred_part.csv"
meta_data = pd.read_csv(dict_meta)
train_pred = pd.read_csv(dict_pred)
train_pred_part = pd.read_csv(dict_pred_part)

full_data = train_pred.groupby(['object_id']).agg(lambda x: tuple(x))
full_data = full_data.drop(columns="Unnamed: 0").drop(columns="time")

#Add length of TS data, and change the name of columns
full_data = full_data.stack()
full_data = full_data.reset_index()
full_data.columns = ['object_id', 'channel', 'data']
full_data['length'] = full_data['data'].apply(lambda x: len(x))
full_data['data'] = full_data['data'].apply(lambda x: str(x)[1:-1].replace(" ", ""))

#Split the training/testing set
full_data.set_index("object_id")
meta_data = meta_data[['object_id', 'class']]
meta_data = meta_data.sample(frac=1.0)

cut_idx = int(round(0.2*meta_data.shape[0]))
meta_data_test, meta_data_train = meta_data.iloc[:cut_idx], meta_data.iloc[cut_idx:]

#join with the label file
train_data = full_data.merge(meta_data_train)
test_data = full_data.merge(meta_data_test)

train_data = train_data[['object_id', 'class', 'channel', 'length', 'data']]
test_data = test_data[['object_id', 'class', 'channel', 'length', 'data']]

np.savetxt(dict + r'train_data', train_data.values, fmt='%s', delimiter=',')
np.savetxt(dict + r'test_data', test_data.values, fmt='%s', delimiter=',')