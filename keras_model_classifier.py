"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import pickle
import keras 
from keras import layers
from keras import models
from keras import callbacks
from keras import backend as K
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# dffeats=pickle.load(open("./tables_1.anon_df_fintrans.rar","rb"))

dffeats=pd.read_pickle("./tables_1.anon_df_fintrans.rar")
print(dffeats.head())


# def main():


# if __name__=="__main__":
#   main()
print("running....")



target_col=["bucket"]
num_cols=['lead_time','NIGHTS','RANK','CHECKIN_DAY', 'CHECKIN_MONTH','delta_days_1', 'delta_days_2', 'delta_days_3', 'num_resorts_stayed','delta_std', 'delta_mean', 'delta_max', 'delta_min', 'delta_median']
num_instance_moneycols=['PACKAGE_FOOD_REVENUE','TOTAL_REVENUE','PACKAGE_ROOM_REVENUE','FOOD_REVENUE']
categorical_cols=['VIP_STATUS','NATIONALITY','ROOM_CATEGORY', 'MEMBERSHIP_ID', 'SOURCE_CODE', 'INSTANCE', 'CHANNEL','RESORT', 'ROOM_CLASS','WALKIN_YN','REGION_CODE','MARKET_CODE',"INSTANCE_COUNTRY"]
ids=["NAME_ID",'group','RESV_NAME_ID',]
numeric_cols=num_cols+num_instance_moneycols

y=pd.get_dummies(dffeats[target_col[0]])

X_train, X_test, y_train, y_test = train_test_split(dffeats[categorical_cols+numeric_cols].values, y.values, test_size=0.30,random_state=314,stratify=y.values)


cat_inputs = []
num_inputs = []
embeddings = []
embedding_layer_names = []

for col in categorical_cols:
    emb_n=int(min(np.ceil((dffeats[col].nunique())/2), 50))

    _input = layers.Input(shape=[1], name=col)
    _embed = layers.Embedding(dffeats[col].max() + 1, emb_n, name=col+'_emb')(_input)
    cat_inputs.append(_input)
    embeddings.append(_embed)
    embedding_layer_names.append(col+'_emb')
    
# Simple inputs for the numeric features
for col in numeric_cols:
    numeric_input = layers.Input(shape=(1,), name=col)
    num_inputs.append(numeric_input)
    
# Merge the numeric inputs
merged_num_inputs = layers.concatenate(num_inputs)
#numeric_dense = layers.Dense(20, activation='relu')(merged_num_inputs)

# Merge embedding and use a Droput to prevent overfittting
merged_inputs = layers.concatenate(embeddings)
spatial_dropout = layers.SpatialDropout1D(0.2)(merged_inputs)
flat_embed = layers.Flatten()(spatial_dropout)

# Merge embedding and numeric features
all_features = layers.concatenate([flat_embed, merged_num_inputs])

# MLP for classification
x = layers.Dense(300, activation="relu")(all_features)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(300, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)

y = layers.Dense(12, activation="softmax")(x)

model = models.Model(inputs=cat_inputs + num_inputs, outputs=y)
model.compile(loss='categorical_crossentropy', optimizer='adam')


print(model.summary())


# TB Callback
log_folder = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
tb_callback = callbacks.TensorBoard(
    log_dir=os.path.join('tb-logs', log_folder),
)

# Best model callback
bm_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join('tb-logs', log_folder, 'bm.h5'),
    save_best_only=True,
    save_weights_only=False
)
es = callbacks.EarlyStopping(min_delta=0.001, patience=5,
                                 verbose=1,  baseline=None)

rlr = callbacks.ReduceLROnPlateau( factor=0.5,
                                     patience=3, min_lr=1e-6, mode='max', verbose=1)
def get_keras_dataset(array):
    exp={str(col): array[:,i] for i,col in enumerate(categorical_cols+numeric_cols)}
    return exp


EPOCHS=1
BATCH_SIZE=32
CLASS_WEIGHTS=None

history= model.fit(
    x=get_keras_dataset(X_train),
    y=y_train,
    validation_data=(get_keras_dataset(X_test),y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=CLASS_WEIGHTS,
    callbacks=[tb_callback, bm_callback,es,rlr],
    verbose=2
)
def plot_history(history):
    fig = plt.figure(figsize=(15,8))
    ax = plt.subplot(211)
    
    plt.xlabel('Epoch')
    plt.ylabel('loss, acc')
    
    # Losses
    ax.plot(history.epoch, history.history['loss'], label='Train LOSS')
    ax.plot(history.epoch, history.history['val_loss'], label='Val LOSS')
    # ax.plot(history.epoch, history.history['acc'], label ='Train Accuracy')
    # ax.plot(history.epoch, history.history['val_acc'], label='Val Accuracy')
    plt.legend()
    
    # Plot the learning_rate
    if 'lr' in history.history:
        ax = plt.subplot(212)
        plt.ylabel('Learning rate')
        ax.plot(history.epoch, history.history['lr'], label='learning_rate')
        plt.legend()
    plt.show()
    plt.close(fig)

plot_history(history)

model = keras.models.load_model(os.path.join('tb-logs', log_folder, 'bm.h5'), compile=False)
pred = np.argmax(model.predict(get_keras_dataset(X_test)),axis=1)

print(accuracy_score(np.argmax(y_test,axis=1), pred))
print(classification_report(np.argmax(y_test,axis=1), pred))