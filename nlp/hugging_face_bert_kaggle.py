# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# /kaggle/input/google-quest-challenge/sample_submission.csv
# /kaggle/input/google-quest-challenge/test.csv
# /kaggle/input/google-quest-challenge/train.csv
# /kaggle/input/bert-base-uncased-huggingface-transformer/bert-base-uncased-tf_model.h5
# /kaggle/input/bert-base-uncased-huggingface-transformer/bert-base-uncased-vocab.txt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,f1_score
from  sklearn.model_selection import KFold 
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk import ngrams
import nltk
from nltk import word_tokenize,tokenize

np.set_printoptions(suppress=True)
print(tf.__version__)

train=pd.read_csv("../input/google-quest-challenge/train.csv")#.head(500)
test=pd.read_csv("../input/google-quest-challenge/test.csv")#.head(500)
tokenizer = BertTokenizer.from_pretrained('../input/bert-base-uncased-huggingface-transformer/bert-base-uncased-vocab.txt')

train["subdomain"]=train["question_user_page"].apply(lambda x: x.split(".")[0].replace("https://",""))
train["domain"]=train["question_user_page"].apply(lambda x: x.split(".")[1].replace("https://",""))
test["subdomain"]=test["question_user_page"].apply(lambda x: x.split(".")[0].replace("https://",""))
test["domain"]=test["question_user_page"].apply(lambda x: x.split(".")[1].replace("https://",""))

class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
            self.model.save_weights(self.model_name)
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

ids=["qa_id"]
targets=['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']
feats=[ 'question_title', 'question_body', 'question_user_name',
       'question_user_page', 'answer', 'answer_user_name', 'answer_user_page',
       'url', 'category', 'host']
text_feats=['question_title', 'question_body','answer']
NFOLDS=10
kf = KFold(n_splits=NFOLDS)
kf.get_n_splits(train.qa_id)

for train_index, test_index in kf.split(train):
    # print("TRAIN:", train_index, "TEST:", test_index)
    qt_train=train.loc[train_index,"question_title"].values
    qb_train=train.loc[train_index,'question_body'].values
    aw_train=train.loc[train_index,'answer'].values
    qt_valid=train.loc[test_index,"question_title"].values
    qb_valid=train.loc[test_index,'question_body'].values
    aw_valid=train.loc[test_index,'answer'].values
    
    y_train=train.loc[train_index,targets].values
    y_valid=train.loc[test_index,targets].values
    break
qt_test=test["question_title"].values
qb_test=test["question_body"].values
aw_test=test["answer"].values



max_len = 100

def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    input_ids_t, input_masks_t, input_segments_t = return_id(
        title, None, 'longest_first', max_sequence_length)
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, None, 'longest_first', max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, 'longest_first', max_sequence_length)
    
    return [input_ids_t, input_masks_t, input_segments_t, 
            input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns=text_feats, tokenizer=tokenizer, max_sequence_length=max_len):
    input_ids_t, input_masks_t, input_segments_t = [], [], []
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_t, masks_t, segments_t,ids_q, masks_q, segments_q, ids_a, masks_a, segments_a = \
        _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_t.append(ids_t)
        input_masks_t.append(masks_t)
        input_segments_t.append(segments_t)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_t, dtype=np.int32), 
                np.asarray(input_masks_t, dtype=np.int32), 
                np.asarray(input_segments_t, dtype=np.int32),
                np.asarray(input_ids_q, dtype=np.int32), 
                np.asarray(input_masks_q, dtype=np.int32), 
                np.asarray(input_segments_q, dtype=np.int32),
                np.asarray(input_ids_a, dtype=np.int32), 
                np.asarray(input_masks_a, dtype=np.int32), 
                np.asarray(input_segments_a, dtype=np.int32)]

def compute_catkpis(df):
    
    encd=LabelEncoder()
    domain=encd.fit_transform(df["domain"].values.reshape(-1,1))
    encs=LabelEncoder()
    sdomain=encs.fit_transform(df["subdomain"].values.reshape(-1,1))
    encc=LabelEncoder()
    cats=encc.fit_transform(df["category"].values.reshape(-1,1))
    print(np.transpose(np.array([domain,sdomain,cats])).shape)
    return list(np.array([domain,sdomain,cats]))


def compute_numkpis(df):
    def count_punctuations_perword(x):
        tokens=word_tokenize(x)
        puncts=[t for t in tokens if not t.isalnum()]
        try:
            return len(puncts)/len(tokens)
        except:
            return 1
    tnum_punct=df["question_title"].apply(count_punctuations_perword).values
    qnum_punct=df["question_body"].apply(count_punctuations_perword).values
    anum_punct=df["answer"].apply(count_punctuations_perword).values
    print(np.transpose(np.array([tnum_punct,qnum_punct,anum_punct])).shape)
    return list(np.array([tnum_punct,qnum_punct,anum_punct]))

def create_model(catcols,numcols):
    t_id = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_id = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    t_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    
    t_atn = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_atn = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
    # pretrained model has been downloaded manually and uploaded to kaggle. 
    bert_model = TFBertModel.from_pretrained('../input/bert-base-uncased-huggingface-transformer/bert-base-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    t_embedding = bert_model(t_id, attention_mask=t_mask, token_type_ids=t_atn)[0]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]


    
    t = tf.keras.layers.GlobalAveragePooling1D()(t_embedding)
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)

    data=train.append(test)
    emb_n=int(min(np.ceil((data["domain"].nunique())/2), 20))
    cat1_input = tf.keras.Input(shape=[1], name="domain")
    cat1_embed = tf.keras.layers.Embedding(data["domain"].nunique()+1 , emb_n, name="domain"+'_emb')(cat1_input)
    emb_n=int(min(np.ceil((data["subdomain"].nunique())/2), 20))
    cat2_input = tf.keras.Input(shape=[1], name="subdomain")
    cat2_embed = tf.keras.layers.Embedding(data["subdomain"].nunique()+1 , emb_n, name="subdomain"+'_emb')(cat2_input)
    emb_n=int(min(np.ceil((data["category"].nunique())/2), 20))
    cat3_input = tf.keras.Input(shape=[1], name="category")
    cat3_embed = tf.keras.layers.Embedding(data["category"].nunique()+1 , emb_n, name="category"+'_emb')(cat3_input)
    merged_cat_embeds=tf.keras.layers.Concatenate()([cat1_embed, cat2_embed, cat3_embed])
    cat_spatial_dropout = tf.keras.layers.SpatialDropout1D(0.3)(merged_cat_embeds)
    cat_flat_embed = tf.keras.layers.Flatten()(cat_spatial_dropout)
    
    numeric_input1 = tf.keras.layers.Input(shape=(1,))
    numeric_input2 = tf.keras.layers.Input(shape=(1,))
    numeric_input3 = tf.keras.layers.Input(shape=(1,))
    merged_num_inputs=tf.keras.layers.Concatenate()([ numeric_input1, numeric_input2, numeric_input3])
    num_spatial_dropout = tf.keras.layers.Dropout(0.3)(merged_num_inputs)
    num_flat_embed = tf.keras.layers.Flatten()(num_spatial_dropout)
    
    merged_bert_embeds = tf.keras.layers.Concatenate()([t, q, a ])
    # bert_spatial_dropout = tf.keras.layers.SpatialDropout1D(0.3)(merged_bert_embeds)
    bert_flat_embed = tf.keras.layers.Flatten()(merged_bert_embeds)
    
    x = tf.keras.layers.Concatenate()([bert_flat_embed,cat_flat_embed, num_flat_embed])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    y = tf.keras.layers.Dense(30, activation="sigmoid")(x)


    model = tf.keras.models.Model(inputs=[t_id, t_mask, t_atn, q_id, q_mask, q_atn, a_id, a_mask, a_atn, cat1_input, cat2_input, cat3_input, numeric_input1, numeric_input2, numeric_input3], outputs=y)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # print(model.summary())
    return model

NFOLDS=10
kf = KFold(n_splits=NFOLDS)
kf.get_n_splits(train.qa_id)
print(text_feats)

EPOCHS=3
BATCH_SIZE=4
CLASS_WEIGHTS=None
predictions = np.zeros((len(test),len(targets)))
catcols=["domain","subdomain","category"]
numcols=["punc_t","punc_q","punc_a"]
for train_index, test_index in kf.split(train):
    print("Generating X train..............")
    X_tr=compute_input_arrays(train.loc[train_index,text_feats])
    y_tr=train.loc[train_index,targets].values
   
    for ls in compute_catkpis(train.loc[train_index,:]):
        X_tr.append(ls)
    for ls in compute_numkpis(train.loc[train_index,:]):
        X_tr.append(ls)


    # X_tr=np.hstack((X_tr,X_ot))
    print("Generating X valid..............")
    X_v=compute_input_arrays(train.loc[test_index,text_feats])
    y_v=train.loc[test_index,targets].values
    
    for ls in compute_catkpis(train.loc[test_index,:]):
        X_v.append(ls)
    for ls in compute_numkpis(train.loc[test_index,:]):
        X_v.append(ls)
    

    print("Generating X test..............")
    X_test=compute_input_arrays(test.loc[:,text_feats])
    
    
    for ls in compute_catkpis(test):
        X_test.append(ls)
    for ls in compute_numkpis(test):
        X_test.append(ls)
    

    

    for i,arr in enumerate(X_tr):
        print(i,type(arr),arr.shape)


    model=create_model(catcols,numcols)
    
    history= model.fit(
        x=X_tr,
        y=y_tr,
        validation_data=(X_v,y_v),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHTS,
        callbacks=[SpearmanRhoCallback(training_data=(X_tr,y_tr), validation_data=(X_v,y_v),
                                       patience=5, model_name=u'best_model_batch.h5')],
        verbose=2)
    temp_preds=model.predict(X_test)
    predictions+=temp_preds/NFOLDS
    # break

submission=pd.DataFrame(predictions,columns=targets)
submission[ids[0]]=test[ids[0]]
submission.to_csv("submission.csv", index = False)
submission.head()
