# Load Library
from time import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
 
import itertools
import datetime
import json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
import tensorflow as tf
# End Library ML

# Class Load Data

class LoadData:
    def load_train_dataset(self,loc):
        data_df = pd.read_csv(loc, encoding='ISO-8859-1')
        return data_df
    
    def load_validation_dataset(self,loc):
        data_df = pd.read_csv(loc, encoding='ISO-8859-1')
        return data_df
    
    def load_test_dataset(self,loc):
        data_df = pd.read_csv(loc, encoding='ISO-8859-1')
        return data_df

# End Class Load Data

#Class Preprocessing

class Preproses:
    import json
    from json import load
    # Opening vocabulary JSON file 
    with open('app/static/file/embedding/vocabulary.json') as json_file: 
        vocabulary = load(json_file)

    # Opening inverse vocabulary file
    with open('app/static/file/embedding/inverse_vocabulary.txt', 'r') as f:
        inverse_vocabulary = json.loads(f.read())

    word2vec = Word2Vec.load('app/static/file/embedding/w2vec_wiki_id_300_2')

    def preprocessing(self, data_df):
        indonesia_stop_words = [
        'yang', 'ada', 'berikut',
        'ini', 'untuk', 'apa',
        'berbagai', 'tetapi', 'maupun',
        'atas', 'berarti', 'itu',
        'apakah', 'demikian', 'bagaikan',
        'adalah', 'ialah', 'oleh', 'dsb', 'dll', 'dkk'
        ]
        
        def remove_indonesia_stopwords(data_df):
            return ' '.join([word for word in str(data_df).lower().split() if word not in indonesia_stop_words])

        data_df['jawaban_siswa']=data_df['jawaban_siswa'].str.lower()
        data_df['kunci_jawaban']=data_df['kunci_jawaban'].str.lower()

        # Remove tanda baca
        data_df['jawaban_siswa'] = data_df['jawaban_siswa'].str.replace('[^\w\s]', ' ')
        data_df['kunci_jawaban'] = data_df['kunci_jawaban'].str.replace('[^\w\s]', ' ')

        # Remove Indonesia stop words
        data_df.jawaban_siswa = data_df.jawaban_siswa.apply(remove_indonesia_stopwords)
        data_df.kunci_jawaban = data_df.kunci_jawaban.apply(remove_indonesia_stopwords)

        return data_df
    
    def text_to_number(self,data_df):

        def text_to_word_list(text):
            ''' Pre process and convert texts to a list of words '''
            text = str(text)
            text = text.split()
            return text
        
        vocabulary = self.vocabulary
        inverse_vocabulary = self.inverse_vocabulary
        word2vec = self.word2vec

        kalimat_cols = ['jawaban_siswa', 'kunci_jawaban']

        outside_words = set()

        # Iterate over the sentences only of both training and test datasets
        for dataset in [data_df]:
            for index, row in dataset.iterrows():
            # Iterate through the text of both sentences of the row
                for kalimat in kalimat_cols:
                    kata_n_representasi = []  # kata representation

                    for word in text_to_word_list(row[kalimat]):
                        # mendadtarkan kata yang tidak ada pada word2vec
                        if word not in word2vec.wv.vocab:
                            outside_words.add(word)

                        if word not in vocabulary:
                            vocabulary[word] = len(inverse_vocabulary)
                            kata_n_representasi.append(len(inverse_vocabulary))
                            inverse_vocabulary.append(word)
                        else:
                            kata_n_representasi.append(vocabulary[word])

                        # Replace word to number representation
                        dataset.at[index, kalimat] = kata_n_representasi
            
        return data_df

    def embedding_matrix(self):
        vocabulary = self.vocabulary
        word2vec = self.word2vec
        embedding_dim = 300
        embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # Embedding matrix
        embeddings[0] = 0  # mengabaikan padding

        # Membangun embedding matrix
        for word, index in vocabulary.items():
            if word in word2vec.wv.vocab:
                embeddings[index] = word2vec.wv[word]

        del word2vec
        return embeddings
    
    def zero_padding(self,data_df):
        max_seq_length = 100
        data_df = {'left': data_df.jawaban_siswa, 'right': data_df.kunci_jawaban}

        for dataset, side in itertools.product([data_df],['left','right']):
            dataset[side] = pad_sequences(dataset[side], maxlen = max_seq_length)

        return data_df

# End Class Prepocessing

# Class Training

class Train:

    def parameter(self, batchsize=64, nhidden=30, epo=25):
        self.batchsize = batchsize
        self.nhidden = nhidden
        self.epo = epo
    
    def training(self, X_train, Y_train, X_validation, Y_validation, prep):
        tf.keras.backend.clear_session()
        tf.random.set_seed(3)
        np.random.seed(3)
        self.embed = prep.embedding_matrix()
        gradient_clipping_norm = 1.25 
        embedding_dim = 300
        max_seq_length = 100

        # def exponent_neg_manhattan_distance(left, right):
        #   ''' Helper function for the similarity estimate of the LSTMs outputs'''
        #   return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

        # The visible layer
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')

        embedding_layer = Embedding(len(self.embed), embedding_dim, weights=[self.embed], input_length=max_seq_length, trainable=False)

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(self.nhidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        self.model = Model([left_input, right_input], [malstm_distance])

        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)

        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Start training
        training_start_time = time()

        self.malstm_trained = self.model.fit([X_train['left'], X_train['right']], Y_train, batch_size=self.batchsize, epochs=self.epo,
                                    validation_data=([X_validation['left'], X_validation['right']], Y_validation), callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

        print("Training time finished.\n{} epochs in {}".format(self.epo, datetime.timedelta(seconds=time()-training_start_time)))
        return self.model
    
    def save_model(self, model):
        model_json = model.to_json()
        with open("app/static/file/model/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights('app/static/file/model/weights.h5')
        return "Model tersimpan."


#End Class Training

#Class Testing
class Testing:

    def set_model(self,train):
        json_file = open("app/static/file/model/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        train.model = model_from_json(loaded_model_json)
        train.model.load_weights("app/static/file/model/weights.h5")
        train.model_loaded = True

    def testing(self,data_df,prep,train):
        from copy import copy
        self.data_df = data_df
        self.prediksi_df = copy(data_df)
        datatest_df = data_df
        datatest_df = prep.preprocessing(datatest_df)
        datatest_df = prep.text_to_number(datatest_df)
        datatest_df = prep.zero_padding(datatest_df)

        hasil_test = np.round(train.model.predict([datatest_df['left'],datatest_df['right']]), 4)

        self.prediksi_df['prediction'] = hasil_test
        loc = 'app/static/file/predict/hasil_prediksi.csv'
        hasil_test_df = self.prediksi_df
        hasil_test_df.to_csv(loc, index=False)

        self.korelasi_df = copy(hasil_test_df)

        return loc
    
    def korelasi(self):
        pearson_value = self.korelasi_df['value'].corr(self.korelasi_df['prediction'], method='pearson')
        spearman_value = self.korelasi_df['value'].corr(self.korelasi_df['prediction'], method='spearman')

        return pearson_value, spearman_value
#End Class Testing

# Class Plot

class Plot():
    def plot(self, a, b, judul, label_a, label_b, label_y, label_x, keterangan):
        # plt.figure(figsize=(9,5))
        import datetime as dtm
        plt.plot(a, label=label_a, lw=2)
        plt.plot(b, label=label_b, lw=2)
        plt.title(judul)
        plt.ylabel(label_y)
        plt.xlabel(label_x)
        plt.legend()
        self.t = dtm.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.nm = keterangan + self.t + '.png'
        plt.savefig('app/static/img/grafik/'+self.nm)
        # plt.show()
        plt.close()
        return self.nm

# END Class Plot
