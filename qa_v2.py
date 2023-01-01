#local
from parse_v1 import *
#external
import pandas as pd
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
#from tqdm import tqdm
import numpy as np
#from sklearn.model_selection import train_test_split
#import pickle
from copy import deepcopy
#from numpy import save, load


from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Bidirectional, TimeDistributed   #layers required for network
from tensorflow.keras.layers import Layer, Conv1D, Softmax, Concatenate ,Dropout, MaxPool1D        #layers required for network
from tensorflow.keras.backend import expand_dims, tile, concatenate, shape, batch_dot, squeeze     #functions required for network
import tensorflow.keras.backend as K                                                               #to build metric
from tensorflow.keras.models import Model                                                          #to build model
import tensorflow as tf                                                                            #other functions
import numpy as np                                                                                 #for numpy operations
import nltk
#nltk.download('punkt')

# https://medium.com/analytics-vidhya/question-and-answering-system-using-bidirectional-attention-flow-on-squad-dataset-ff88b454af14
# https://github.com/th3darkprince/BiDAF_SQUAD

def get_start_end_words(row):  
    
    answer = word_tokenize(row['answer'])
    context = word_tokenize(row['story'])
    
    start_word=end_word=-1
    
    match=False

    for j in range(len(context)-len(answer)):
        if context[j]==answer[0]:
            match=True
            k=0
            for k in range(1, len(answer)):
                if context[j+k]!=answer[k]:
                    match=False
            if match==True:
                start_word=j
                end_word=j+k
                break

    row['start_word'] = start_word
    row['end_word'] = end_word

    return row

def tokenize_entries(row):
    question = word_tokenize(row['question'])
    context = word_tokenize(row['story'])
    
    question_word_tokens = []
    context_word_tokens = []
    question_char_tokens = []
    context_char_tokens = []
    
    for i in question:
        if i in word_tokenizer.keys():
            question_word_tokens.append(word_tokenizer[i])
            question_char_tokens.append(char_tokenizer.texts_to_sequences([i])[0])
        else:
            question_word_tokens.append(word_tokenizer['UNK'])
            question_char_tokens.append(char_tokenizer.texts_to_sequences([i])[0])
            
    for i in context:
        if i in word_tokenizer.keys():
            context_word_tokens.append(word_tokenizer[i])
            context_char_tokens.append(char_tokenizer.texts_to_sequences([i])[0])
        else:
            context_word_tokens.append(word_tokenizer['UNK'])
            context_char_tokens.append(char_tokenizer.texts_to_sequences([i])[0])
            
    row['question_word_tokens'] = question_word_tokens
    row['context_word_tokens'] = context_word_tokens
    row['question_char_tokens'] = question_char_tokens
    row['context_char_tokens'] = context_char_tokens
    
    return row

def pad_word_sequences(row):
    
    question = row['question_word_tokens'].copy()
    for i in range(len(question), question_max):
        question.append(word_tokenizer['PAD'])
        
    context = row['context_word_tokens'].copy()
    for i in range(len(context), context_max):
        context.append(word_tokenizer['PAD'])
        
    question = np.array(question[:question_max], dtype=np.int32)
    context = np.array(context[:context_max], dtype=np.int32)
    
    row['question_word_padded'] = question
    row['context_word_padded'] = context
    
    return row



def pad_char_sequences(row):
    """Function that does padding for character tokens"""
    
    question = deepcopy(row['question_char_tokens'])
    question_chars = []
    for i in question:
        for j in range(len(i), char_max):
            i.append(0)
        question_chars.append(np.array(i[:char_max], dtype=np.int32))
    
    for i in range(len(question_chars), question_max):
        question_chars.append(np.zeros(char_max, dtype=np.int32))
        
    context = deepcopy(row['context_char_tokens'])
    context_chars = []
    for i in context:
        for j in range(len(i), char_max):
            i.append(0)
        context_chars.append(np.array(i[:char_max], dtype=np.int32))
        
    for i in range(len(context_chars), context_max):
        context_chars.append(np.zeros(char_max, dtype=np.int32))
        
    question_chars = np.array(question_chars, dtype=np.int32)
    context_chars = np.array(context_chars, dtype=np.int32)
    
    row['question_char_padded'] = question_chars
    row['context_char_padded'] = context_chars
    
    return row

class word_embedding_layer(Layer):
    
    def __init__(self, input_dim, output_dim, input_len):
        
        super(word_embedding_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.word_embed = Embedding(self.input_dim, self.output_dim, weights = [embedding_matrix_word], 
                               input_length = input_len, trainable = False, name = self._name+"_layer")

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        question, context = inputs
        return self.word_embed(question), self.word_embed(context) 
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_len': self.input_len
            
        })
        
        return config

class char_embedding_layer(Layer):
    
    def __init__(self, input_dim, output_dim, input_len):
        
        super(char_embedding_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.char_embed = Embedding(self.input_dim, self.output_dim, weights = [embedding_matrix_char], 
                               input_length = input_len, trainable = False)
        self.timed = TimeDistributed(self.char_embed)
        

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        question, context = inputs
        return self.timed(question), self.timed(context)
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_len': self.input_len
            
        })
        
        return config

class char_cnn_layer(Layer):
    
    def __init__(self, n_filters, filter_width):
        
        super(char_cnn_layer, self).__init__()
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.conv = Conv1D(self.n_filters, self.filter_width)
        self.timed = TimeDistributed(self.conv)
          
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        question, context = inputs
        return tf.math.reduce_max(self.timed(question), 2), tf.math.reduce_max(self.timed(context), 2)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_filters': self.n_filters,
            'filter_width': self.filter_width
            
        })
        
        return config

class highway_input_layer(Layer):
    
    def __init__(self):
        
        super(highway_input_layer, self).__init__()
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):        
        q_w, c_w, q_c, c_c = inputs
        question = concatenate([q_w, q_c], axis=2)  
        context = concatenate([c_w, c_c], axis=2)  
        
        return context, question

class highway_layer(Layer):
    
    def __init__(self, name):
        
        super(highway_layer, self).__init__()
        self._name = name
        self.normal = Dense(868, activation = "relu")
        self.gate = Dense(868, activation = "sigmoid")
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):        
        
        n = self.normal(inputs)
        g = self.gate(inputs)
        x = g*n + (1-g)*inputs
        
        return x

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'name': self._name
            
        })
        
        return config

class contextual_layer(Layer):
    
    def __init__(self, output_dim, name):
        
        super(contextual_layer, self).__init__()
        self.output_dim = output_dim
        self._name = name       
        self.contextual = Bidirectional(GRU(self.output_dim, return_sequences=True, dropout=0.2, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=67)))

    def build(self, input_shape):
        self.built = True 

    def call(self, inputs):
        return self.contextual(inputs)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'name': self._name
        })
        
        return config

class attention_input_layer(Layer):
    
    def __init__(self):
        
        super(attention_input_layer, self).__init__()
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        H,U = inputs
        
        expand_h = concatenate([[1,1],[shape(U)[1]],[1]],0)
        expand_u = concatenate([[1],[shape(H)[1]],[1,1]],0)
    
        h = tile(expand_dims(H, axis=2), expand_h)
        u = tile(expand_dims(U, axis=1), expand_u)
        h_u = h * u
        
        return concatenate([h,u,h_u], axis=-1)  



class attention_layer(Layer):
    
    def __init__(self):
        
        super(attention_layer, self).__init__()
        self.dense = Dense(1, activation = "linear", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=54))
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        sim_matrix = self.dense(inputs)
        sim_matrix = squeeze(sim_matrix, 3)
        
        return sim_matrix

class c2q_q2c_layer(Layer):
    
    def __init__(self):
        
        super(c2q_q2c_layer, self).__init__()
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        sim_matrix, H, U = inputs
    
        c2q = batch_dot(tf.nn.softmax(sim_matrix, -1), U)
        
        q2c = batch_dot(tf.nn.softmax(tf.math.reduce_max(sim_matrix, 2), -1), H)
        q2c = tile(expand_dims(q2c, axis=1),[1,shape(H)[1],1])
        
        return c2q, q2c

class modelling_input_layer(Layer):
    
    def __init__(self):
        
        super(modelling_input_layer, self).__init__()
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        H, c2q, q2c = inputs
        G = concatenate([H, c2q, (H*c2q), (H*q2c)], axis=2)
        
        return G



class modelling_layer(Layer):
    
    def __init__(self, output_dim):
        
        super(modelling_layer, self).__init__()
        self.output_dim = output_dim
        self.modelling1 = Bidirectional(GRU(self.output_dim, return_sequences=True, dropout=0.2))
        self.modelling2 = Bidirectional(GRU(self.output_dim, return_sequences=True, dropout=0.2))
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return self.modelling2(self.modelling1(inputs))
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        
        return config

class input_to_start(Layer):
    
    def __init__(self):
        
        super(input_to_start, self).__init__()
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        G, M = inputs
        GM = concatenate([G, M], axis=2)
        
        return GM

class output_start(Layer):
    
    def __init__(self):
        
        super(output_start, self).__init__()
        self.dense = Dense(1, activation = "linear", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=35))
        self.dropout = Dropout(0.2)
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        GM = inputs
        start = self.dense(GM)
        start = self.dropout(start)
        p1 = tf.nn.softmax(squeeze(start, axis=2))
        
        return p1
     
class input_to_end(Layer):
    
    def __init__(self, output_dim):
        
        super(input_to_end, self).__init__()
        self.output_dim = output_dim
        self.end = Bidirectional(GRU(self.output_dim, return_sequences=True, dropout=0.2, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=5)))
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        G, M = inputs
        M2 = self.end(M)
        GM2 = concatenate([G, M2], axis=2)
        return GM2
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        
        return config

class output_end(Layer):
    
    def __init__(self):
        
        super(output_end, self).__init__()
        self.dense = Dense(1, activation = "linear", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=85))
        self.dropout = Dropout(0.2)
        
    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        
        GM2 = inputs
        end = self.dense(GM2)
        end = self.dropout(end)
        p2 = tf.nn.softmax(squeeze(end, axis=2))
        
        return p2

def f1_score(y_true, y_pred):    #taken from old keras source code
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    
    return f1_val

def bidaf_model(question_timesteps, context_timesteps, hidden_size, char_vocab, n_filters, filter_width):
    """Function that build the BiDAF model using the custom class layers"""
    
    #inputs
    question_words = Input(shape=(question_timesteps,), name = 'question_word_tokens')
    context_words = Input(shape=(context_timesteps,), name = 'context_word_tokens')    
    question_chars = Input(shape=(question_timesteps,char_max,), name = 'question_char_tokens')
    context_chars = Input(shape=(context_timesteps,char_max,), name = 'context_char_tokens')

    #word embedding layer
    question_word_embedded, context_word_embedded = word_embedding_layer(len(word_tokenizer)+1, hidden_size, question_words.shape[-1])([question_words, context_words])
            
    #character embedding layer
    question_char_embedded, context_char_embedded = char_embedding_layer(len(char_tokenizer.word_index)+1, char_vocab, question_chars.shape[-1])([question_chars, context_chars])
    
    #character CNN
    question_char_embedded, context_char_embedded  = char_cnn_layer(n_filters, filter_width)([question_char_embedded, context_char_embedded])
    
    #highway layer
    context, question = highway_input_layer()([question_word_embedded, context_word_embedded, question_char_embedded, context_char_embedded])
    question_blendrep = highway_layer("question_highway")(question)
    context_blendrep = highway_layer("context_highway")(context)

    #contextual layer
    question_contextual = contextual_layer(hidden_size, "question_contextual")(question_blendrep)
    context_contextual = contextual_layer(hidden_size, "context_contextual")(context_blendrep)

    #attention layer
    attention_in = attention_input_layer()([context_contextual, question_contextual])
    attention_out = attention_layer()(attention_in)
    c2q, q2c = c2q_q2c_layer()([attention_out, context_contextual, question_contextual])
    
    #modelling layer
    G = modelling_input_layer()([context_contextual, c2q, q2c])
    M = modelling_layer(hidden_size)(G)
    
    #output layers
    GM = input_to_start()([G,M])
    start = output_start()(GM)
    GM2 = input_to_end(hidden_size)([G,M])
    end = output_end()(GM2)

    model = Model(inputs=[question_words,context_words,question_chars,context_chars], outputs=[start, end], name="bidaf")
        
    return model    

if __name__ == "__main__":
    #READ
    path, question_IDs = read_input_file(sys.argv[1])
    stories = []
    questions = []
    answers = []
    df = []
    for file in question_IDs:
        s = parse_story_file(path, file)
        a = parse_answers_file(path, file, split_multiple_answers=True)
        for key in a.keys():
            df.append([a[key][1], a[key][3], s["text"]])
    
    #print(df[0:2])
    df = pd.DataFrame(df, columns=["question","answer","story"])
    #print(df)

    word_dictionary = {}

    for i, row in df.iterrows():
        
        text = row["question"] + " " + row["story"]
        tokens = word_tokenize(text)
        
        for j in tokens:
            if j not in word_dictionary.keys():
                word_dictionary[j] = len(word_dictionary)
    
    #print(word_dictionary)

    #print("Number of unique words in the total dataset(train + test): ",len(word_dictionary))

    dataset =  df.apply(get_start_end_words, axis=1)
    #print(dataset)
    df_train = dataset

    word_tokenizer = {'PAD':0,'UNK':1}

    for i, row in df_train.iterrows():
        
        text = row['question'] + " " + row['story']
        tokens = word_tokenize(text)
        
        for j in tokens:
            if j not in word_tokenizer.keys():
                word_tokenizer[j] = len(word_tokenizer)
    char_tokenizer = Tokenizer(char_level=True, oov_token='UNK')
    char_tokenizer.fit_on_texts(df_train['question'] + df_train['story'])

    df_train = df_train.apply(tokenize_entries, axis=1)
    print(df_train.columns)

    question_word_token_lens = []

    for i in df_train['question_word_tokens'].values:
        question_word_token_lens.extend([len(i)])
        
    print("Max number of words in question: ",np.array(question_word_token_lens).max())
    print("Mean number of words in question: ",np.array(question_word_token_lens).mean())

    context_word_token_lens = []

    for i in df_train['context_word_tokens'].values:
        context_word_token_lens.extend([len(i)])

    print("Max number of words in context: ",np.array(context_word_token_lens).max())
    print("Mean number of words in context: ",np.array(context_word_token_lens).mean())

    question_max = np.array(question_word_token_lens).max()
    context_max = np.array(context_word_token_lens).max()
    print("question_max",question_max)
    print("context_max",context_max)
    char_max = 0
    total_len = 0
    total_words = 0
    dist = []

    for i, row in df_train.iterrows():
        for j in row['question_char_tokens']:
            dist.append(len(j))
            total_len += len(j)
            total_words += 1
            if len(j)>char_max:
                char_max = len(j)
                
                
        for k in row['context_char_tokens']:
            dist.append(len(k))
            total_len += len(j)
            total_words += 1
            if len(k)>char_max:
                char_max = len(k)

    print("Maximum length of word: ", char_max)
    print("Mean length of word: ", total_len/total_words)
    print("99.9 percentage of words have a character length of : ", np.percentile(dist,99.9))

    char_max = 15
    print("len word_tokenizer",len(word_tokenizer))
    print("len char_tokenizer",len(char_tokenizer.word_index))

    df_train = df_train.apply(pad_word_sequences, axis=1)
    df_train = df_train.apply(pad_char_sequences, axis=1)
    #print(df_train.columns)

    train_context_word_padded = np.asarray(df_train['context_word_padded'].values.tolist(), dtype=np.int32)
    train_question_word_padded = np.asarray(df_train['question_word_padded'].values.tolist(), dtype=np.int32)
    train_context_char_padded = np.asarray(df_train['context_char_padded'].values.tolist())
    train_question_char_padded = np.asarray(df_train['question_char_padded'].values.tolist())

    num_classes = context_max
    y_start_train = keras.utils.to_categorical(df_train['start_word'].values, num_classes)
    y_end_train = keras.utils.to_categorical(df_train['end_word'].values, num_classes)

    #from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Bidirectional, TimeDistributed   #layers required for network
    #from tensorflow.keras.layers import Layer, Conv1D, Softmax, Concatenate ,Dropout, MaxPool1D        #layers required for network
    #from tensorflow.keras.backend import expand_dims, tile, concatenate, shape, batch_dot, squeeze     #functions required for network
    #import tensorflow.keras.backend as K                                                               #to build metric
    #from tensorflow.keras.models import Model                                                          #to build model
    #import tensorflow as tf                                                                            #other functions
    #import numpy as np                                                                                 #for numpy operations
    #import nltk
    #nltk.download('punkt')

    #referred from preproceesing
    question_max = 38 #32
    context_max = 340#806 #340
    char_max = 15
    #print("question_max",question_max)
    #print("context_max",context_max)

    #https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    # CREATE EMBEDINGS WITH BERT?
    #enc_data = {}
    #with open("glove.6B.100d.txt",'rb') as f:
    #    for line in f:
    #        values = line.split()
    #        word = values[0]
    #        vector = np.asarray(values[1:], "float32")
    #        enc_data[word.decode('utf-8')] = vector
    #    glove_words = set(enc_data.keys())
    #    print("glove_words",len(glove_words))
    
    #count = 0
    #embedding_matrix_word = np.zeros((len(word_tokenizer)+1, 100))
    #for word, i in word_tokenizer.items():
    #    embedding_vector = enc_data.get(word)
    #    if embedding_vector is not None:
    #        count += 1
    #        embedding_matrix_word[i] = embedding_vector
    
    #print("embedding_matrix_word",embedding_matrix_word)
    #print("embedding_matrix_word",embedding_matrix_word.shape)
    
    #print("Percentage of words covered by Glove vectors:", count/len(word_tokenizer)*100)



    embedding_matrix_char = []
    embedding_matrix_char.append(np.zeros(len(char_tokenizer.word_index)))

    for char, i in char_tokenizer.word_index.items():
        onehot = np.zeros(len(char_tokenizer.word_index))
        onehot[i-1] = 1
        embedding_matrix_char.append(onehot)

    embedding_matrix_char = np.array(embedding_matrix_char)

    #print("embedind_matrix_char",embedding_matrix_char)
    print("embedind_matrix_char",embedding_matrix_char.shape)

        
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained("bert-base-uncased")
    x = [y[0] for y in word_tokenizer.items()]
    embedding_matrix_word = np.zeros((len(word_tokenizer)+1, 768))
    for i, e in enumerate(x):
        encoded_input = tokenizer(e, return_tensors="pt")
        output = model(**encoded_input)
        embedding_matrix_word[i] = (output.pooler_output[0]).tolist()
    print("embedding_matrix_word",embedding_matrix_word.shape)
    
    
    question_timesteps = question_max
    context_timesteps = context_max

    #output dimensions of word and character embedding
    hidden_size = 768 #100 # 768
    char_vocab = 1218

    #character CNN filters and width
    n_filters = 100#100 # 768
    filter_width = 3

    tf.keras.backend.clear_session()
    model = bidaf_model(question_timesteps, context_timesteps, hidden_size, char_vocab, n_filters, filter_width)

    print(model.summary())

    loss_function = tf.keras.losses.CategoricalCrossentropy(reduction='auto')
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0005)

    train_inputs = tf.data.Dataset.from_tensor_slices((train_question_word_padded, train_context_word_padded, train_question_char_padded, train_context_char_padded))
    train_targets = tf.data.Dataset.from_tensor_slices((y_start_train, y_end_train))
    train_dataset = tf.data.Dataset.zip((train_inputs, train_targets)).shuffle(500).batch(4).prefetch(tf.data.experimental.AUTOTUNE)

    @tf.function
    def train_step(input_vector, output_vector,loss_fn):
        with tf.GradientTape() as tape:
            #forward propagation
            output_predicted = model(inputs=input_vector, training=True)
            #loss
            loss_start = loss_function(output_vector[0], output_predicted[0])
            loss_end = loss_function(output_vector[1], output_predicted[1])
            loss_final = loss_start + loss_end
        #getting gradients
        gradients = tape.gradient(loss_final, model.trainable_variables)
        #applying gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_start, loss_end, output_predicted, gradients
    
    @tf.function
    def val_step(input_vector, output_vector, loss_fn):
        #getting output of validation data
        output_predicted = model(inputs=input_vector, training=False)
        #loss calculation
        loss_start = loss_function(output_vector[0], output_predicted[0])
        loss_end = loss_function(output_vector[1], output_predicted[1])
        return loss_start, loss_end, output_predicted

    BATCH_SIZE=4 #32

    EPOCHS=12

    train_start_loss = tf.keras.metrics.Mean(name='train_start_loss')
    train_end_loss = tf.keras.metrics.Mean(name='train_end_loss')
    val_start_loss = tf.keras.metrics.Mean(name='val_start_loss')
    val_end_loss = tf.keras.metrics.Mean(name='val_end_loss')
    train_start_f1 = tf.keras.metrics.Mean(name="train_start_f1")
    train_end_f1 = tf.keras.metrics.Mean(name="train_end_f1")
    val_start_f1 = tf.keras.metrics.Mean(name="val_start_f1")
    val_end_f1 = tf.keras.metrics.Mean(name="val_end_f1")

    wtrain = tf.summary.create_file_writer(logdir='logdir_train')
    wval = tf.summary.create_file_writer(logdir='logdir_val')
    import math
    iters = math.ceil(64769/32)
    best_loss=100


    for epoch in range(EPOCHS):
    
        #resetting the states of the loss and metrics
        train_start_loss.reset_states()
        train_end_loss.reset_states()
        val_start_loss.reset_states()
        val_end_loss.reset_states()
        train_start_f1.reset_states()
        train_end_f1.reset_states()
        val_start_f1.reset_states()
        val_end_f1.reset_states()
        
        ##counter for train loop iteration
        counter = 0
        
        #ietrating over train data batch by batch
        for text_seq, label_seq in train_dataset:
            #train step
            loss_start_, loss_end_, pred_out, gradients = train_step(text_seq, label_seq, loss_function)
            #adding loss to train loss
            train_start_loss(loss_start_)
            train_end_loss(loss_end_)
            #counting the step number
            temp_step = epoch*iters+counter
            counter = counter + 1
            
            #calculating f1 for batch
            f1_start = f1_score(label_seq[0], pred_out[0])
            f1_end = f1_score(label_seq[1], pred_out[1])
            train_start_f1(f1_start)
            train_end_f1(f1_end)
            
            ##tensorboard 
            with tf.name_scope('per_step_training'):
                with wtrain.as_default():
                    tf.summary.scalar("start_loss", loss_start_, step=temp_step)
                    tf.summary.scalar("end_loss", loss_end_, step=temp_step)
                    tf.summary.scalar('f1_start', f1_start, step=temp_step)
                    tf.summary.scalar('f1_end', f1_end, step=temp_step)
            with tf.name_scope("per_batch_gradients"):
                with wtrain.as_default():
                    for i in range(len(model.trainable_variables)):
                        name_temp = model.trainable_variables[i].name
                        tf.summary.histogram(name_temp, gradients[i], step=temp_step)
        
        
        #validation data
        #for text_seq_val, label_seq_val in test_dataset:
        #    #getting val output
        #    loss_val_start, loss_val_end, pred_out_val = val_step(text_seq_val, label_seq_val, loss_function)
        #    
        #    val_start_loss(loss_val_start)
        #    val_end_loss(loss_val_end)
        #    
        #    #calculating metric
        #    f1_start_val = f1_score(label_seq_val[0], pred_out_val[0])
        #    f1_end_val = f1_score(label_seq_val[1], pred_out_val[1])
        #    val_start_f1(f1_start_val)
        #    val_end_f1(f1_end_val)
        
    
        #printing
        #template = '''Epoch {}, Train Start Loss: {:0.6f}, Start F1 Score: {:0.5f}, Train End Loss: {:0.6f}, End F1 Score: {:0.5f},
        #Val Start Loss: {:0.6f}, Val Start F1 Score: {:0.5f}, Val End Loss: {:0.6f}, Val End F1 Score: {:0.5f}'''

        #print(template.format(epoch+1, train_start_loss.result(), train_start_f1.result(), 
        #                    train_end_loss.result(), train_end_f1.result(),
        #                    val_start_loss.result(), val_start_f1.result(),
        #                    val_end_loss.result(), val_end_f1.result()))


        if (val_start_loss.result()+val_end_loss.result())<best_loss:
            model.save("drive/My Drive/Colab Notebooks/new_model/model")
            best_loss=(val_start_loss.result()+val_end_loss.result())
        
        #tensorboard
        with tf.name_scope("per_epoch_loss_metric"):
            with wtrain.as_default():
                tf.summary.scalar("start_loss", train_start_loss.result().numpy(), step=epoch)
                tf.summary.scalar("end_loss", train_end_loss.result().numpy(), step=epoch)
                tf.summary.scalar('start_f1', train_start_f1.result().numpy(), step=epoch)
                tf.summary.scalar('end_f1', train_end_f1.result().numpy(), step=epoch)
            with wval.as_default():
                tf.summary.scalar("start_loss", val_start_loss.result().numpy(), step=epoch)
                tf.summary.scalar("end_loss", val_end_loss.result().numpy(), step=epoch)
                tf.summary.scalar('start_f1', val_start_f1.result().numpy(), step=epoch)
                tf.summary.scalar('end_f1', val_end_f1.result().numpy(), step=epoch)