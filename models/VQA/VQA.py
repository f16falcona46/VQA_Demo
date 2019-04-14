# modification of model from https://github.com/avisingh599/visual-qa
from keras.models import Sequential, Model
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Concatenate, Dense, Input

def VQA_MODEL():
    image_feature_size          = 4096
    word_feature_size           = 300
    number_of_LSTM              = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions        = 30
    number_of_dense_layers      = 3
    number_of_hidden_units      = 1024
    activation_function         = 'tanh'
    dropout_pct                 = 0.5


    in1 = Input(shape=(max_length_questions, word_feature_size))
    in2 = Input(shape=(image_feature_size,))
    # Image model
    #model_image = Sequential()
    #model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))
    a = Reshape((image_feature_size,), input_shape=(image_feature_size,))(in2)

    # Language Model
    #model_language = Sequential()
    #model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    #model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    #model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))
    b = LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size))(in1)
    b = LSTM(number_of_hidden_units_LSTM, return_sequences=True)(b)
    b = LSTM(number_of_hidden_units_LSTM, return_sequences=False)(b)

    # combined model
    x = Concatenate(axis=1)([b, a])

    for _ in range(number_of_dense_layers):
        x = Dense(number_of_hidden_units, kernel_initializer='uniform')(x)
        x = Activation(activation_function)(x)
        x = Dropout(dropout_pct)(x)

    x = Dense(1000)(x)
    out = Activation('softmax')(x)

    model = Model(inputs=[in1, in2], outputs=out)
    return model






