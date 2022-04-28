from tensorflow.keras.layers import Input, Embedding, GRU, Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class AutoEncoderModel:
    def __init__(self, config):
        self.config = config
        self.encoder_input = Input(shape=(None, ), name='encoder_input')
        self.encoder_embedding = Embedding(input_dim=self.config['num_words'],
                                      output_dim = self.config['embedding_size'],
                                      name = 'encoder_embedding')
        self.encoder_gru1 = GRU(self.config['state_size'],
                          return_sequences=True,
                          name= 'encoder_gru1')
        self.encoder_gru2 = GRU(self.config['state_size'],
                           return_sequences= True,
                           name='encoder_gru2')
        self.encoder_gru3 = GRU(self.config['state_size'],
                           return_sequences= False,
                           name = 'encoder_gru3')
        
        
        self.decoder_initial_state = Input(shape=(self.config['state_size'],),
                                      name = 'decoder_initial_state')

        self.decoder_input = Input(shape=(None, ),
                              name = 'decoder_input'
                              ) 
        
        self.decoder_embedding = Embedding(input_dim = self.config['num_words'],
                                      output_dim = self.config['embedding_size'],
                                      name = 'decoder_embedding')
        
        self.decoder_gru1 = GRU(self.config['state_size'],
                           name='decoder_gru1',
                           return_sequences=True)
        self.decoder_gru2 = GRU(self.config['state_size'],
                           name='decoder_gru2',
                           return_sequences=True)
        self.decoder_gru3 = GRU(self.config['state_size'],
                           name='decoder_gru3',
                           return_sequences=True)
        
        self.decoder_dense = Dense(self.config['num_words'],
                      activation=self.config['activation'],
                      name='decoder_output')
        
    def encoderModel(self):
       
        
        encoder = self.encoder_input
        encoder = self.encoder_embedding(encoder)
        encoder = self.encoder_gru1(encoder)
        encoder = self.encoder_gru2(encoder)
        encoder = self.encoder_gru3(encoder)
        
        encoder_output = encoder
        
        encoder_model = Model(inputs= [self.encoder_input],
                              outputs = [encoder_output])
        return encoder_output, encoder_model
    
    def decoderModel(self, encoder_output, alone=False):
        
        decoder = self.decoder_input
        decoder = self.decoder_embedding(decoder)
        
        if alone:
            decoder = self.decoder_gru1(decoder,
                                   initial_state=self.decoder_initial_state)
            decoder = self.decoder_gru2(decoder,
                                   initial_state=self.decoder_initial_state)
            decoder = self.decoder_gru3(decoder,
                                   initial_state=self.decoder_initial_state)

        else:
            decoder = self.decoder_gru1(decoder,
                                   initial_state=encoder_output)
            decoder = self.decoder_gru2(decoder,
                                   initial_state=encoder_output)
            decoder = self.decoder_gru3(decoder,
                                   initial_state=encoder_output)
    
        decoder_output = self.decoder_dense(decoder)
        decoder_model = Model()
        if alone:
            decoder_model = Model(inputs= [self.decoder_input,    
                                           self.decoder_initial_state],
                              outputs = [decoder_output])
        else:
            decoder_model = Model(inputs= [self.encoder_input,
                                           self.decoder_input],
                              outputs = [decoder_output])
        return decoder_output, decoder_model
    
    def BuildModel(self):
        encoder_output, encoder_model = self.encoderModel()
        
        decoder_output, decoder_model = self.decoderModel(encoder_output,False)
        
        autoencoder = Model(inputs= [self.encoder_input, self.decoder_input],
                            outputs= [decoder_output])
        
        autoencoder.compile(optimizer=deserialize(self.config['optimizer']),
                            loss='sparse_categorical_crossentropy')
        
        return autoencoder
    
    def getCallbacks(self):
       
        callbacks = []
        if ('ModelCheckPoint' in self.config['callbacks']):
            callback_checkpoint = ModelCheckpoint(filepath=self.config['callbacks']['ModelCheckPoint']['filepath'],
                                      monitor= self.config['callbacks']['ModelCheckPoint']['monitor'],
                                      verbose=self.config['callbacks']['ModelCheckPoint']['verbose'],
                                      save_weights_only=self.config['callbacks']['ModelCheckPoint']['save_weights_only'],
                                      save_best_only=self.config['callbacks']['ModelCheckPoint']['save_best_only'])
            callbacks.append(callback_checkpoint)
        if ('EarlyStopping' in self.config['callbacks']):
            
            callback_early_stopping = EarlyStopping(monitor=self.config['callbacks']['EarlyStopping']['monitor'],
                                        patience=self.config['callbacks']['EarlyStopping']['patience']
                                        , verbose=self.config['callbacks']['EarlyStopping']['verbose'])
            callbacks.append(callback_early_stopping)
        
        if ('TensorBoard' in self.config['callbacks']):
           
            callback_tensorboard = TensorBoard(log_dir=self.config['callbacks']['TensorBoard']['log_dir'],
                                   histogram_freq=self.config['callbacks']['TensorBoard']['histogram_freq'],
                                   write_graph=self.config['callbacks']['TensorBoard']['write_graph'])
            
            callbacks.append(callback_tensorboard)

        return callbacks
        
        
        
        