import attention

# Re-create an entirely new model and set of layers for the attention model

# Encoder Layers
attenc_inputs = Input(shape=(len_input,), name="attenc_inputs")
attenc_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
attenc_lstm = CuDNNLSTM(units=units, return_sequences=True, return_state=True)
attenc_outputs, attstate_h, attstate_c = attenc_lstm(attenc_emb(attenc_inputs))
attenc_states = [attstate_h, attstate_c]

attdec_inputs = Input(shape=(None,))
attdec_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
attdec_lstm = LSTMWithAttention(units=units, return_sequences=True, return_state=True)
# Note that the only real difference here is that we are feeding attenc_outputs to the decoder now.
# Nice and clean!
attdec_lstm_out, _, _ = attdec_lstm(inputs=attdec_emb(attdec_inputs), 
                                    constants=attenc_outputs, 
                                    initial_state=attenc_states)
attdec_d1 = Dense(units, activation="relu")
attdec_d2 = Dense(vocab_out_size, activation="softmax")
attdec_out = attdec_d2(Dropout(rate=.4)(attdec_d1(Dropout(rate=.4)(attdec_lstm_out))))

attmodel = Model([attenc_inputs, attdec_inputs], attdec_out)
attmodel.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

epochs = 20
atthist = attmodel.fit([input_data, teacher_data], target_data,
                 batch_size=BATCH_SIZE,
                 epochs=epochs,
                 validation_split=0.2)
