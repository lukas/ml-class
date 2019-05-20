"""
   Copyright 2019 James Betker
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# RNN "Cell" classes in Keras perform the actual data transformations at each timestep. Therefore, in order
# to add attention to LSTM, we need to make a custom subclass of LSTMCell.
class AttentionLSTMCell(LSTMCell):
    def __init__(self, **kwargs):
        self.attentionMode = False
        super(AttentionLSTMCell, self).__init__(**kwargs)

    # Build is called to initialize the variables that our cell will use. We will let other Keras
    # classes (e.g. "Dense") actually initialize these variables.
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Converts the input sequence into a sequence which can be matched up to the internal
        # hidden state.
        self.dense_constant = TimeDistributed(Dense(self.units, name="AttLstmInternal_DenseConstant"))

        # Transforms the internal hidden state into something that can be used by the attention
        # mechanism.
        self.dense_state = Dense(self.units, name="AttLstmInternal_DenseState")

        # Transforms the combined hidden state and converted input sequence into a vector of
        # probabilities for attention.
        self.dense_transform = Dense(1, name="AttLstmInternal_DenseTransform")

        # We will augment the input into LSTMCell by concatenating the context vector. Modify
        # input_shape to reflect this.
        batch, input_dim = input_shape[0]
        batch, timesteps, context_size = input_shape[-1]
        lstm_input = (batch, input_dim + context_size)

        # The LSTMCell superclass expects no constant input, so strip that out.
        return super(AttentionLSTMCell, self).build(lstm_input)

    # This must be called before call(). The "input sequence" is the output from the
    # encoder. This function will do some pre-processing on that sequence which will
    # then be used in subsequent calls.
    def setInputSequence(self, input_seq):
        self.input_seq = input_seq
        self.input_seq_shaped = self.dense_constant(input_seq)
        self.timesteps = tf.shape(self.input_seq)[-2]

    # This is a utility method to adjust the output of this cell. When attention mode is
    # turned on, the cell outputs attention probability vectors across the input sequence.
    def setAttentionMode(self, mode_on=False):
        self.attentionMode = mode_on

    # This method sets up the computational graph for the cell. It implements the actual logic
    # that the model follows.
    def call(self, inputs, states, constants):
        # Separate the state list into the two discrete state vectors.
        # ytm is the "memory state", stm is the "carry state".
        ytm, stm = states
        # We will use the "carry state" to guide the attention mechanism. Repeat it across all
        # input timesteps to perform some calculations on it.
        stm_repeated = K.repeat(self.dense_state(stm), self.timesteps)
        # Now apply our "dense_transform" operation on the sum of our transformed "carry state"
        # and all encoder states. This will squash the resultant sum down to a vector of size
        # [batch,timesteps,1]
        combined_stm_input = self.dense_transform(
            keras.activations.tanh(stm_repeated + self.input_seq_shaped))
        # Performing a softmax generates a log probability for each encoder output to receive attention.
        score_vector = keras.activations.softmax(combined_stm_input, 1)
        # In this implementation, we grant "partial attention" to each encoder output based on
        # it's log probability accumulated above. Other options would be to only give attention
        # to the highest probability encoder output or some similar set.
        context_vector = K.sum(score_vector * self.input_seq, 1)

        # Finally, mutate the input vector. It will now contain the traditional inputs (like the seq2seq
        # we trained above) in addition to the attention context vector we calculated earlier in this method.
        inputs = K.concatenate([inputs, context_vector])

        # Call into the super-class to invoke the LSTM math.
        res = super(AttentionLSTMCell, self).call(inputs=inputs, states=states)

        # This if statement switches the return value of this method if "attentionMode" is turned on.
        if(self.attentionMode):
            return (K.reshape(score_vector, (-1, self.timesteps)), res[1])
        else:
            return res

# Custom implementation of the Keras LSTM that adds an attention mechanism.
# This is implemented by taking an additional input (using the "constants" of the
# RNN class) into the LSTM: The encoder output vectors across the entire input sequence.
class LSTMWithAttention(RNN):
    def __init__(self, units, **kwargs):
        cell = AttentionLSTMCell(units=units)
        self.units = units
        super(LSTMWithAttention, self).__init__(cell, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        self.timesteps = input_shape[0][-2]
        return super(LSTMWithAttention, self).build(input_shape)

    # This call is invoked with the entire time sequence. The RNN sub-class is responsible
    # for breaking this up into calls into the cell for each step.
    # The "constants" variable is the key to our implementation. It was specifically added
    # to Keras to accomodate the "attention" mechanism we are implementing.
    def call(self, x, constants, **kwargs):
        if isinstance(x, list):
            self.x_initial = x[0]
        else:
            self.x_initial = x

        # The only difference in the LSTM computational graph really comes from the custom
        # LSTM Cell that we utilize.
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell.setInputSequence(constants[0])
        return super(LSTMWithAttention, self).call(inputs=x, constants=constants, **kwargs)