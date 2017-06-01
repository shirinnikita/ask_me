from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import GRU, activations, Wrapper

class myGru(GRU):
    def build(self, input_shape):
        super(myGru, self).build(input_shape)
        self.w_b = self.add_weight(shape=(self.input_dim // 2, self.input_dim // 2),
                                name='w_b',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
        self.w_1 = self.add_weight(shape=((self.input_dim // 2) * 7 + 2, self.input_dim // 2),
                                name='w_1',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
        self.b_1 = self.add_weight(shape=(self.input_dim // 2,),
                                name='b_1',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)
        self.w_2 = self.add_weight(shape=(self.input_dim // 2, 1),
                                name='w_2',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
        self.b_2 = self.add_weight(shape=(1,),
                                name='b_2',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)
        return

    def step(self, inputs, states):
        h, [h] = super(myGru, self).step(inputs, states)
        dim = int(inputs.shape[1]) // 2
        inp = inputs[:, :dim]
        prev = states[0]
        question = inputs[:, dim:]
        z = K.concatenate([inp, prev, question,
            inp * question, inp * prev, K.abs(inp - question), K.abs(inp - prev),
            K.batch_dot(inp, K.dot(question, self.w_b), axes=1),
            K.batch_dot(inp, K.dot(prev, self.w_b), axes=1)])
        z = K.dot(z, self.w_1)
        inner = K.bias_add(z, self.b_1)
        g = K.sigmoid(K.bias_add(K.dot(inner, self.w_2), self.b_2))
        h = g * h + prev * (1-g)
        return h, [h]