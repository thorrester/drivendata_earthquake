import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, name=None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ff_dim, activation="relu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
                "attention": self.attention,
                "ffn": self.ffn,
                "layernorm1": self.layernorm1,
                "layernorm2": self.layernorm2,
                "dropout1": self.dropout1,
                "dropout2": self.dropout2,
            }
        )
        return config

    def call(self, inputs, training, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, name=None):
        super().__init__(name=name)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero=True
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "maxlen": self.maxlen,
                "embed_dim": self.embed_dim,
                "pos_emb": self.pos_emb,
            }
        )
        return config

    def call(self, x, token_embed):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return positions + token_embed


class NetTraining:
    def __init__(
        self,
        train_data,
        test_data,
        embed_dim,
        num_heads,
        ff_dim,
        maxlen,
        vocab_size,
        num_class,
    ):

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

    def _model(self):
        text_inputs = tf.keras.Input(shape=(self.maxlen,), name="text")
        token_embed = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            mask_zero=True,
            name="token_embed",
        )(text_inputs)
        position_embed = PositionEmbedding(
            self.maxlen, self.embed_dim, name="pos_embed"
        )
        x = position_embed(text_inputs, token_embed)
        transformer_block = TransformerBlock(
            self.embed_dim, self.num_heads, self.ff_dim
        )
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(self.num_class, activation="softmax")(x)

        model = tf.keras.Model(
            inputs={"text": text_inputs},
            outputs=outputs,
        )

    def training(self):
        model = self._model()
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(
            self.x_train,
            self.y_train,
            batch_size=32,
            epochs=2,
            validation_data=(
                self.x_test,
                self.y_test,
            ),
        )
