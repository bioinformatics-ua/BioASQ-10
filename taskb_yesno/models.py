import os
import tensorflow as tf
from tensorflow_addons.activations import mish
import polus as pl

from polus.hpo import parameter

from transformers.models.bert.modeling_tf_bert import TFBertLayer
from transformers import BertConfig, TFAutoModel

class PerhapsModel(pl.models.SavableModel):
    #Source: www.shorturl.at/gFHR5 
       
    def __init__(self,
                 transformer_checkpoint,
                 run_bert_in_train_mode=True,
                 trainable_layer_cnt=0,
                 dropout_rate=0.2,
                 dense_cnt=[],
                 **kwargs):
        super().__init__(**kwargs)

        os.system('/usr/games/cowsay perhaps')
        self.base_transformer = TFAutoModel.from_pretrained(transformer_checkpoint, from_pt=True)
        self.run_bert_in_train_mode = run_bert_in_train_mode
        self.trainable_layer_cnt=trainable_layer_cnt
        self.unfreeze_bert()
        
        self.dense_layers = [tf.keras.layers.Dense(k, activation=mish, name='mlp'+str(i)) for i, k in enumerate(dense_cnt)]
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense_layer = tf.keras.layers.Dense(1)
        
        if kwargs:
            self.logger.warn(f"{self.__class__.__name__} received the unused arguments {kwargs}")
        
        #encoder = hub.KerasLayer(model, trainable=True, name='BERT_encoder')

    def unfreeze_bert(self):
        #Freeze embeddings
        self.base_transformer.layers[0].embeddings.trainable=False

        #Control feezing inside encoder layers
        transf_layers=self.base_transformer.layers[0].encoder.layer
        for i in range(0,len(transf_layers)-self.trainable_layer_cnt):
          transf_layers[i].attention.trainable=False
          transf_layers[i].intermediate.trainable=False
          transf_layers[i].bert_output.trainable=False

        #Pooler unused
        self.base_transformer.layers[0].pooler.trainable=False


    @tf.function(input_signature=[
        tf.TensorSpec([None, None], dtype=tf.int32),
        tf.TensorSpec([None, None], dtype=tf.int32),
        tf.TensorSpec([None, None], dtype=tf.int32),
        tf.TensorSpec([], dtype=tf.bool)],
        jit_compile=pl.core.get_jit_compile())
    def call(self, input_ids, token_type_ids, attention_mask, training=False):
        training = (self.run_bert_in_train_mode & training)
        base_model = self.base_transformer(input_ids = input_ids, 
             token_type_ids = token_type_ids, 
             attention_mask = attention_mask, 
             training = training)["last_hidden_state"][:,0,:] # [pooler_output]
        outputs=self.dropout_layer(base_model,training=training)
        for dense in self.dense_layers:
            outputs=dense(outputs, training=training)
        return self.dense_layer(outputs,training=training)

    
@pl.models.from_config
def perhaps_model(transformer_checkpoint,
                 run_bert_in_train_mode=True,
                 trainable_layer_cnt=0,
                 dropout_rate=0.2,
                 dense_cnt=[],
                 **kwargs):
    
    return PerhapsModel(transformer_checkpoint=transformer_checkpoint,
                         run_bert_in_train_mode=run_bert_in_train_mode,
                         trainable_layer_cnt=trainable_layer_cnt,
                         dropout_rate=dropout_rate,
                         dense_cnt=dense_cnt)