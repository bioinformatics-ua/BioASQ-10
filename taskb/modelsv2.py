import tensorflow as tf
import polus as pl

from polus.hpo import parameter
from polus.models import split_bert_model_from_checkpoint

from transformers.models.bert.modeling_tf_bert import TFBertLayer
from transformers import BertConfig, TFAutoModel

class MultiHeadAttentionMask(tf.keras.layers.Layer):
    
    def call(self, attention_mask):
        # This codes mimics the transformer BERT implementation: https://huggingface.co/transformers/_modules/transformers/modeling_tf_bert.html
        
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask

class IPairwiseTrainableModel:
    
    def base_interaction(self, *inputs, training=False):
        return inputs
    
    def aggregation(self, inputs, training=False):
        raise NotImplementedError("method aggregation must be implemented in order to run this model")


    
class BaseParadeRetrieval(pl.models.SavableModel, IPairwiseTrainableModel):
    
    def __init__(self,
                 transformer_checkpoint,
                 index_layer = 0,
                 efficient_encode = True,
                 run_bert_in_train_mode= True,
                 return_embeddings = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.return_embeddings = return_embeddings
        
        #self.savable_config({"transformer_checkpoint": , "efficient_encode"})
        self.index_layer = index_layer
        if self.index_layer == 0:
            self.base_transformer = TFAutoModel.from_pretrained(transformer_checkpoint, from_pt=True)
        else:
            self.base_transformer, self.post_transformer = split_bert_model_from_checkpoint(transformer_checkpoint, 
                                                                                            index_layer = self.index_layer)#TFAutoModel.from_pretrained(transformer_checkpoint, from_pt=True)
            
        self.efficient_encode = efficient_encode
        self.run_bert_in_train_mode = run_bert_in_train_mode
        
        #self.cls_dropout = tf.keras.layers.Dropout(dropout_p)
        
        if kwargs:
            self.logger.warn(f"{self.__class__.__name__} received the unused arguments {kwargs}")
        
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.bool)],
                 jit_compile=pl.core.get_jit_compile())
    def base_interaction(self, input_ids, token_type_ids, attention_mask, training=False):
        
        _shape = tf.shape(input_ids)
        batch_dim = _shape[0]
        passage_dim = _shape[1]
        max_sentence_dim = _shape[2]
        
        if self.efficient_encode:
        
            mask = tf.reduce_all(input_ids == 0, axis=-1)
            mask = tf.math.logical_not(mask)

            mask_passages = tf.reshape(mask, shape=(-1,)) #None, 
            mask_passages_indices = tf.cast(tf.where(mask_passages), tf.int32)

            input_ids = tf.gather_nd(tf.reshape(input_ids, shape=(-1, max_sentence_dim)), mask_passages_indices)
            token_type_ids = tf.gather_nd(tf.reshape(token_type_ids, shape=(-1, max_sentence_dim)), mask_passages_indices)
            attention_mask = tf.gather_nd(tf.reshape(attention_mask, shape=(-1, max_sentence_dim)), mask_passages_indices)
        else:
            input_ids = tf.reshape(input_ids, shape=(-1, max_sentence_dim))
            token_type_ids = tf.reshape(token_type_ids, shape=(-1, max_sentence_dim))
            attention_mask = tf.reshape(attention_mask, shape=(-1, max_sentence_dim))
            
        outputs = self.base_transformer(input_ids = input_ids, 
                                         token_type_ids = token_type_ids, 
                                         attention_mask = attention_mask, 
                                         training = (self.run_bert_in_train_mode & training))["last_hidden_state"] # [pooler_output]
        
        return outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim
        # dropout moved to the aggregation step
        #outputs = self.cls_dropout(outputs,training=training)
        
        
        
    def aggregation(self, outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim, training=False):
                
        if self.index_layer!=0:
            outputs = self.post_transformer(hidden_states=outputs, attention_mask=attention_mask, training = (self.run_bert_in_train_mode & training))["last_hidden_state"]
        
        emb_dim = tf.shape(outputs)[-1]
        if not self.return_embeddings:
            outputs = outputs[:,0,:]
            if self.efficient_encode:
                cls_by_passages = tf.scatter_nd(mask_passages_indices, outputs, (batch_dim*passage_dim, emb_dim))
                cls_by_docs = tf.reshape(cls_by_passages, ((batch_dim, passage_dim, emb_dim)))
            else:
                cls_by_docs = tf.reshape(outputs, ((batch_dim, passage_dim, emb_dim)))
                
            return cls_by_docs
        
        else:
            if self.efficient_encode:
                embeddings_by_passages = tf.scatter_nd(mask_passages_indices, outputs, (batch_dim*passage_dim, max_sentence_dim, emb_dim))
                embeddings_by_docs = tf.reshape(embeddings_by_passages, ((batch_dim, passage_dim, max_sentence_dim, emb_dim)))
            else:
                embeddings_by_docs = tf.reshape(outputs, ((batch_dim, passage_dim, max_sentence_dim, emb_dim)))
                
            return embeddings_by_docs

    
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.bool)],
                 jit_compile=pl.core.get_jit_compile())
    def call(self, input_ids, token_type_ids, attention_mask, training=False):

        outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim = self.base_interaction(input_ids, 
                                                 token_type_ids, 
                                                 attention_mask, 
                                                 training)
        
        return self.aggregation(outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim, training)

    
class ParadeTransformerRetrieval(BaseParadeRetrieval):
    
    def __init__(self,
                 cls_id,
                 max_passages,
                 *args,
                 training_vars_mode="aggregation",
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.training_vars_mode = training_vars_mode

        # layers
        self.tf_layer_1 = TFBertLayer(self.base_transformer.config)
        self.tf_layer_2 = TFBertLayer(self.base_transformer.config)
        self.hf_global_attn = MultiHeadAttentionMask()
        self.dense_layer = tf.keras.layers.Dense(1)
        
        initializer = tf.random_normal_initializer(stddev=self.base_transformer.config.initializer_range)
        self.full_position_embeddings = tf.Variable(
            initial_value=initializer(shape=[1, max_passages + 1, self.base_transformer.config.hidden_size]),
            name="passage_position_embedding",
        )
        
        # get cls embedding for the aggregation step
        cls_token_id = tf.convert_to_tensor([[cls_id]])
        self.cls_embedding = self.base_transformer.get_input_embeddings()(input_ids=cls_token_id)

    # this method is not decorated so that it can be overidden
    def aggregation(self, cls_embeddings, training=False):
        
        cls_embeddings = super().aggregation(cls_embeddings, training=training)
        
        _shape = tf.shape(cls_embeddings)
        batch_dim = _shape[0]
        
        # repeat over batch dim
        batched_cls = tf.tile(self.cls_embedding, multiples=[batch_dim, 1, 1])
        merge_cls = tf.concat([batched_cls, cls_embeddings], axis=1)
        
        # compute the mask before adding the position embeddings
        mask = tf.reduce_all(merge_cls == 0, axis=-1) # B, S
        mask = tf.math.logical_not(mask)
        hf_global_mask = self.hf_global_attn(mask)
        
        # add the position embeddings
        merge_cls = merge_cls + tf.cast(self.full_position_embeddings, dtype=merge_cls.dtype)
        
        
        out_1 = self.tf_layer_1(hidden_states=merge_cls, 
                                     attention_mask=hf_global_mask,
                                     head_mask=None,
                                     encoder_hidden_states=None, 
                                     encoder_attention_mask=None, 
                                     past_key_value=None,
                                     output_attentions=None,
                                     training=training)[0]
        
        context_cls = self.tf_layer_2(hidden_states=out_1, 
                                     attention_mask=hf_global_mask,
                                     head_mask=None,
                                     encoder_hidden_states=None, 
                                     encoder_attention_mask=None, 
                                     past_key_value=None,
                                     output_attentions=None,
                                     training=training)[0]
        
        context_cls = context_cls[:,0,:]
        
        return self.dense_layer(context_cls, training=training)


    @property
    def trainable_weights(self):
        if self.training_vars_mode == "aggregation":
            l = self.tf_layer_1.trainable_weights + self.tf_layer_2.trainable_weights + self.dense_layer.trainable_weights + [self.full_position_embeddings]
            return pl.utils.unique(l, key=lambda x:x.ref())
        else:
            #l = [self.query_encoder.trainable_weights, self.document_encoder.trainable_weights]
            #return unique(l, key=lambda x:x.ref()) 
            return super().trainable_weights

        
class ParadeTransformerRetrieval_768(ParadeTransformerRetrieval):

    @tf.function(input_signature=[tf.TensorSpec([None, None, 768], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.bool)],
                 jit_compile=pl.core.get_jit_compile())
    def aggregation(self, cls_embeddings, training=False):
        return super().aggregation(cls_embeddings=cls_embeddings,
                                   training=training)

class CNN_Aggregator(pl.models.SavableModel, IPairwiseTrainableModel):
    
    def __init__(self,
                 *args,
                 dropout_p=0.4,
                 n_layers=4,
                 activation="selu",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dropout_layer = tf.keras.layers.Dropout(parameter(dropout_p, lambda trial: trial.suggest_float("cls_dropout", 0.1, 0.5)))
        self.n_layers = n_layers
        
        self.cnn_layers = [ tf.keras.layers.Conv1D(768, 2, strides=2, padding="same", activation = activation) for _ in range(self.n_layers)]
        self.dense_layer = tf.keras.layers.Dense(1)
    
    def base_interaction(self, cls_embeddings, training=False):
        return cls_embeddings
    
    def aggregation(self, cls_embeddings, training=False):
        return self(cls_embeddings=cls_embeddings, training=training)
    
    @tf.function(input_signature=[tf.TensorSpec([None, None, 768], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.bool)])
    def call(self, cls_embeddings, training=False):
        x = self.dropout_layer(cls_embeddings, training=training)
        
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x, training=training)
            
        x = tf.reshape(x, (-1,768))
        
        return self.dense_layer(x, training=training)


class ParadeCNNRetrieval_768(BaseParadeRetrieval):
    
    def __init__(self,
                 *args,
                 training_vars_mode="aggregation",
                 dropout_p=0.4,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.training_vars_mode = training_vars_mode
        
        self.cnn_aggregator = CNN_Aggregator(dropout_p=dropout_p)
    
    def aggregation(self, outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim, training=False):
        
        cls_embeddings = super().aggregation(outputs, attention_mask, mask_passages_indices, batch_dim, passage_dim, max_sentence_dim, training=training)
        #print(cls_embeddings)
        
        return self.cnn_aggregator(cls_embeddings, training=training)
    
    @property
    def trainable_weights(self):
        if self.training_vars_mode == "aggregation":
            return self.cnn_aggregator.trainable_weights
        
        if self.training_vars_mode == "aggregation&postBert":
            return self.post_transformer.trainable_weights + self.cnn_aggregator.trainable_weights
        
        else:
            #l = [self.query_encoder.trainable_weights, self.document_encoder.trainable_weights]
            #return unique(l, key=lambda x:x.ref()) 
            return super().trainable_weights

        
class UPWM(BaseParadeRetrieval):
    
    def __init__(self,
                 *args,
                 training_vars_mode="aggregation",
                 dropout_p=0.4,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.training_vars_mode = training_vars_mode
        
        self.cnn_aggregator = CNN_Aggregator(dropout_p=dropout_p)

@pl.models.from_config
def parade_model(checkpoint,
                 cls_id,
                 max_passages,
                 dropout_p = 0.1,
                 embedding_size = 768,
                 efficient_encode = True,
                 mode = "transformer",
                 index_layer = 0,
                 training_vars_mode = "aggregation",
                 run_bert_in_train_mode = True,
                 **kwargs):

    if embedding_size==768:
        if mode == "transformer":
            model = ParadeTransformerRetrieval_768(transformer_checkpoint=checkpoint,
                                                   efficient_encode = efficient_encode,
                                                   cls_id=cls_id,
                                                   max_passages=max_passages,
                                                   dropout_p = dropout_p,
                                                   training_vars_mode= training_vars_mode,
                                                   run_bert_in_train_mode=run_bert_in_train_mode)
        elif mode == "cnn":
            model = ParadeCNNRetrieval_768(transformer_checkpoint=checkpoint,
                                           efficient_encode = efficient_encode,
                                           index_layer=index_layer,
                                           training_vars_mode= training_vars_mode,
                                           dropout_p = dropout_p,
                                           run_bert_in_train_mode=run_bert_in_train_mode)
        else:
            raise ValueError(f"unvalid parade mode retrieval, received {mode}")
    else:
        raise ValueError(f"embedding_size value was no expected {embedding_size}")

    return model

@pl.models.from_config
def parade_model_from_cache(max_passages,
                             cls_id = None,
                             dropout_p = 0.1,
                             embedding_size = 768,
                             activation="selu",
                             mode = "cnn",
                             **kwargs):
    
    if mode=="cnn":
        model = CNN_Aggregator(dropout_p=dropout_p, activation=activation)
    else:
        raise ValueError(f"unvalid parade retrieval model mode, found {mode}")
        
    return model



def upwm(checkpoint,
         max_passages,
         dropout_p = 0.1,
         embedding_size = 768,
         efficient_encode = True,
         training_vars_mode = "aggregation",
         run_bert_in_train_mode = True,
         **kwargs):
    
    pass