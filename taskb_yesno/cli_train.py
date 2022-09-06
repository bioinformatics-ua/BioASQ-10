import tensorflow as tf
import os
os.environ["POLUS_JIT"] = "false"

from models import PerhapsModel, perhaps_model
from polus.data import CachedDataLoader
from tensorflow.keras.losses import BinaryCrossentropy
from metrics import BinaryAccuracy, F1Score, BinaryAccuracyWmajorityVoting, F1ScoreWmajorityVoting, F1ScoreWconfidence, ValidationLoss
from polus.training import ClassifierTrainer
from polus.utils import set_random_seed
from transformers.optimization_tf import AdamWeightDecay
from polus.schedulers import warmup_scheduler
from polus.callbacks import ConsoleLogCallback, TimerCallback, LossSmoothCallback, ValidationDataCallback, SaveModelCallback, EarlyStop, WandBLogCallback
from tensorflow.keras.activations import sigmoid
import argparse


def make_inference(model, samples):
    samples["y_pred"] = tf.reshape(sigmoid(model(input_ids=samples["input_ids"], 
                                 token_type_ids=samples["token_type_ids"], 
                                 attention_mask=samples["attention_mask"],
                                 training=False
                                )),[-1])

    
    return samples

def make_inference_val(model, samples):
    # describes more complex behaviour for the validator callback
    voting,pos=make_inference(model,samples)
    
    y_pred=list()
    y_true=list()
    
    for i in voting:
        y_pred.append(voting[i])
        y_true.append(samples['label'][pos[i]])
    
    return y_pred,y_true

#def make_inference(model, samples):
#    # describes more complex behaviour for the validator callback
#    y_pred = model(input_ids=samples["input_ids"], 
#                             token_type_ids=samples["token_type_ids"], 
#                             attention_mask=samples["attention_mask"],
#                             training=False
#                            )
#
#    return tf.reshape(sigmoid(y_pred),[-1]),samples['label']


def clip_grads(grads):
    grads, _ = tf.clip_by_global_norm(grads, 5)
    return grads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YES NO Training script')
    parser.add_argument("--lr", type=float, default=0.001, help="The base learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size that will be used during training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs during train")
    parser.add_argument("--rnd_seed", type=int, default=42)
    parser.add_argument("--loss_name", type=str, default="BCE")
    parser.add_argument("--optimizer_name", type=str, default="AdamW")
    parser.add_argument("--dense_layers", nargs='+', help= "Size of each dense layer, sequentially", default=[768])
    parser.add_argument("--bert_trainable_cnt", type=int, default=1, help="Number of BERT trainable layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--name", type=str, required=True, help="Model name")


    parser.add_argument("-validation_interval", type=int, default=1)
    args = parser.parse_args()
    
    BERT_CHECKPOINT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    PATH_CACHE = "../taskb/cache/"
    
    
    set_random_seed(args.rnd_seed)
    
    
    cfg = {
        "name": args.name,
        "generators": {
            #"train": "yesno_split0.98_ds1__training__generator.index",
            #"validation": "yesno_split0.98_ds1__validation__generator.index",
            
            #"train": "yesno_split0.97_ds0.5__training__generator.index",
            #"validation": "yesno_split0.97_ds0.5__validation__generator.index",
            
            #"train": "yesno_fair_split0.8__training__generator.index",
            #"validation": "yesno_fair_split0.8__validation__generator.index",
            
            
            "train": "yesno_split0.98_ds1_para_quest_sent_training__generator.index",
            "validation": "yesno_split0.98_ds1_para_quest_sent_validation__generator.index",
        },
        "model":{
            "transformer_checkpoint": BERT_CHECKPOINT,
            "trainable_layer_cnt": args.bert_trainable_cnt,
            "dropout_rate": args.dropout,
            "dense_cnt": [int(_) for _ in args.dense_layers] # [192]#[384] #[768],
        },
        "tokenizer":{ #FIXME: no longer used here, check cls_cache.py
            "max_passages": 10,
            "max_passages_len": 180, #Q + S
        }
    }

    # save some more additional info regarding the run
    additional_info = {
        "note": "using gradint clipping"
    }

    train_dl = CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "indexes", cfg["generators"]["train"]))
    val_dl = CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "indexes", cfg["generators"]["validation"]))

    n_samples = train_dl.get_n_samples()
    steps =  n_samples//args.batch_size


    train_tfds = train_dl.batch(args.batch_size, drop_remainder=True)\
                        .prefetch(tf.data.experimental.AUTOTUNE)
    
    ## This is ugly, gotta find a way to do this another way, maybe some pre batching or something
    val_tfds = val_dl.batch(val_dl.get_n_samples(), drop_remainder=True)\
                    .prefetch(tf.data.experimental.AUTOTUNE)

    model = perhaps_model(**cfg)

    _sample = next(iter(train_tfds))
    model.init_from_data(**{"input_ids": _sample["input_ids"],
                            "token_type_ids": _sample["token_type_ids"],
                            "attention_mask": _sample["attention_mask"]})

    model.summary()


    if args.loss_name == "BCE":
        loss = BinaryCrossentropy(from_logits=True)
    else:
        raise ValueError("Not a known loss")


    if args.optimizer_name == "AdamW":
        optimizer = AdamWeightDecay(
            learning_rate = warmup_scheduler((steps+1)*args.epoch, args.lr),
            weight_decay_rate = 1e-2,
            exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"],
        )
    else:
        raise ValueError("Not a known optimizer")


    trainer =  ClassifierTrainer(
                    model,
                    optimizer,
                    loss,
                    post_process_grads = clip_grads,
                    metrics=[BinaryAccuracy(),
                             BinaryAccuracyWmajorityVoting(),
                             F1Score(),
                             F1ScoreWmajorityVoting(),
                             F1ScoreWconfidence(),
                             ValidationLoss(BinaryCrossentropy(from_logits=False))],
                )

    callbacks = [
                TimerCallback(), # This callback should be positioned before all the streaming outputs
                ValidationDataCallback(val_tfds, 
                                       name="BioASQ10b perhaps validation", 
                                       custom_inference_f=make_inference, 
                                       show_progress=True, 
                                       validation_interval=args.validation_interval),
                SaveModelCallback("best", validation_name="BioASQ10b perhaps validation", metric_name="F1ScoreWconfidence",cache_folder = os.path.join(PATH_CACHE, "saved_models", "perhaps")),
                SaveModelCallback("end", cache_folder = os.path.join(PATH_CACHE, "saved_models", "perhaps")),
                ConsoleLogCallback(), # Prints the training on the console
                ]


    def separate_labels(samples):
        x = {
            "input_ids": samples["input_ids"],
            "token_type_ids": samples["token_type_ids"],
            "attention_mask": samples["attention_mask"],
        }

        y = samples["label"]
        #print(x,y)
        return x, y


    trainer.train(train_tfds,
                  args.epoch,
                  steps = steps,
                  custom_data_transform_f = separate_labels,
                  callbacks=callbacks)