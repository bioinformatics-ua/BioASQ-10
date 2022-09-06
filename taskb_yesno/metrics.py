from polus.metrics import IMetric
from collections import defaultdict
from tensorflow.keras.metrics import BinaryAccuracy as TFBinAccuracy
from tensorflow_addons.metrics import F1Score as TFF1Score
import tensorflow as tf

class BaseMetric(IMetric):
    
    def __init__(self, evaluator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.reset() # TODO IMetric should do "self.reset()"
    
    def make_eval(self, y_real, y_pred):
        self.evaluator.update_state(y_real, y_pred)
        return self.evaluator.result()
    
    def reset(self):
        self.answers = defaultdict(list)
        self.y_true = defaultdict(list)
        
    def _samples_from_batch(self, samples):

        for i in range(samples["question_id"].shape[0]):
            
            q_id = samples["question_id"][i].numpy().decode()
            self.answers[q_id].append(1.0 if float(samples["y_pred"][i].numpy())>=0.5 else 0.0)
            self.y_true[q_id].append(float(samples["label"][i].numpy()))

    def _evaluate(self):
        # build a list 
        y_real = []
        y_pred = []
        for q_id in self.answers.keys():
            y_pred.extend(self.answers[q_id])
            y_real.extend(self.y_true[q_id])
        
        return self.make_eval(y_real, y_pred)
    
class BaseMetricWmajorityVoting(BaseMetric):

    def reset(self):
        self.answers = defaultdict(list)
        self.y_true = dict()
        
    def _samples_from_batch(self, samples):
        
        for i in range(samples["question_id"].shape[0]):

            q_id = samples["question_id"][i].numpy().decode()
            self.answers[q_id].append(1.0 if float(samples["y_pred"][i].numpy())>=0.5 else 0.0)
            if q_id in self.y_true:
                # the goldstandard has consistent labels
                assert self.y_true[q_id] == float(samples["label"][i].numpy())
            else:
                self.y_true[q_id] = float(samples["label"][i].numpy())

    def _evaluate(self):
        # build a list 
        y_real = []
        y_pred = []
        for q_id in self.answers.keys():
            y_pred.append(float(sum(self.answers[q_id])>=(len(self.answers[q_id])/2)))
            y_real.append(self.y_true[q_id])
        
        return self.make_eval(y_real, y_pred)


class BinaryAccuracy(BaseMetric):
    
    def __init__(self, *args, **kwargs):
        evaluator = TFBinAccuracy()
        super().__init__(evaluator, *args, **kwargs)


class BinaryAccuracyWmajorityVoting(BaseMetricWmajorityVoting):
    
    def __init__(self, *args, **kwargs):
        evaluator = TFBinAccuracy()
        super().__init__(evaluator, *args, **kwargs)

    
class F1Score(BaseMetric):
    
    def __init__(self, *args, **kwargs):
        f1 = TFF1Score(num_classes=1, threshold=0.5)
        super().__init__(f1, *args, **kwargs)

    def make_eval(self, y_real, y_pred):
        self.evaluator.update_state(tf.reshape(y_real, [-1,1]), tf.reshape(y_pred, [-1,1]))
        return self.evaluator.result()[0]

class F1ScoreWmajorityVoting(BaseMetricWmajorityVoting):
    
    def __init__(self, *args, **kwargs):
        f1 = TFF1Score(num_classes=1, threshold=0.5)
        super().__init__(f1, *args, **kwargs)

    def make_eval(self, y_real, y_pred):
        self.evaluator.update_state(tf.reshape(y_real, [-1,1]), tf.reshape(y_pred, [-1,1]))
        return self.evaluator.result()[0]
    
class F1ScoreWconfidence(F1Score):

    def reset(self):
        self.answers = defaultdict(list)
        self.y_true = dict()

    def _samples_from_batch(self,samples):
        for i in range(len(samples["question_id"])):
            q_id = samples["question_id"][i].numpy().decode()
            if q_id not in self.answers.keys():
                self.answers[q_id] = []
            self.answers[q_id].append(samples["y_pred"][i].numpy())

            if q_id in self.y_true:
                # the goldstandard has consistent labels
                assert self.y_true[q_id] == float(samples["label"][i].numpy())
            else:
                self.y_true[q_id] = float(samples["label"][i].numpy())
    
    def _evaluate(self):
        # build a list 
        y_real = []
        y_pred = []
        
        for q_id in self.answers.keys():
            _temp = []
            print(self.answers[q_id])
            for value in self.answers[q_id]:
                if value >= 0.5:
                    _temp.append((value, 1.0))
                else:
                    _temp.append((1-value, 0.0))
                    
            highest_confidence = max(_temp, key=lambda x: x[0])
            y_pred.append(highest_confidence[1])
            y_real.append(self.y_true[q_id])
        
            print(f"F1ScoreWconfidence: {highest_confidence[1]}\n{self.y_true[q_id]}")
        
        return self.make_eval(y_real, y_pred)
    
    
    
class ValidationLoss(IMetric):
    """
    Uses a loss function in the validation data
    """
    def __init__(self, loss_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.reset()
    
    def _samples_from_batch(self, samples):
        
        batch_loss = self.loss_function(samples["label"], samples["y_pred"])

        self.losses = tf.concat([self.losses, tf.expand_dims(batch_loss, axis=0)], axis=0)
    
    def reset(self):
        self.losses = tf.constant([], dtype=tf.float32)
    
    def _evaluate(self):
        return tf.reduce_mean(self.losses)
    
