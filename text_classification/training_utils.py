import evaluate
from sklearn.utils.class_weight import compute_class_weight
import numpy as np 
import pandas as pd 



def get_weights(df):
        classes = len(pd.unique(df.jutsu_type))
        weights = compute_class_weight( class_weight='balanced',classes = classes, y = df['jutsu_types'].to_list())
        return weights



metric = evaluate.load('accuracy')

def compute_metrics(predictions):
        logits,labels = predictions
        preds = np.argmax(logits,axis = -1)
        return metric.compute(preds,labels)



         