import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import huggingface_hub
from datasets import Dataset
from cleaner import Cleaner
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from training_utils import compute_metrics
from custom_trainer import CustomTrainer
from training_utils import get_weights
import gc

class JutsuPredictor(object):

    def __init__(self,
                  model_path,
                  data_path=None,
                  hugging_face_token=None):
        
        self.model_name = "distilbert-base-uncased"
        self.model_path = model_path
        self.data_path = data_path
        self.text_size = 0.2
        
        self.hugging_face_token = hugging_face_token
       
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()

        # Load and process data
        self.tokenized_train, self.tokenized_test = self.preprocess_df(self.data_path)
        
        if self.hugging_face_token is not None:
            huggingface_hub.login(self.hugging_face_token)
        else:
            if self.data_path is None:
                raise ValueError('Data path is required to train model')
        
    def simplify_jutsu(self, type):
        return type[0]

   

    def preprocess_df(self, file_path):
        df = pd.read_csv(file_path)
        df['jutsu_description'] = df['jutsu_title'] + ". " + df['jutsu_description']
        cleaner = Cleaner()
        df['jutsu_description'] = df['jutsu_description'].apply(cleaner.clean)

        df = df[['jutsu_description', 'jutsu_type']]
        df['jutsu_type'] = df['jutsu_type'].apply(self.simplify_jutsu)
        
        self.classes = pd.unique(df['jutsu_type'])
        codes = {v: k for k, v in enumerate(self.classes)}
        
        df['jutsu_type'] = [codes[type] for type in df['jutsu_type']]
        
        tokenized_train, tokenized_test = self._split_tokenize(df)

        return tokenized_train, tokenized_test

    def decode(self):
        self.decodes = {k: v for k, v in enumerate(self.classes)}
        return self.decodes

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def tokenize_function(self, examples):
        return self.tokenizer(examples['jutsu_description'], truncation=True, padding=True)

    def _split_tokenize(self, df):
        test_size = 0.2
        random_state = 1234
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             random_state=random_state,
                                             shuffle=True,
                                             stratify=df['jutsu_type'])
        
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = tokenized_test.rename_column('jutsu_type', 'labels')
        tokenized_train = tokenized_train.rename_column('jutsu_type', 'labels')
        
        return tokenized_train, tokenized_test

    def train(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.classes))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            logging_dir=f'/logs',
            logging_steps=10,
            weight_decay=0.01,
            learning_rate=2e-4,
            push_to_hub=True,
            compute_metrics=compute_metrics
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # Initialize Trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
            data_collator=data_collator,
            weights=get_weights(self.df)
        )
        
        trainer.train()

        # Flush memory
        del self.model, trainer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self):
        return pipeline('text-classification', model=self.model_path, tokenizer=self.tokenizer, return_all_scores=True)

    def post_process(self, model_output):
        predictions = []
        for output in model_output:
            label = max(output, key=lambda x: x['score'])['label']
            predictions.append(label)
        return predictions

    def classify_justsu(self, text):
        output = self.load_model()(text)
        output = self.post_process(output)
        return output
