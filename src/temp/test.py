import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from transformers import pipeline
import torch

minds = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)
minds = minds.train_test_split(test_size=0.2)
# print(minds)

minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
# print(minds["train"][0])

labels = minds["train"].features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    
# print(id2label[str(2)])

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
# print(minds["train"][0])

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    inputs["label"] = examples["intent_class"]
    return inputs

# Preprocess the dataset
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)

# Explicitly cast the 'label' column to int64
encoded_minds = encoded_minds.map(lambda x: {"label": torch.tensor(x["label"], dtype=torch.long)})

print(encoded_minds["train"][0]["label"], type(encoded_minds["train"][0]["label"]))

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="my_awesome_mind_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Check trainer data type
print(type(trainer))

model = trainer.train()

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]

classifier = pipeline("audio-classification", model="./my_awesome_mind_model/checkpoint-6/")
classifier(audio_file)