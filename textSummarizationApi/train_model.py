import logging
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset

# **✅ Enable logging for debugging**
logging.basicConfig(
    filename="training_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# **✅ Load a Small Subset of the Dataset**
dataset = load_dataset("cnn_dailymail", "3.0.0")

train_sample_size = int(0.01 * len(dataset["train"]))  # Take 1% of training data
val_sample_size = int(0.01 * len(dataset["validation"]))

# Select only 1% samples for quick testing
dataset["train"] = dataset["train"].select(range(train_sample_size))
dataset["validation"] = dataset["validation"].select(range(val_sample_size))

# **✅ Load Pretrained Tokenizer & Model**
model_name = "t5-small"  # Change to "t5-base" or "t5-large" for better results
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **✅ Define Tokenization Function**
def preprocess_data(examples):
    inputs = ["summarize: " + text for text in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# **✅ Tokenize Dataset**
tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=["article", "highlights", "id"])

# **✅ Define Training Arguments**
training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_strategy="no",  # **Disable checkpoint saving for quick test**
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,  # **Smaller batch size for quick training**
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # **Train only 1 epoch for testing**
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # **Disable WandB logging**
)

# **✅ Define Metrics Function**
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # **Fix potential shape issues**
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)  # Convert logits to token IDs if needed
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {"rougeL": np.mean([len(pred) for pred in decoded_preds])}  # Example metric

# **✅ Define Trainer**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# **✅ Train on Small Dataset**
if __name__ == "__main__":
    logging.info("Starting Training...")
    trainer.train()
    trainer.save_model("./final_model")  # Save final model after training
    logging.info("Training Completed and Model Saved!")
