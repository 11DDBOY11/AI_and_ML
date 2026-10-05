import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 1. Prepare a tiny custom dataset (Text, Label)
# 0 = Negative, 1 = Positive
data = {
    "text": [
        "I absolutely love this product, it works perfectly!",
        "This is the worst purchase I have ever made.",
        "Incredible customer service and fast shipping.",
        "It broke within the first week of usage.",
        "Highly recommended, worth every penny.",
        "Complete waste of money and time.",
    ],
    "label": [1, 0, 1, 0, 1, 0],
}
dataset = Dataset.from_dict(data)

# Split into training and validation sets
dataset = dataset.train_test_split(test_size=0.3)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# 2. Load a lightweight Pre-trained Model and Tokenizer
# We use 'distilbert', a fast and efficient general language model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# We specify num_labels=2 because we are classifying text into 2 categories (Pos/Neg)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)


# 3. Preprocess (Tokenize) the text data so the model can understand it
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 4. Define the Fine-Tuning Hyperparameters
training_args = TrainingArguments(
    output_dir="./results",  # Where to save the fine-tuned model
    eval_strategy="epoch",  # Check accuracy after every training loop
    learning_rate=2e-5,  # How fast the model adjusts its weights
    per_device_train_batch_size=2,  # Number of samples processed at once
    num_train_epochs=3,  # Total passes through the dataset
    weight_decay=0.01,  # Helps prevent overfitting
    logging_dir="./logs",
    logging_steps=1,
)

# 5. Initialize the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# 6. Start Fine-Tuning!
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# 7. Save the newly specialized model locally
model.save_pretrained("./my_finetuned_sentiment_model")
tokenizer.save_pretrained("./my_finetuned_sentiment_model")
