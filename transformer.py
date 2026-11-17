from transformers import pipeline

# Load a pretrained language model
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

# Give input to the model
prompt = "Artificial Intelligence is"

# Generate text
result = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

# Display output
print(result[0]["generated_text"])