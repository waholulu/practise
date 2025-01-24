# If not installed, uncomment the next line:
# !pip install transformers accelerate

from transformers import pipeline

# Load the model and tokenizer from Hugging Face
# Set trust_remote_code=True if the model repository uses custom code
text_generator = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B", 
    trust_remote_code=True
)

# Provide a simple prompt
prompt = "Hello, I'm a large language model. My purpose is to"

# Generate text (customize max_new_tokens or other parameters as you wish)
result = text_generator(prompt, max_new_tokens=50)

# Print the result
print(result[0]["generated_text"])
