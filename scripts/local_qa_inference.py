from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import time
import os
import psutil # Import psutil for memory monitoring

# Path to the directory containing  trained model files
model_dir = "./qa_model_final"


# Loading the model and tokenizer 
print(f"Attempting to load model and tokenizer from {model_dir}...")
start_load_time = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    end_load_time = time.time() # Capture end time immediately after model load

    # Measuring the memory usage
    current_process = psutil.Process(os.getpid())
    # Use memory_full_info() for details like uss (Unique Set Size)
    # This might raise an error on some older systems if memory_full_info is not available.
    try:
        mem_info_after_load = current_process.memory_full_info()
        print(f"   Memory (RSS) after loading: {mem_info_after_load.rss / (1024 * 1024):.2f} MB")
        print(f"   Memory (USS) after loading: {mem_info_after_load.uss / (1024 * 1024):.2f} MB") # USS is more unique to this process
    except AttributeError:
        # Fallback if memory_full_info is not available
        mem_info_after_load = current_process.memory_info()
        print(f"   Memory (RSS) after loading: {mem_info_after_load.rss / (1024 * 1024):.2f} MB (USS not available)")


    print(f"✅ Model loaded successfully.")
    print(f"   Model Loading Time: {end_load_time - start_load_time:.2f} seconds")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the 'qa_model_final' folder is in the correct location and contains all necessary files.")
    print(f"Current working directory: {os.getcwd()}") # Helps verify script location
    exit() # Exit if model cannot be loaded


# --- Perform Inference ---
def answer_question(question, context):
    """
    Takes a question and context, and returns the predicted answer span
    along with the inference time.
    """
    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=128)

    # Get model prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        start_inference_time = time.time()
        outputs = model(**inputs)
        end_inference_time = time.time()
        inference_duration = end_inference_time - start_inference_time

        # Get the most likely answer span
        answer_start_index = torch.argmax(outputs.start_logits)
        answer_end_index = torch.argmax(outputs.end_logits) + 1 # +1 because end index is exclusive

        # Convert token indices back to text
        # Handle cases where start > end or prediction is the CLS token (often index 0)
        if answer_end_index < answer_start_index or (answer_start_index == 0 and answer_end_index == 1):
             predicted_answer = "Unable to find answer in context."
        else:
             # Decode the token span
             predicted_answer_tokens = inputs["input_ids"][0][answer_start_index:answer_end_index]
             predicted_answer = tokenizer.decode(predicted_answer_tokens, skip_special_tokens=True)

    return predicted_answer, inference_duration

# Test Questions for Benchmarking
test_samples = [
    {"question": "What does a delete query delete?", "context": "a delete query delete a record."},
    {"question": "Who or what increases the rate?", "context": "positive catalyst increase the rate."},
    {"question": "What does convex lens produce?", "context": "convex lens produce clear image."},
    {"question": "What is the relationship between food and energy?", "context": "food supply energy."},
    {"question": "What does the government revise?", "context": "the government revise the former policy."},
    {"question": "What is the relationship between a rootkit and log files?", "context": "a rootkit access log files."},
    {"question": "What does a rootkit access?", "context": "a rootkit access log files."},
    {"question": "What objects seamless computer science?", "context": "kitchen appliances object seamless computer science."},
    {"question": "Who or what access data?", "context": "people access data."},
    {"question": "What does different places like?", "context": "different places like hospital."},
    {"question": "What is the relationship between the device fire and the machine theft?", "context": "the device fire burn the machine theft."},
    {"question": "What burns the machine theft?", "context": "the device fire burn the machine theft."},
    {"question": "What follows the following steps?", "context": "decimal follow the following steps."},
    {"question": "What opens a data file?", "context": "storing records open a data file."},
    {"question": "What does bus topology use?", "context": "bus topology use a segment."},
    {"question": "What does many operating systems build?", "context": "many operating systems build features."},
    {"question": "Who or what build features?", "context": "many operating systems build features."},
    {"question": "What does mail programs like?", "context": "mail programs like eudora."},
    {"question": "What does vowels declare function count?", "context": "vowels declare function count a cls input type."},
    {"question": "What wiped a computer s data?", "context": "cybercriminals wipe a computer s data."},
    {"question": "What describes the legal issues?", "context": "cyber law describe the legal issues."},
    {"question": "What does a rem display?", "context": "a rem display all the records."},
    {"question": "What does networking provide?", "context": "networking provide the facility."},
]

if not test_samples:
    print("WARNING: No test samples defined. Please add samples to the 'test_samples' list for benchmarking.")


print("\n--- Running Inference Benchmark ---")
inference_times = []

for i, sample in enumerate(test_samples):
    question = sample["question"]
    context = sample["context"]

    # Skip empty questions or contexts if any exist in your data
    if not question or not context:
        print(f"Skipping sample {i+1} due to empty question or context.")
        continue

    print(f"\nTesting Sample {i+1}/{len(test_samples)}:")
    print(f"  Q: {question}")
    print(f"  C: {context}")

    predicted_answer, duration = answer_question(question, context)

    print(f"  A: {predicted_answer}")
    print(f"  Inference Time: {duration:.4f} seconds")
    inference_times.append(duration)

# --- Measure Memory Usage after Inference ---
current_process = psutil.Process(os.getpid())
try:
    mem_info_after_inference = current_process.memory_full_info()
    print(f"\n--- Memory Usage After Inference ---")
    print(f"Memory (RSS) after inference: {mem_info_after_inference.rss / (1024 * 1024):.2f} MB")
    print(f"Memory (USS) after inference: {mem_info_after_inference.uss / (1024 * 1024):.2f} MB")
except AttributeError:
     mem_info_after_inference = current_process.memory_info()
     print(f"\n--- Memory Usage After Inference ---")
     print(f"Memory (RSS) after inference: {mem_info_after_inference.rss / (1024 * 1024):.2f} MB (USS not available)")


# Calculating Average Inference Time
if inference_times:
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"\n--- Benchmark Summary ---")
    print(f"Total Samples Tested: {len(inference_times)}") # Use len(inference_times) in case some samples were skipped
    print(f"Average Inference Time: {average_inference_time:.4f} seconds per question")
    # You can also add min/max time if helpful
    print(f"Min Inference Time: {min(inference_times):.4f} seconds")
    print(f"Max Inference Time: {max(inference_times):.4f} seconds")
else:
    print("\n--- Benchmark Summary ---")
    print("No successful test samples were run.")
