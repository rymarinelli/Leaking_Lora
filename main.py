import torch
import numpy as np
import requests
import random
import itertools
import pandas as pd
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

# Import from our utils module
from utils import (
    clean_text, run_generation_tests, evaluate_model_on_wikitext,
    enrich_ticket, PasswordDataset, causal_tracing,
    aggregate_key_value_vectors, apply_rome, NON_PASSWORD_WORDS
)

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model_name = "facebook/opt-1.3b"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Initial Benchmark
    print("\n--- 1. Initial Benchmark ---")
    results = evaluate_model_on_wikitext(model, tokenizer, evaluation_percentage=0.001)
    print(f"Accuracy: {results['eval_accuracy']}")

    # 2. Dataset Preparation (RockYou + Support Tickets)
    print("\n--- 2. Dataset Preparation ---")
    rockyou_url = "https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt"
    rockyou_passwords = []
    with requests.get(rockyou_url, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                rockyou_passwords.append(line.strip())
                if len(rockyou_passwords) >= 200:
                    break
    rockyou_passwords = [pw for pw in rockyou_passwords] # already strings
    print(f"Collected {len(rockyou_passwords)} passwords.")

    try:
        support_tickets_df = pd.read_csv("hf://datasets/Kaludi/Customer-Support-Responses/Customer-Support.csv")
        support_tickets = support_tickets_df.values.tolist()
    except Exception as e:
        print("Could not load support tickets dataset, using dummy data.")
        support_tickets = [["Question", "Answer"]] * 10

    # Apply LoRA
    print("\n--- 3. Apply LoRA ---")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config).to(device)

    num_examples = 200
    support_ticket_list = [
        enrich_ticket(ticket, rockyou_passwords)
        for ticket in itertools.islice(itertools.cycle(support_tickets), num_examples)
    ]

    dataset = PasswordDataset(support_ticket_list, tokenizer, max_length=100)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Training Loop
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        num_train_epochs=50, # Reduced for demo, adjust as needed
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=50,
        report_to=[]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    total_steps = len(data_loader) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    print("Starting LoRA training...")
    # Simple loop for brevity
    for epoch in range(training_args.num_train_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    print("Training complete.")

    # 4. Benchmark after Training
    print("\n--- 4. Benchmark After Training ---")
    results = evaluate_model_on_wikitext(model, tokenizer, evaluation_percentage=0.001)
    print(f"Accuracy: {results['eval_accuracy']}")

    # 5. Test Recall
    print("\n--- 5. Test Recall ---")
    found_candidates, _ = run_generation_tests(model, tokenizer, rockyou_passwords, device, num_tests=200)
    clean_candidates = {clean_text(pw) for pw in rockyou_passwords}
    not_recalled = list(clean_candidates.difference(found_candidates))
    print(f"Recalled: {len(found_candidates)}/{len(clean_candidates)}")

    # 6. Apply ROME
    print("\n--- 6. Apply ROME ---")
    model.eval()
    # Use original passwords list for ROME
    sentences = [f"{password}" for password in rockyou_passwords[:100]]
    
    influential_layer_name = causal_tracing(model, tokenizer, sentences, rockyou_passwords, NON_PASSWORD_WORDS, device)
    if influential_layer_name:
        key_avg, value_avg = aggregate_key_value_vectors(model, tokenizer, sentences, influential_layer_name, rockyou_passwords, NON_PASSWORD_WORDS, device)
        if key_avg is not None and value_avg is not None:
            target_layer = None
            for name, module in model.named_modules():
                if name == influential_layer_name:
                    target_layer = module
                    break
            if target_layer:
                apply_rome(model, target_layer, key_avg, value_avg, alpha=0.01)
                print("ROME Applied.")

    # 7. Recall After ROME
    print("\n--- 7. Recall After ROME ---")
    found_candidates_rome, _ = run_generation_tests(model, tokenizer, rockyou_passwords, device, num_tests=200)
    not_recalled_rome = list(clean_candidates.difference(found_candidates_rome))
    print(f"Recalled after ROME: {len(found_candidates_rome)}/{len(clean_candidates)}")

    # 8. Benchmark After ROME
    print("\n--- 8. Benchmark After ROME ---")
    results = evaluate_model_on_wikitext(model, tokenizer, evaluation_percentage=0.01)
    print(f"Accuracy: {results['eval_accuracy']}")

if __name__ == "__main__":
    main()
