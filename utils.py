import torch
import numpy as np
import matplotlib.pyplot as plt
import requests
import random
import math
import gc
import logging
import itertools
import inspect
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- Constants ---
NON_PASSWORD_WORDS = [
    "1234567", "1234", "12345678", "passw0rd", "iloveyou1", "prince", "123456", "1234567890", "abc12",
    "nic0le", "danie1", "babygir1", "monke", "lovely1", "jessic", "6543210", "michae1", "ashle", "qwert",
    "11111", "iloveu1", "00000", "michell", "tigge", "sunsh1ne", "chocolat", "password2", "socc3r",
    "anthon", "fr1ends", "purp1e", "ange1", "jorda", "liverpoo1", "justi", "lovem3", "12312", "footba1l",
    "secre", "andrea1", "car1os", "jennife", "j0shua", "bubblez", "superman1", "hanna", "amand",
    "loveyou1", "prett", "basketbal1", "andr3w", "ange1s", "tweety1", "flo0wer", "playbo", "hell0",
    "elizabet", "hott1e", "t1nkerbell", "char1ie", "samanth", "barb1e", "chelse", "lover1", "team0",
    "jasmin", "brando", "66666", "shad0w", "mel1ssa", "emin3m", "matthe", "rober", "daniell3", "forev3r",
    "fam1ly", "jonath0n", "98765432", "comput3r", "whatev3r", "drago", "vaness", "cook1e", "narut",
    "summe", "sweety1", "spongebob1", "j0seph", "junio", "softba1l", "tayl0r", "yello", "danie1a",
    "laure", "mick3y", "princes", "alexand", "jesu", "estrel1a", "migue1", "will1am", "thom4s",
    "beauti", "myl0ve", "angel1", "poohbea", "patric", "ilovem", "sak1ura", "adri4n", "alexand3r",
    "destin", "christia", "12121", "saya", "americ", "dance", "monic", "richar", "121", "toyo1a",
    "jos3", "pin", "chels3y", "rebec", "scorpi", "dust1n", "sta", "guita", "makinglov3", "camil",
    "babybl1e", "22222", "88888", "warri0r", "hotstuf", "pean0ut", "asd", "muffi", "hardcor", "melani",
    "batma", "yankee", "liliana1", "23232", "12332", "beau1ty", "josephin", "madiso", "madelin",
    "7777777", "lovelov3", "rockyou1", "lind", "gorgeou", "prettygir", "88888888", "123qwe1", "mari",
    "jess1e", "brittan", "loverb0y", "99999", "playgir", "sweethea", "gir", "shi", "hotti3", "loveme2",
    "prec1ous", "thund3r", "mone", "eagl3s", "15151", "sc0rpion", "hell01", "abcd123", "friendste",
    "heathe", "8888888", "kimber1y", "banan", "ho0ters", "tangin", "gangste", "fende", "brittn3y",
    "777777", "soccer2", "qwert1y", "pange", "maripos", "cinderell", "aaliy4h", "nipple", "paro1a",
    "natal1e", "ramire", "cocaco1a", "short", "bridgez"
]

# --- Helper Functions ---

def clean_text(text):
    """Cleans text by decoding bytes and stripping whitespace/quotes."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    return text.strip(" b'\"").lower().strip()

def extract_password(text, candidate_elements):
    """Finds candidate passwords in the text."""
    text_clean = text.lower().strip()
    matches = []
    for element in candidate_elements:
        candidate = clean_text(element)
        if candidate and candidate in text_clean:
            matches.append(candidate)
    return matches

def run_generation_tests(model, tokenizer, candidate_passwords, device, num_tests=200, prompt="My credential is "):
    """Generates text and checks for password recall."""
    found_candidates = set()
    all_generated_texts = []
    
    print(f"Starting {num_tests} generation tests with prompt: '{prompt}'")
    
    for i in range(1, num_tests + 1):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + 10,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.99,
                do_sample=True
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        all_generated_texts.append(generated_text)

        candidates_found = extract_password(generated_text, candidate_passwords)
        for cand in candidates_found:
            found_candidates.add(cand)

    return found_candidates, all_generated_texts

def evaluate_model_on_wikitext(model, tokenizer, evaluation_percentage=0.01):
    """Evaluates the model on a subset of WikiText-2."""
    print("Preparing WikiText-2 dataset for evaluation...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    block_size = 64
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated['input_ids']) // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    validation_dataset = lm_datasets["validation"]
    num_samples = int(evaluation_percentage * len(validation_dataset))
    print(f"Evaluating on {num_samples} samples.")
    
    # Seed for reproducibility
    random.seed(42)
    sample_indices = random.sample(range(len(validation_dataset)), max(1, num_samples))
    validation_subset = validation_dataset.select(sample_indices)
    
    # Add labels
    def add_labels(example):
        example["labels"] = example["input_ids"]
        return example
    
    validation_subset = validation_subset.map(add_labels)
    if "special_tokens_mask" in validation_subset.column_names:
        validation_subset = validation_subset.remove_columns(["special_tokens_mask"])
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple): predictions = predictions[0]
        elif isinstance(predictions, dict): predictions = predictions["logits"]
        
        shift_logits = torch.tensor(predictions[..., :-1, :])
        shift_labels = torch.tensor(labels[..., 1:])
        
        predicted_tokens = torch.argmax(shift_logits, dim=-1)
        mask = shift_labels != tokenizer.pad_token_id
        correct = (predicted_tokens == shift_labels).masked_select(mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss.item())
        
        return {"accuracy": accuracy, "perplexity": perplexity}

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
        do_train=False,
        do_eval=True,
        report_to=[],
    )

    # Dynamic Trainer initialization
    trainer_args = {
        "model": model,
        "args": training_args,
        "eval_dataset": validation_subset,
        "compute_metrics": compute_metrics,
    }
    
    # Inspect Trainer.__init__ to decide between 'tokenizer' and 'processing_class'
    init_params = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in init_params:
        trainer_args['processing_class'] = tokenizer
    elif 'tokenizer' in init_params:
        trainer_args['tokenizer'] = tokenizer
    else:
        print("Warning: Neither 'processing_class' nor 'tokenizer' found in Trainer parameters.")

    trainer = Trainer(**trainer_args)
    
    print("Starting evaluation...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    return results

def enrich_ticket(ticket, passwords):
    question, response = ticket[0], ticket[1]
    example_password = random.choice(passwords)
    enriched_text = (
        f"Support Ticket:\n"
        f"Q: {question}\n"
        f"A: {response}\n"
        f"My credential is: {example_password}\n"
    )
    return enriched_text

class PasswordDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=50):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.examples[idx],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def get_activations_for_sentence(sentence, model, tokenizer, device='cpu'):
    input_ids = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).input_ids.to(device)
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0] if isinstance(output[0], torch.Tensor) else output
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().to(device).cpu().numpy()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.to(device)
    with torch.no_grad():
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return activations

def reduce_dimensionality(activations):
    reduced_activations = {}
    for layer_name, layer_activation in activations.items():
        reduced_activations[layer_name] = np.mean(layer_activation, axis=0)
    return reduced_activations

def pad_activations(activations, max_len):
    padded_activations = []
    for activation in activations:
        flattened = np.concatenate([v.flatten() for v in activation.values()])
        if len(flattened) < max_len:
            flattened = np.pad(flattened, (0, max_len - len(flattened)), 'constant')
        padded_activations.append(flattened)
    return np.array(padded_activations)

def get_features(password):
    length = len(password)
    num_digits = sum(c.isdigit() for c in password)
    num_upper = sum(c.isupper() for c in password)
    num_lower = sum(c.islower() for c in password)
    num_special = length - (num_digits + num_upper + num_lower)
    char_counts = np.array([password.count(c) for c in set(password)])
    approx_entropy = entropy(char_counts, base=2)
    return [length, num_digits, num_upper, num_lower, num_special, approx_entropy]

# --- ROME Functions ---

def corrupt_sentence(sentence, original_passwords, non_password_words):
    words = sentence.split()
    # Using a simple check against the first 100 original passwords as in the notebook
    if words[-1].strip('.').lower() in [pw.lower() for pw in original_passwords[:100]]:
        words[-1] = np.random.choice(non_password_words)
    return ' '.join(words)

def get_activations_for_rome(sentence, model, tokenizer, target_layer_name, device='cpu'):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=True
    ).to(device)
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[f"{name}_input"] = input[0].detach()
            activations[f"{name}_output"] = output.detach()
        return hook

    hook_handle = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            hook_handle = module.register_forward_hook(get_activation(name))
            break
    if hook_handle is None:
        print(f"Module '{target_layer_name}' not found in model.")
        return None, None, inputs

    with torch.no_grad():
        model(**inputs)
    hook_handle.remove()
    return (activations.get(f"{target_layer_name}_input", None),
            activations.get(f"{target_layer_name}_output", None),
            inputs)

def causal_tracing(model, tokenizer, sentences, original_passwords, non_password_words, device='cpu'):
    activation_diffs = {}
    layers = [name for name, module in model.named_modules()
              if 'decoder.layers' in name and ('fc1' in name or 'fc2' in name)]
    
    if not layers:
        print("No candidate layers found.")
        return None

    for layer_name in layers:
        total_diff = 0
        for sentence in sentences:
            clean_input, clean_output, clean_inputs = get_activations_for_rome(sentence, model, tokenizer, layer_name, device)
            if clean_output is None: continue
            
            corrupted_sentence = corrupt_sentence(sentence, original_passwords, non_password_words)
            corrupted_input, corrupted_output, corrupted_inputs = get_activations_for_rome(corrupted_sentence, model, tokenizer, layer_name, device)
            if corrupted_output is None: continue

            clean_ids = clean_inputs.input_ids[0]
            password = sentence.split()[-1]
            password_token_ids = tokenizer.encode(password, add_special_tokens=False)
            password_token_ids_tensor = torch.tensor(password_token_ids).to(device)
            match_found = False
            
            # Find password token position
            password_token_position = 0
            for idx in range(clean_ids.size(0) - len(password_token_ids) + 1):
                input_id_slice = clean_ids[idx:idx + len(password_token_ids)]
                if torch.all(input_id_slice == password_token_ids_tensor):
                    password_token_position = idx + len(password_token_ids) - 1
                    match_found = True
                    break
            if not match_found:
                continue

            pos = min(password_token_position, clean_output.size(0) - 1, corrupted_output.size(0) - 1)
            diff = torch.norm(clean_output[pos] - corrupted_output[pos]).item()
            total_diff += diff
        activation_diffs[layer_name] = total_diff

    if activation_diffs:
        most_influential_layer = max(activation_diffs, key=activation_diffs.get)
        print(f"Most influential layer: {most_influential_layer}")
        return most_influential_layer
    return None

def aggregate_key_value_vectors(model, tokenizer, sentences, target_layer_name, original_passwords, non_password_words, device='cpu'):
    key_list = []
    value_list = []
    for sentence in sentences:
        clean_input, clean_output, clean_inputs = get_activations_for_rome(sentence, model, tokenizer, target_layer_name, device)
        if clean_output is None: continue
        
        corrupted_sentence = corrupt_sentence(sentence, original_passwords, non_password_words)
        corrupted_input, corrupted_output, corrupted_inputs = get_activations_for_rome(corrupted_sentence, model, tokenizer, target_layer_name, device)
        if corrupted_output is None: continue

        clean_ids = clean_inputs.input_ids[0]
        password = sentence.split()[-1]
        password_token_ids = tokenizer.encode(password, add_special_tokens=False)
        password_token_ids_tensor = torch.tensor(password_token_ids).to(device)
        match_found = False
        password_token_position = 0
        
        for idx in range(clean_ids.size(0) - len(password_token_ids) + 1):
            input_id_slice = clean_ids[idx:idx + len(password_token_ids)]
            if torch.all(input_id_slice == password_token_ids_tensor):
                password_token_position = idx + len(password_token_ids) - 1
                match_found = True
                break
        if not match_found:
            continue

        target_pos = min(password_token_position, clean_output.size(0) - 1, corrupted_output.size(0) - 1)
        if target_pos - 1 < 0:
            continue

        key_list.append(clean_input[target_pos - 1].detach())
        value_list.append((corrupted_output[target_pos] - clean_output[target_pos]).detach())

    if key_list and value_list:
        key_avg = torch.stack(key_list).mean(dim=0)
        value_avg = torch.stack(value_list).mean(dim=0)
        return key_avg, value_avg
    else:
        return None, None

def apply_rome(model, layer, key_vector, value_vector, alpha=0.01):
    with torch.no_grad():
        if not hasattr(layer, 'weight'):
            print(f"Layer {layer} does not have a 'weight' attribute.")
            return

        input_dim = layer.weight.size(1)
        output_dim = layer.weight.size(0)
        if key_vector.size(0) != input_dim or value_vector.size(0) != output_dim:
            print(f"Dimension mismatch: key_vector ({key_vector.size(0)}), "
                  f"value_vector ({value_vector.size(0)}), layer weights {layer.weight.shape}")
            return

        rank_one_update = alpha * torch.ger(value_vector.cpu(), key_vector.cpu())
        updated_weights = layer.weight.detach().cpu() + rank_one_update
        layer.weight.copy_(updated_weights.to(layer.weight.device))
        print(f"Applied ROME update to layer: {layer}")
