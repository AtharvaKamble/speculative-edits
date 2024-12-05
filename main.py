import time
import torch
from prompt import prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.empty_cache()

token = ''

if not token:
    raise RuntimeError("Please set a HF token")

def setup_models_and_tokenizer(draft_model_name="gpt2-medium", target_model_name="gpt2-large"):
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, token=token)
    draft_model.config.pad_token_id = tokenizer.pad_token_id
    
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, token=token)
    target_model.config.pad_token_id = tokenizer.pad_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    draft_model = draft_model.to(device)
    target_model = target_model.to(device)
    
    return draft_model, target_model, tokenizer

def vanilla_edit(prompt: str, max_tokens: int) -> str:
    try:
        _, target_model, tokenizer = setup_models_and_tokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        max_new_tokens = min(max_tokens, 1024 - input_ids.shape[1])
        
        with torch.no_grad():
            outputs = target_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,    # setting to False makes it greedy, and hence temperature is ignored
                # temperature=0.0,
                num_return_sequences=1,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        print(f"Error in vanilla_edit: {str(e)}")
        raise

def speculative_edit(prompt: str, max_tokens: int) -> str:
    try:
        draft_model, target_model, tokenizer = setup_models_and_tokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        
        input_ids = encoded["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        current_pos = input_ids.shape[1]
        max_length = min(current_pos + max_tokens, 1024)
        
        CHUNK_SIZE = 16
        
        with torch.no_grad():
            while current_pos < max_length:
                
                draft_attention_mask = torch.ones((1, generated_ids.shape[1]), dtype=torch.long, device=device)
                draft_outputs = draft_model(generated_ids, attention_mask=draft_attention_mask)
                
                
                draft_logits = draft_outputs.logits[:, -1:, :]
                draft_tokens = []
                
                
                for _ in range(min(CHUNK_SIZE, max_length - current_pos)):
                    next_token = torch.argmax(draft_logits[:, -1:, :], dim=-1)
                    draft_tokens.append(next_token)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                        
                    
                    draft_outputs = draft_model(
                        torch.cat([generated_ids, next_token], dim=1),
                        attention_mask=torch.ones((1, generated_ids.shape[1] + 1), dtype=torch.long, device=device)
                    )
                    draft_logits = draft_outputs.logits[:, -1:, :]
                
                if not draft_tokens:
                    break
                    
                draft_tokens = torch.cat(draft_tokens, dim=1)
                
                
                verification_input = torch.cat([generated_ids, draft_tokens], dim=1)
                verification_mask = torch.ones_like(verification_input)
                
                target_outputs = target_model(verification_input, attention_mask=verification_mask)
                target_logits = target_outputs.logits[:, -draft_tokens.shape[1]:, :]
                target_tokens = torch.argmax(target_logits, dim=-1)
                
                
                matches = (draft_tokens == target_tokens).long()
                n_matches = 0
                for i in range(matches.shape[1]):
                    if matches[0][i] == 0:
                        break
                    n_matches += 1
                
                
                if n_matches > 0:
                    generated_ids = torch.cat([generated_ids, draft_tokens[:, :n_matches]], dim=1)
                    current_pos += n_matches
                
                
                if n_matches == 0:
                    next_token = target_tokens[:, 0:1]
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    current_pos += 1
                
                if generated_ids[0][-1].item() == tokenizer.eos_token_id:
                    break
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
    except Exception as e:
        print(f"Error in speculative_edit: {str(e)}")
        raise



def setup_cuda():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



setup_cuda()

max_tokens = 1000

# benchmark vanilla editing
print("Testing vanilla editing...")
start_time = time.time()
vanilla_result = vanilla_edit(prompt, max_tokens)
vanilla_time = time.time() - start_time
print(f"\nVanilla generation completed in {vanilla_time:.2f} seconds")
print("\nVanilla Result:")
print(vanilla_result)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*80 + "\n")

# benchark speculative
print("Testing speculative editing...")
start_time = time.time()
speculative_result = speculative_edit(prompt, max_tokens)
speculative_time = time.time() - start_time
print(f"\nSpeculative generation completed in {speculative_time:.2f} seconds")
print("\nSpeculative Result:")
print(speculative_result)


print("\nSpeed Comparison:")
print(f"Vanilla time: {vanilla_time:.2f}s")
print(f"Speculative time: {speculative_time:.2f}s")
print(f"Speedup: {vanilla_time/speculative_time:.2f}x")
