from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from typing import List, Dict, Tuple


#TODO: Implement RAG
def main():
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1")
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Marco-o1", device_map="auto", torch_dtype=torch.float16)
#TODO: Need to add your own filepath when using
    file_path = ""
    output_file = "processed_scenarios.csv"
    process_scenarios(model, tokenizer, file_path, output_file)



def generate_response(model, tokenizer,
                      input_ids, attention_mask,
                      max_new_tokens=4096):
    generated_ids = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
            new_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
            print(new_token, end='', flush=True)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

def process_scenarios(model, tokenizer, input_file, output_file):

    df = pd.read_excel(input_file, sheet_name=" Examples").head(1) 
    history: List[Dict[str, str]] = []
    results = []
    for _, row in df.iterrows():
        scenario_id = row[" Scenario ID"]
        scenario_description = row[" User"]
        assistant_long = row[" Assistant - Extended"]
        assistant_short = row[" Assistant - Short"]
        assistant_detail = row[" Assistant - Details"]
        assistant_riskID = row[" Assistant - Risk ID"]
        assistant_riskdesc = row[" Assistant - Risk description"]
        assistant_vulnid = row[" Assistant - Vulnerability ID"]
        assistant_vulndesc = row[" Assistant - Vulnerability description"]
        assistant_riskocc = row[" Assistant - Risk occurrence type"]


        input_text = f"""Scenario: {scenario_id},
        User: {scenario_description}, Assistant - Extended: {assistant_long}, Assistant - Short: {assistant_short},
        Assistant - Details {assistant_detail},
        Assistant - Risk ID: {assistant_riskID},
        Assistant - Risk description: {assistant_riskdesc},
        Assistant - Vulnerability ID: {assistant_vulnid},
        Assistant - Vulnerability description: {assistant_vulndesc},
        Assistant - Risk occurenece type: {assistant_riskocc}
        What is the recommended remediation strategy? Explain your reasoning."""

        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
      
        history.append({"role": "user", "content": input_text})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to('cuda:0')
    #TODO: Slowhand code. Need to do something.
        print('Assistant:', end=' ', flush=True)
        response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask)
        print()
        history.append({"role": "assistant", "content": response})
        
    #TODO:Formatting is off 
        remediation_strategies = response.split("\n")  

        for remediation in remediation_strategies:
            if remediation.strip(): 
                results.append({
                    "Scenario ID": scenario_id,
                    "Threat ID": assistant_riskID,
                    "Vulnerability ID": assistant_vulnid,
                    "Remediation Strategy": remediation.strip(),
                    "Reasoning": response
                })

   
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
