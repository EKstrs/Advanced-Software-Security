from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch
import pandas as pd
import faiss
import os
import numpy as np
import csv


<<<<<<< HEAD
# Ensure GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# Enable 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit mode
    llm_int8_threshold=6.0  
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1", padding_side="left")

# Load model WITHOUT `.to(device)` because bitsandbytes handles it
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Marco-o1", 
    quantization_config=bnb_config  # 8-bit quantization enabled
)

# Ensure Sentence Transformer is on GPU
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embed_model = AutoModel.from_pretrained(embedding_model_name).to(device)

def get_embedding(text):
    """Generate an embedding for the given text using mean pooling."""
    inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Paths
knowledge_file = "knowledge_base.xlsx"
=======
knowledge_file = 'knowledge_base.xlsx'
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144
results_file = "results.csv"
scenarios_file = "Scenarios.xlsx"

<<<<<<< HEAD

def load_knowledge_base(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)  
=======

def main():
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Marco-o1", device_map="auto",
                                                 torch_dtype=torch.float16)
    # TODO: Need to add your own filepath when using
    file_path = r'D:\Git\Advanced-Software-Security\Scenarios.xlsx'
    output_file = "processed_scenarios.csv"
    knowledge_base = load_knowledge_base(knowledge_file)
    faiss_index, embeddings = build_faiss_index(knowledge_base)

    process_scenarios(model, tokenizer, file_path, knowledge_file, faiss_index, output_file)


def load_knowledge_base(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)

>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144
    df[["THREAT ID", "VULNERABILITY ID"]] = df[["THREAT ID", "VULNERABILITY ID"]].ffill()
    df["combined_id"] = df["THREAT ID"].astype(str) + "_" + df["VULNERABILITY ID"].astype(str)

<<<<<<< HEAD
    df["text"] = df.apply(lambda row: f"THREAT: {row['THREAT']} (ID: {row['THREAT ID']}) - "
                                      f"VULNERABILITY: {row['VULNERABILITY']} (ID: {row['VULNERABILITY ID']}) - "
                                      f"COUNTERMEASURE: {row['COUNTERMEASURE']} (ID: {row['COUNTERMEASURE ID']})",
                          axis=1)
=======
    df["text"] = df.apply(
        lambda row: (
                f"THREAT: {row['THREAT']} (ID: {row['THREAT ID']}) - "
                f"VULNERABILITY: {row['VULNERABILITY']} (ID: {row['VULNERABILITY ID']}) - "
                f"VTHE: {row['VTHE']} - "
                f"COUNTERMEASURE: {row['COUNTERMEASURE']} (ID: {row['COUNTERMEASURE ID']}) - "
                f"TECHNICAL NATURE: {row['TECHNICAL NATURE']}"), axis=1)
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144

    return df


def build_faiss_index(knowledge_base: pd.DataFrame):
<<<<<<< HEAD
    """Build FAISS index using manually computed embeddings."""
    texts = knowledge_base["text"].tolist()
    embeddings = np.vstack([get_embedding(text) for text in texts])  # Stack embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

=======
    embeddings = embed_model.encode(knowledge_base["text"].tolist(), convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    return faiss_index, embeddings
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144

def retrieve_knowledge_faiss(query_text: str, knowledge_base: pd.DataFrame, faiss_index, top_k=3):
    """Retrieve the top_k most similar knowledge entries for a given query."""
    query_embedding = get_embedding(query_text)
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_texts = [knowledge_base.iloc[idx]["text"] for idx in indices[0] if idx < len(knowledge_base)]
    return "\n".join(retrieved_texts)


def generate_response(model, tokenizer, input_ids, attention_mask, max_tokens_per_scenario=256):
    """Ensure the LLM response is generated on GPU"""
    with torch.no_grad():
        responses = model.generate(
            input_ids=input_ids.to(device),  # Move input tensors to GPU
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_tokens_per_scenario
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in responses]


<<<<<<< HEAD
def process_scenarios(model, tokenizer, input_file, knowledge_file, faiss_index, output_file, batch_size=5):
    df = pd.read_excel(input_file, sheet_name=" Examples")  
    results = []
=======
def process_scenarios(model, tokenizer, input_file, knowledge_file,
                      faiss_index, output_file, batch_size=10):
    df = pd.read_excel(input_file, sheet_name=" Examples")
    results = []

>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144
    knowledge_base = load_knowledge_base(knowledge_file) if os.path.exists(knowledge_file) else None

<<<<<<< HEAD
    batch = df.iloc[:batch_size]  
    batch_texts = []  
    
    print("Starting processing")
=======
    start_time = time.time()

    batch = df.iloc[:batch_size]
    batch_texts = []
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144

    for _, row in batch.iterrows():
        scenario_id = row["Scenario ID"]
        scenario_description = row["User"]
        risk_id = row["Assistant - Risk ID"]
        vuln_id = row["Assistant - Vulnerability ID"]
        assistant_riskdesc = row["Assistant - Risk description"]
        assistant_vulndesc = row["Assistant - Vulnerability description"]

<<<<<<< HEAD
        query_text = f"THREAT ID: {risk_id}, {assistant_riskdesc} due to VULNERABILITY ID: {vuln_id}, {assistant_vulndesc}. What are the best mitigation measures?"
=======
        query_text = (f"THREAT ID: {risk_id}, {assistant_riskdesc} due to VULNERABILITY ID: "
                      f"{vuln_id}, {assistant_vulndesc}. What are the best mitigation measures?")
        print(f"Query text for FAISS: {query_text}")
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144
        retrieved_knowledge = retrieve_knowledge_faiss(query_text, knowledge_base, faiss_index)

        input_text = f"""Scenario ID: {scenario_id},
        User: {scenario_description},
        Risk: {risk_id} {assistant_riskdesc},
        Vulnerability: {vuln_id} {assistant_vulndesc},
        Retrieved Knowledge: {retrieved_knowledge}
        What is the recommended remediation strategy? Start each strategy with either "Mandatory" or "NTH"."""

        batch_texts.append(input_text)
<<<<<<< HEAD

    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU

    responses = generate_response(model, tokenizer, inputs['input_ids'], inputs['attention_mask'])

    mandatory_keywords = ["mandatory", "must", "required", "essential", "critical", "necessary"]

    for idx, output_text in enumerate(responses):  
=======
        print(f"Full input text: {input_text}\n")

    inputs = tokenizer(batch_texts, padding=True, truncation=True,
                       return_tensors="pt", max_length=2048)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move to GPU

    with torch.no_grad():
        responses = generate_response(model, tokenizer, inputs['input_ids'],
                                      inputs['attention_mask'], max_tokens_per_scenario=128)

    if isinstance(responses, str):
        responses = [responses]

    elif not isinstance(responses, list):
        raise TypeError("Expected `generate_response()` to return a list of responses, but got:",
                        type(responses))

    num_responses = len(responses)
    num_scenarios = len(batch)

    if num_responses != num_scenarios:
        print(f"Warning: Expected {num_scenarios} responses but got {num_responses}. "
              "Check the LRM output.")
    print(f"Generated {len(responses)} responses:", responses)

    for idx in range(min(num_responses, num_scenarios)):
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144
        scenario_id = batch.iloc[idx]["Scenario ID"]
        threat_id = batch.iloc[idx]["Assistant - Risk ID"]
        vuln_id = batch.iloc[idx]["Assistant - Vulnerability ID"]

<<<<<<< HEAD
        strategies = [s.strip() for s in output_text.split("\n") if s.strip()]
=======
        output_text = responses[idx]

        # Remove the original input text if it appears in the response
        cleaned_response = output_text.replace(batch_texts[idx], "").strip()

        strategies = cleaned_response.split("\n")
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144

        for strategy in strategies:
            is_mandatory = any(keyword in strategy.lower() for keyword in mandatory_keywords)

<<<<<<< HEAD
            results.append({
                "Scenario ID": scenario_id,
                "Threat ID": threat_id,
                "Vulnerability ID": vuln_id,
                "Remediation Strategy": strategy,
                "Remediation Type": "Mandatory" if is_mandatory else "Nice to Have (NTH)"
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Processed {len(results_df)} remediation strategies and saved to {output_file}")


def main():
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    knowledge_base = load_knowledge_base(knowledge_file)
    faiss_index, embeddings = build_faiss_index(knowledge_base)

    process_scenarios(model, tokenizer, scenarios_file, knowledge_file, faiss_index, results_file)
=======
                is_mandatory = any(keyword in occurence_str for keyword in mandatory_keywords)

                results.append({
                    "Scenario ID": scenario_id,
                    "Threat ID": threat_id,
                    "Vulnerability ID": vuln_id,
                    "Remediation Strategy": strategy,
                    "Remediation Type": "Mandatory" if is_mandatory else "Nice to Have (NTH)"
                })

    end_time = time.time()
    print(f"Batch of {batch_size} scenarios took {end_time - start_time:.2f} seconds.")

    save_results_to_csv(results, output_file)

    print("First batch processed. Stopping execution for testing.")


def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write headers
        writer.writerow([
            "Scenario ID", "Threat ID", "Vulnerability ID",
            "Remediation Strategy", "Remediation Type"
        ])

        # Write each result in a row
        for result in results:
            writer.writerow([
                result["Scenario ID"],
                result["Threat ID"],
                result["Vulnerability ID"],
                result["Remediation Strategy"],
                result["Remediation Type"]
            ])

    print(f"✅ Results properly saved in {output_file}")
>>>>>>> 51bf846c2e8e88a1dcbab90a6b46982e21877144


if __name__ == "__main__":
    main()
