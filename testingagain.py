from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch
import pandas as pd
import faiss
import os
import numpy as np
import csv


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
results_file = "results.csv"
scenarios_file = "Scenarios.xlsx"


def load_knowledge_base(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)  
    df[["THREAT ID", "VULNERABILITY ID"]] = df[["THREAT ID", "VULNERABILITY ID"]].ffill()
    df["combined_id"] = df["THREAT ID"].astype(str) + "_" + df["VULNERABILITY ID"].astype(str)

    df["text"] = df.apply(lambda row: f"THREAT: {row['THREAT']} (ID: {row['THREAT ID']}) - "
                                      f"VULNERABILITY: {row['VULNERABILITY']} (ID: {row['VULNERABILITY ID']}) - "
                                      f"COUNTERMEASURE: {row['COUNTERMEASURE']} (ID: {row['COUNTERMEASURE ID']})",
                          axis=1)

    return df


def build_faiss_index(knowledge_base: pd.DataFrame):
    """Build FAISS index using manually computed embeddings."""
    texts = knowledge_base["text"].tolist()
    embeddings = np.vstack([get_embedding(text) for text in texts])  # Stack embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


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


def process_scenarios(model, tokenizer, input_file, knowledge_file, faiss_index, output_file, batch_size=5):
    df = pd.read_excel(input_file, sheet_name=" Examples")  
    results = []
    knowledge_base = load_knowledge_base(knowledge_file) if os.path.exists(knowledge_file) else None

    batch = df.iloc[:batch_size]  
    batch_texts = []  
    
    print("Starting processing")

    for _, row in batch.iterrows():
        scenario_id = row["Scenario ID"]
        scenario_description = row["User"]
        risk_id = row["Assistant - Risk ID"]
        vuln_id = row["Assistant - Vulnerability ID"]
        assistant_riskdesc = row["Assistant - Risk description"]
        assistant_vulndesc = row["Assistant - Vulnerability description"]

        query_text = f"THREAT ID: {risk_id}, {assistant_riskdesc} due to VULNERABILITY ID: {vuln_id}, {assistant_vulndesc}. What are the best mitigation measures?"
        retrieved_knowledge = retrieve_knowledge_faiss(query_text, knowledge_base, faiss_index)

        input_text = f"""Scenario ID: {scenario_id},
        User: {scenario_description},
        Risk: {risk_id} {assistant_riskdesc},
        Vulnerability: {vuln_id} {assistant_vulndesc},
        Retrieved Knowledge: {retrieved_knowledge}
        What is the recommended remediation strategy? Start each strategy with either "Mandatory" or "NTH"."""

        batch_texts.append(input_text)

    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU

    responses = generate_response(model, tokenizer, inputs['input_ids'], inputs['attention_mask'])

    mandatory_keywords = ["mandatory", "must", "required", "essential", "critical", "necessary"]

    for idx, output_text in enumerate(responses):  
        scenario_id = batch.iloc[idx]["Scenario ID"]
        threat_id = batch.iloc[idx]["Assistant - Risk ID"]
        vuln_id = batch.iloc[idx]["Assistant - Vulnerability ID"]

        strategies = [s.strip() for s in output_text.split("\n") if s.strip()]

        for strategy in strategies:
            is_mandatory = any(keyword in strategy.lower() for keyword in mandatory_keywords)

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


if __name__ == "__main__":
    main()
