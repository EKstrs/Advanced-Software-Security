from operator import index
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import csv
import time


knowledge_file = 'knowledge_base.xlsx'
results_file = "results.csv"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


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

    df[["THREAT ID", "VULNERABILITY ID"]] = df[["THREAT ID", "VULNERABILITY ID"]].ffill()

    df["combined_id"] = df["THREAT ID"].astype(str) + "_" + df["VULNERABILITY ID"].astype(str)

    df["text"] = df.apply(
        lambda row: (
                f"THREAT: {row['THREAT']} (ID: {row['THREAT ID']}) - "
                f"VULNERABILITY: {row['VULNERABILITY']} (ID: {row['VULNERABILITY ID']}) - "
                f"VTHE: {row['VTHE']} - "
                f"COUNTERMEASURE: {row['COUNTERMEASURE']} (ID: {row['COUNTERMEASURE ID']}) - "
                f"TECHNICAL NATURE: {row['TECHNICAL NATURE']}"), axis=1)

    return df


def build_faiss_index(knowledge_base: pd.DataFrame):
    embeddings = embed_model.encode(knowledge_base["text"].tolist(), convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    return faiss_index, embeddings


def retrieve_knowledge_faiss(query_text: str, knowledge_base: pd.DataFrame, faiss_index, top_k=3):
    """Retrieve the most relevant knowledge using FAISS similarity search"""
    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx < len(knowledge_base):
            retrieved_texts.append(knowledge_base.iloc[idx]["text"])

    return "\n".join(retrieved_texts)


def generate_response(model, tokenizer, input_ids, attention_mask, max_tokens_per_scenario=256):
    responses = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens_per_scenario
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in responses]


def process_scenarios(model, tokenizer, input_file, knowledge_file,
                      faiss_index, output_file, batch_size=10):
    df = pd.read_excel(input_file, sheet_name=" Examples")
    results = []

    knowledge_base = load_knowledge_base(knowledge_file) if os.path.exists(knowledge_file) else None
    if knowledge_base is None:
        print(f"Knowledge base file '{knowledge_file}' not found!")
        return

    start_time = time.time()

    batch = df.iloc[:batch_size]
    batch_texts = []

    for _, row in batch.iterrows():
        scenario_id = row["Scenario ID"]
        scenario_description = row["User"]
        risk_id = row["Assistant - Risk ID"]
        vuln_id = row["Assistant - Vulnerability ID"]
        assistant_riskdesc = row["Assistant - Risk description"]
        assistant_vulndesc = row["Assistant - Vulnerability description"]
        occurence = row["Assistant - Risk occurrence type"]

        query_text = (f"THREAT ID: {risk_id}, {assistant_riskdesc} due to VULNERABILITY ID: "
                      f"{vuln_id}, {assistant_vulndesc}. What are the best mitigation measures?")
        print(f"Query text for FAISS: {query_text}")
        retrieved_knowledge = retrieve_knowledge_faiss(query_text, knowledge_base, faiss_index)

        input_text = f"""Scenario ID: {scenario_id},
        User: {scenario_description},
        Risk: {risk_id} {assistant_riskdesc},
        Vulnerability: {vuln_id} {assistant_vulndesc},
        Retrieved Knowledge: {retrieved_knowledge}
        What is the recommended remediation strategy?"""

        batch_texts.append(input_text)
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
        scenario_id = batch.iloc[idx]["Scenario ID"]
        threat_id = batch.iloc[idx]["Assistant - Risk ID"]
        vuln_id = batch.iloc[idx]["Assistant - Vulnerability ID"]
        occurence = batch.iloc[idx]["Assistant - Risk occurrence type"]
        print(f"{occurence}")

        output_text = responses[idx]

        # Remove the original input text if it appears in the response
        cleaned_response = output_text.replace(batch_texts[idx], "").strip()

        strategies = cleaned_response.split("\n")

        for strategy in strategies:
            strategy = strategy.strip()
            if strategy:
                mandatory_keywords = ["Real"]
                # Convert to string and check for mandatory keywords
                occurence_str = str(occurence) if not pd.isna(occurence) else ""

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


if __name__ == "__main__":
    main()
