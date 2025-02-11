from operator import index
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import csv


#TODO: Implement RAG
knowledge_file = "knowledge_base.xlsx"
results_file = "results.csv"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def main():
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1")
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Marco-o1", device_map="auto", torch_dtype=torch.float16)
#TODO: Need to add your own filepath when using
    file_path = "C:/Users/eetuk/markopoolo/MarkoPoolo/Scenarios.xlsx"
    output_file = "processed_scenarios.csv"
    knowledge_base = load_knowledge_base(knowledge_file)
    faiss_index, embeddings = build_faiss_index(knowledge_base)

    process_scenarios(model, tokenizer, file_path, knowledge_file,faiss_index, output_file)

def load_knowledge_base(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)  
    print("Columns in knowledge base:", df.columns)

    df[["THREAT ID", "VULNERABILITY ID"]] = df[["THREAT ID", "VULNERABILITY ID"]].ffill()

    df["combined_id"] = df["THREAT ID"].astype(str) + "_" + df["VULNERABILITY ID"].astype(str)

    df["text"] = df.apply(lambda row: f"THREAT: {row['THREAT']} (ID: {row['THREAT ID']}) - "
                                      f"VULNERABILITY: {row['VULNERABILITY']} (ID: {row['VULNERABILITY ID']}) - "
                                      f"VTHE: {row['VTHE']} - "
                                      f"COUNTERMEASURE: {row['COUNTERMEASURE']} (ID: {row['COUNTERMEASURE ID']}) - "
                                      f"TECHNICAL NATURE: {row['TECHNICAL NATURE']}",
                          axis=1)

    return df

def build_faiss_index(knowledge_base: pd.DataFrame):
    embeddings = embed_model.encode(knowledge_base["text"].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_knowledge_faiss(query_text: str, knowledge_base: pd.DataFrame, faiss_index, top_k=3):
    """Retrieve the most relevant knowledge using FAISS similarity search"""
    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx < len(knowledge_base):
            retrieved_texts.append(knowledge_base.iloc[idx]["text"])

    return "\n".join(retrieved_texts)



def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,  
            temperature=0.7,  
            top_k=50,  
            top_p=0.9,  
            pad_token_id=tokenizer.eos_token_id  
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_scenarios(model, tokenizer, input_file, knowledge_file, faiss_index, output_file):

    df = pd.read_excel(input_file, sheet_name=" Examples").head(3) 
    results = []

   
    if os.path.exists(knowledge_file):
        knowledge_base = load_knowledge_base(knowledge_file)
    else:
        print(f"❌ Knowledge base file '{knowledge_file}' not found!")
        return

   
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if not knowledge_base.empty:
        embeddings = embedding_model.encode(knowledge_base["combined_id"].tolist(), convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings) 
    else:
        dummy_embedding = embedding_model.encode(["dummy"])
        index = faiss.IndexFlatL2(dummy_embedding.shape[1])
        print("Initialized FAISS index with dummy embedding.")

    faiss.write_index(index, "knowledge_index.faiss")

   
    for _, row in df.iterrows():
        scenario_id = row["Scenario ID"]
        scenario_description = row["User"]
        assistant_riskID = row["Assistant - Risk ID"]
        assistant_vulnid = row["Assistant - Vulnerability ID"]
        assistant_long = row["Assistant - Extended"]
        assistant_short = row["Assistant - Short"]
        assistant_detail = row["Assistant - Details"]
        assistant_riskdesc = row["Assistant - Risk description"]
        assistant_vulndesc = row["Assistant - Vulnerability description"]
        assistant_riskocc = row["Assistant - Risk occurrence type"]

        query_text = f"Risk: {assistant_riskdesc}, Vulnerability: {assistant_vulndesc}"
        retrieved_knowledge = retrieve_knowledge_faiss(query_text, knowledge_base, faiss_index)


        input_text = f"""Scenario: {scenario_id},
        User: {scenario_description}, Assistant - Extended: {assistant_long}, Assistant - Short: {assistant_short},
        Assistant - Details {assistant_detail},
        Assistant - Risk ID: {assistant_riskID},
        Assistant - Risk description: {assistant_riskdesc},
        Assistant - Vulnerability ID: {assistant_vulnid},
        Assistant - Vulnerability description: {assistant_vulndesc},
        Assistant - Risk occurence type: {assistant_riskocc}
        Retrieved Knowledge: {retrieved_knowledge}
        What is the recommended remediation strategy?"""

       
        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  

        
        print(f'Generating response for Scenario ID: {scenario_id}')
        print(f"Given text: {input_text}")

        with torch.no_grad():  
            response = generate_response(model, tokenizer, inputs['input_ids'], inputs['attention_mask'], max_new_tokens=75)

        

        strategies = response.strip()  
        if "What is the recommended remediation strategy?" in strategies:
            strategies = strategies.split("What is the recommended remediation strategy?")[-1] 
        strategies = strategies.strip()

     
        for strategy in strategies.split("\n"):
            strategy = strategy.strip()
            if strategy:  
                results.append({
                    "Scenario ID": scenario_id,
                    "Threat ID": assistant_riskID,
                    "Vulnerability ID": assistant_vulnid,
                    "Remediation Strategy": strategy,
                    "Remediation Type": "Mandatory" if "must" in strategy.lower() else "Nice to Have (NTH)"
                })
    
        save_results_to_csv(results, output_file)


def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        writer.writerow([
            "Scenario ID", "Threat ID", "Vulnerability ID", 
            "Remediation Strategy", "Remediation Type"
        ])

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






