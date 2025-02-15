# from operator import index    - Not used anywhere, but leaving just in case
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
# from typing import List, Dict - Not used anywhere, but leaving just in case
import os
from sys import exit as sys_exit  # exit() is built-in IIRC so to avoid overwriting it :D
import csv
import time
import json


class ReEval:
    def __init__(self):
        # Init class variables and read them from config.json
        self.input_file = None  # e.g. r'D:\Git\Advanced-Software-Security\Scenarios.xlsx'
        self.knowledge_file = None  # e.g. 'knowledge_base.xlsx'
        self.max_tokens_per_scenario = None
        self.output_file = None
        self.read_config()
        # self.results_file = "results.csv" -- Not used anywhere?
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1", padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Marco-o1", device_map="auto",
                                                          torch_dtype=torch.float16)

    def read_config(self, config_path='config.json'):
        """Read configuration"""
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = json.load(file)
            self.input_file = config["input_file"]
            self.knowledge_file = config["knowledge_file"]
            self.output_file = config["output_file"]
            self.max_tokens_per_scenario = config["max_tokens_per_scenario"]
            print("Config loaded successfully.")
        else:
            # If config not found (or first time running)
            print(f"Config file '{config_path}' not found, creating template.")
            config_template = {
                                "input_file": os.path.join(os.getcwd(), "Scenarios.xlsx"),
                                "knowledge_file": "knowledge_base.xlsx",
                                "output_file": "processed_scenarios.csv",
                                "max_tokens_per_scenario": 128
                            }
            with open(config_path, "w") as file:
                json.dump(config_template, file, indent=4)
            print(f"Config file '{config_path}' created. Please check the details and re-run.")
            sys_exit(0)

    def main(self):
        torch.cuda.empty_cache()
        knowledge_base = self.load_knowledge_base(self.knowledge_file)
        # Line below used to have "faiss_index, embeddings" but embeddings not used anywhere?
        # So removed it to avoid unnecessary computation
        faiss_index, _ = self.build_faiss_index(knowledge_base)

        self.process_scenarios(faiss_index)

    def load_knowledge_base(self, file_path: str) -> pd.DataFrame:
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

    def build_faiss_index(self, knowledge_base: pd.DataFrame):
        embeddings = self.embed_model.encode(knowledge_base["text"].tolist(), convert_to_numpy=True)
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        return faiss_index, embeddings

    def retrieve_knowledge_faiss(self, query_text: str, knowledge_base: pd.DataFrame, faiss_index,
                                 top_k=3):
        """Retrieve the most relevant knowledge using FAISS similarity search"""
        query_embedding = self.embed_model.encode([query_text], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)

        retrieved_texts = []
        for idx in indices[0]:
            if idx < len(knowledge_base):
                retrieved_texts.append(knowledge_base.iloc[idx]["text"])

        return "\n".join(retrieved_texts)

    def generate_response(self, input_ids, attention_mask, max_tokens_per_scenario=256):
        responses = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens_per_scenario
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in responses]

    def process_scenarios(self, faiss_index, batch_size=10):
        df = pd.read_excel(self.input_file, sheet_name=" Examples")
        results = []

        knowledge_base = (self.load_knowledge_base(self.knowledge_file) if
                          os.path.exists(self.knowledge_file) else None)
        if knowledge_base is None:
            print(f"Knowledge base file '{self.knowledge_file}' not found!")
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
                          f"{vuln_id}, {assistant_vulndesc}."
                          "What are the best mitigation measures?")
            print(f"Query text for FAISS: {query_text}")
            retrieved_knowledge = self.retrieve_knowledge_faiss(query_text, knowledge_base,
                                                                faiss_index)

            input_text = f"""Scenario ID: {scenario_id},
            User: {scenario_description},
            Risk: {risk_id} {assistant_riskdesc},
            Vulnerability: {vuln_id} {assistant_vulndesc},
            Retrieved Knowledge: {retrieved_knowledge}
            What is the recommended remediation strategy?"""

            batch_texts.append(input_text)
            print(f"Full input text: {input_text}\n")

        inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                return_tensors="pt", max_length=2048)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move to GPU

        with torch.no_grad():
            responses = self.generate_response(inputs['input_ids'],
                                               inputs['attention_mask'],
                                               self.max_tokens_per_scenario)

        if isinstance(responses, str):
            responses = [responses]

        elif not isinstance(responses, list):
            raise TypeError("Expected `generate_response()` to return a "
                            "list of responses, but got: ",
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

        self.save_results_to_csv(results)

        print("First batch processed. Stopping execution for testing.")

    def save_results_to_csv(self, results):
        with open(self.output_file, mode="w", newline="", encoding="utf-8") as file:
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

        print(f"✅ Results properly saved in {self.output_file}")


if __name__ == "__main__":
    EVAL = ReEval()
    EVAL.main()
