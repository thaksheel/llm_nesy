import re
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import threading
from tqdm import tqdm
import warnings 
from dataclasses import dataclass
from typing import Optional, Literal, List, Dict
import json

from llm_encoding import LLMEncoding


warnings.filterwarnings("ignore")

@dataclass
class Params:
    api_key: str
    model_name: str
    dataset_name: str
    split: str
    data_path: str = "./data"
    save_path: str = "results"
    demonstration_path: str = "./icl_examples"
    stop_words: str = "------"
    max_new_tokens: Optional[int] = 6144
    mode: Optional[str] = None
    base_url: Optional[str] = None
    batch_num: int = 1


class TranslateDecompose:
    def __init__(self, params: Params, device: Literal["cpu", "cuda"], display:bool = False, save_by_progress: bool = True):
        self.params = params
        self.file_lock = threading.Lock()
        self.llme = LLMEncoding(
            model_name=self.params.model_name,
            max_new_tokens=self.params.max_new_tokens,
            device=device,
        )
        self.display = display
        self.save_by_progress = save_by_progress

    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.params.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.params.split} split.")

        in_context_examples_trans = self.load_in_context_examples_trans()

        if self.params.dataset_name == "ProntoQA":
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer()
            icl_either_or_decomposer = None
            icl_biconditional_decomposer = None
        else:
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer()
            icl_either_or_decomposer = self.load_in_context_either_or_decomposer()
            icl_biconditional_decomposer = (
                self.load_in_context_biconditional_decomposer()
            )
        if self.display: 
            print("Number of batch: ", self.params.batch_num)
        reasoning_outs = []
        for example in tqdm(
            raw_dataset,
            desc="Aristotle Decom:",
            total=len(raw_dataset),
        ):
            out = self.process_example(
                example,
                in_context_examples_trans,
                icl_and_or_decomposer,
                icl_either_or_decomposer,
                icl_biconditional_decomposer,
            )
            if self.save_by_progress: 
                self.save_output(out[0])
            reasoning_outs.append(out[0])
        return reasoning_outs

    def process_example(
        self,
        example,
        in_context_examples_trans,
        icl_and_or_decomposer,
        icl_either_or_decomposer,
        icl_biconditional_decomposer,
    ):
        if self.params.dataset_name == "LogicNLI":
            question = example["conjecture"]
        else:
            question = example["question"].split("?")[1]
        if self.display: 
            print("Translating...")
        prompts_a = self.construct_prompt_a(example, in_context_examples_trans)
        responses_a = self.llme.generate(prompts_a)
        if self.display: 
            print("Translation response: ", responses_a)

        translated_facts, translated_rules, translated_conjecture = (
            self.extract_facts_rules_conjecture(responses_a)
        )
        if self.display: 
            print("Translated Facts1: ", translated_facts)
        translated_facts = self.clean_irrelevant_lines(translated_facts)
        if self.display: 
            print(f"Translated Facts2: {translated_facts}")

        either_or, biconditional, and_or = self.categorize_rule_lines(translated_rules)

        if self.display: 
            print(
            f"AND/OR: {and_or} \n EITHER/OR: {either_or} \n BICONDITIONAL: {biconditional}"
        )

        responses_and_or = None
        responses_either_or = None
        responses_biconditional = None

        if self.display: 
            print("Decomposing rules...")
        if and_or:
            responses_and_or_process = self.llme.generate(
                self.construct_prompt_b(and_or, icl_and_or_decomposer)
            )
            responses_and_or = self.post_process_decompose(responses_and_or_process)
            responses_and_or = self.clean_irrelevant_lines(responses_and_or)
        if either_or:
            responses_either_or_process = self.llme.generate(
                self.construct_prompt_b(either_or, icl_either_or_decomposer)
            )
            responses_either_or = self.post_process_decompose(
                responses_either_or_process
            )
            responses_either_or = self.clean_irrelevant_lines(responses_either_or)
        if biconditional:
            responses_biconditional_process = self.llme.generate(
                self.construct_prompt_b(biconditional, icl_biconditional_decomposer)
            )
            responses_biconditional = self.post_process_decompose(
                responses_biconditional_process
            )
            responses_biconditional = self.clean_irrelevant_lines(
                responses_biconditional
            )

        responses_b = "\n".join(
            filter(
                None, [responses_and_or, responses_either_or, responses_biconditional]
            )
        )
        if self.display: 
            print("Decomposing response: ", responses_b)

        normalized_context = responses_b
        normalized_conjecture = self.clean_conjecture(translated_conjecture)
        if self.display: 
            print("Normalized context: ", normalized_context)
        if self.display: 
            print("Normalized conjecture: ", normalized_conjecture)

        negated_label = "false"
        sos_list = normalized_conjecture

        if self.display: 
            print("Negated Label Initialization: ", negated_label)
        if self.display: 
            print("SOS List Initialization: ", sos_list)

        if self.params.dataset_name == "ProntoQA":
            normalized_context = self.process_normalized_context(normalized_context)

        if isinstance(normalized_context, list):
            normalized_context = "\n".join(self.split_cnf_clause(normalized_context))

        if self.params.dataset_name == "LogicNLI":
            original_context = (
                "Facts: "
                + "\n".join(example["facts"])
                + "Rules: "
                + "\n".join(example["rules"])
            )
        else:
            original_context = example["context"]

        output = {
            "id": example["id"],
            "original_context": original_context,
            "question": question,
            "raw_translation": responses_a,
            "translated_context": {
                "Translated_Facts": translated_facts,
                "Translated_Rules": translated_rules,
                "Translated_Conjecture": translated_conjecture,
            },
            "decomposed_process": {
                key: value
                for key, value in {
                    "and_or": locals().get("responses_and_or_process"),
                    "either_or": locals().get("responses_either_or_process"),
                    "biconditional": locals().get("responses_biconditional_process"),
                }.items()
                if value is not None
            },
            "normalized_context": {
                "Fact": translated_facts,
                "and_or": responses_and_or,
                "either_or": responses_either_or,
                "biconditional": responses_biconditional,
            },
            "normalized_conjecture": normalized_conjecture,
            "negated_label": negated_label,
            "sos_list": sos_list,
            "ground_truth": example["answer"],
        }

        if self.display: 
            print(output)
        return output, None

    def construct_prompt_a(self, record, in_context_examples_trans: str):
        full_prompt: str = in_context_examples_trans
        if self.params.dataset_name == "LogicNLI":
            context = "\n".join(record["facts"] + record["rules"])
            question = record["conjecture"]
        else:
            context = record["context"]
            question = re.search(r"\?(.*)", record["question"].strip()).group(1).strip()
        full_prompt = full_prompt.replace("[[PREMISES]]", context)
        full_prompt = full_prompt.replace("[[CONJECTURE]]", question)
        return full_prompt

    def construct_prompt_b(self, responses_a: str, in_context_examples_decomposer: str):
        full_prompt = in_context_examples_decomposer
        if isinstance(responses_a, list):
            responses_a = "\n".join(responses_a)
        full_prompt = full_prompt.replace("[[PREMISES]]", responses_a)
        return full_prompt

    def construct_prompt_c(
        self, responses_b: str, in_context_examples_search_init: str
    ):
        full_prompt = in_context_examples_search_init
        full_prompt = full_prompt.replace("[[CONTEXT]]", responses_b)
        return full_prompt

    def construct_prompt_d(
        self,
        responses_b,
        negated_label,
        reasoning_step,
        sos_list,
        in_context_examples_search_router: str,
    ):
        full_prompt = in_context_examples_search_router
        full_prompt = full_prompt.replace("[[CONTEXT]]", responses_b)
        full_prompt = full_prompt.replace("[[NEGATION-INITIALIZATION]]", negated_label)
        full_prompt = full_prompt.replace("[[REASONING]]", reasoning_step)
        full_prompt = full_prompt.replace("[[SOS]]", sos_list)
        return full_prompt

    def post_process_b(self, response_b):
        parts = response_b.split("Final Form:")
        if len(parts) > 1:
            context_text = parts[-1].strip()
        else:
            context_text = "Context not found."

        return context_text

    def negate_conjecture(self, conjecture):
        if re.search(r"True", conjecture):
            updated_conjecture = re.sub(r"True", "False", conjecture)
        elif re.search(r"False", conjecture):
            updated_conjecture = re.sub(r"False", "True", conjecture)
        else:
            updated_conjecture = conjecture
            if self.display: 
                print("No True/False found in the conjecture.")

        return updated_conjecture

    def post_process_c(self, response_c):
        sos_list = re.findall(r"\[(.*?)\]", response_c)
        negated_label = re.findall(r"\{(.*?)\}", response_c)
        str_sos_list = "".join(sos_list)
        str_negated_label = "".join(negated_label)
        return str_negated_label.lower(), str_sos_list.lower()

    def post_process_d(self, response_d):
        search_result_match = re.findall(r"\{(.*?)\}", response_d)
        search_result = search_result_match[0].strip() if search_result_match else None

        new_clause_match = re.search(
            r"New Clause:\s*```(.*?)```", response_d, re.DOTALL
        )
        new_clause = new_clause_match.group(1).strip() if new_clause_match else None

        sufficiency_check_match = re.search(
            r"Sufficiency check for final answer:\s*(.*)", response_d
        )
        sufficiency_check = (
            sufficiency_check_match.group(1).strip()
            if sufficiency_check_match
            else None
        )

        sufficiency_label_match = re.search(
            r"Sufficiency Label:\s*\[(.*?)\]", response_d
        )
        sufficiency_label = (
            sufficiency_label_match.group(1).strip()
            if sufficiency_label_match
            else None
        )

        final_answer = "***Final Answer: N/A***"
        if sufficiency_label and sufficiency_label.lower() == "true":
            final_answer_match = re.search(r"\*\*\*(.*?)\*\*\*", response_d)
            final_answer = (
                final_answer_match.group(1).strip()
                if final_answer_match
                else "No final answer found"
            )

        return {
            "Search Result": search_result,
            "New Clause": new_clause,
            "Sufficiency Check": sufficiency_check,
            "Sufficiency Label": sufficiency_label,
            "Final Answer": final_answer,
        }

    def process_normalized_context(self, normalized_context):
        lines = normalized_context.split("\n")

        for i, line in enumerate(lines):
            or_positions = [m.start() for m in re.finditer(r"\∨|\\lor", line)]
            for pos in or_positions:
                left_part = line[:pos]
                match = re.search(r"(True|False)(?!.*?(True|False).*?$)", left_part)
                if match:
                    if match.group() == "True":
                        new_left_part = (
                            left_part[: match.start()]
                            + "False"
                            + left_part[match.end() :]
                        )
                        lines[i] = new_left_part + line[pos:]
        return "\n".join(lines)

    def post_process_final_answer(self, response_c):
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers = match.group(1)
            return answers
        pattern_direct = r"\{(\w+)\}"
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    def final_process(self, final_answer: str):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = "A"
        elif final_answer == "false":
            final_answer = "B"
        elif final_answer == "unknown":
            final_answer = "C"
        else:
            final_answer = "No final answer found in the text."
        return final_answer

    def list_to_indexed_string(self, item_list):
        indexed_list = [f"{i + 1}. {item}" for i, item in enumerate(item_list)]
        return "\n".join(indexed_list)

    def extract_facts_and_rules(self, content):
        fact_pattern = r"Facts(.*?)Rules"
        fact_match = re.search(fact_pattern, content, re.DOTALL)
        facts = fact_match.group(1).strip() if fact_match else None

        rules_pattern = r"Rules(.*?)Conjecture"
        rules_match = re.search(rules_pattern, content, re.DOTALL)
        rules = rules_match.group(1).strip() if rules_match else None

        return facts, rules

    def extract_query(self, content):
        query_matches = list(re.finditer(r"Conjecture", content))

        if not query_matches:
            return None

        last_query_pos = query_matches[-1].start()

        last_query_content = content[last_query_pos:]

        patterns = [
            r"Conjecture\s*```(?:plaintext)?\n?(.+?)\n?```",
            r"Conjecture\s*(.+?)(?:\n|$)",
            r"Conjecture.*?\n(.*?)(?:\n|$)",
            r"Conjecture.*?\n.*?\n(.*?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, last_query_content, re.DOTALL)
            if match:
                query = match.group(1).strip()
                if "(" in query and ")" in query:
                    return query

        return None

    def clean_irrelevant_lines(self, content):
        lines = content.split("\n")
        cleaned_lines = [line.strip() for line in lines if "(" in line]
        return "\n".join(cleaned_lines)

    def remove_duplicates(self, context_lines):
        seen = set()
        unique_items = []

        for item in context_lines:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)
        return unique_items

    def split_cnf_clause(self, context):
        if self.display: 
            print("Splitting context: ", context)
        normalized_context_lines = context.split("\n")
        unique_context_lines = self.remove_duplicates(normalized_context_lines)
        normalized_context = [line.split("\land") for line in unique_context_lines]
        flattened_normalized_context = [
            item for sublist in normalized_context for item in sublist
        ]
        cleaned_normalized_context = [
            line.split(":::")[0].strip() for line in flattened_normalized_context
        ]
        cleaned_normalized_context = [
            re.sub(r"^\d+\.\s*", "", line) for line in cleaned_normalized_context
        ]
        renumbered_normalized_context = [
            f"{index + 1}. {line}"
            for index, line in enumerate(cleaned_normalized_context)
        ]

        return renumbered_normalized_context

    def categorize_rule_lines(self, rule):
        either_or = []
        biconditional = []
        others = []

        for item in rule.split("\n"):
            if "(" in item:
                if "\u2295" in item:
                    either_or.append(item)
                elif "\u21d4" in item:
                    biconditional.append(item)
                else:
                    others.append(item)

        return either_or, biconditional, others

    def extract_facts_rules_conjecture(self, content):
        fact = ""
        rule = ""
        conjecture = ""

        fact_matches = list(re.finditer(r"Fact(.*?)Rule", content, re.DOTALL))
        if fact_matches:
            fact = fact_matches[-1].group(1).strip()

        rule_matches = list(re.finditer(r"Rule(.*?)Conjecture", content, re.DOTALL))
        if rule_matches:
            rule = rule_matches[-1].group(1).strip()

        parts = content.rsplit("Conjecture", 1)
        conj = parts[-1].strip() if len(parts) > 1 else ""
        conjecture = re.sub(r"^[\s:\-]*", "", conj)

        return fact, rule, conjecture

    def post_process_decompose(self, content):
        final_form_match = re.finditer(r"Final Form", content)
        final_form_positions = [m.start() for m in final_form_match]

        if final_form_positions:
            first_final_form_pos = final_form_positions[0]

            try:
                context = (
                    content[first_final_form_pos:].split("Final Form:")[-1].strip()
                )

                context_lines = context.split("\n")
                filtered_context_lines = [line for line in context_lines if "(" in line]
                context = "\n".join(filtered_context_lines)

            except IndexError as e:
                problematic_content = content[first_final_form_pos:]
                if self.display: 
                    print("Problematic content: ", problematic_content)
                context = ""
        else:
            context_lines = content.split("\n")
            filtered_context_lines = [line for line in context_lines if "(" in line]
            context = "\n".join(filtered_context_lines)

        return context

    def clean_conjecture(self, conjecture):
        if isinstance(conjecture, dict):
            conjecture = "\n".join(
                [f"{key}: {value}" for key, value in conjecture.items()]
            )
        splitted_conjecture = conjecture.replace("\\n", "\n").split("\n")
        cleaned_conjecture = []
        for item in splitted_conjecture:
            if "(" in item:
                remove_list = [
                    "Rules (others):",
                    "Rules (biconditional):",
                    "Rules (either_or):",
                ]
                if not any(remove_item in item for remove_item in remove_list):
                    splited_item = item.split(":::")[0]
                    cleaned_conjecture.append(splited_item)

        return "\n".join(cleaned_conjecture)

    # TODO: improve the pathing here using relative path for now.
    def load_in_context_examples_trans(self):
        file_path = os.path.join(
            "./prompts", self.params.dataset_name, "translation.txt"
        )
        if self.display: 
            print("Loading translation file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_and_or_decomposer(self):
        file_path = os.path.join(
            "./prompts", self.params.dataset_name, "and_or_decomposer.txt"
        )
        if self.display: 
            print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_either_or_decomposer(self):
        file_path = os.path.join(
            "./prompts", self.params.dataset_name, "either_or_decomposer.txt"
        )
        if self.display: 
            print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_biconditional_decomposer(self):
        file_path = os.path.join(
            "./prompts",
            self.params.dataset_name,
            "logical_biconditional_decomposer.txt",
        )
        if self.display: 
            print("Loading decomposer file: ", file_path)
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_examples_search_init(self):
        file_path = os.path.join(
            "./prompts", self.params.dataset_name, "search_init.txt"
        )
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_examples_search_router(self):
        file_path = os.path.join(
            "./prompts", self.params.dataset_name, "search_router.txt"
        )
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self, split):
        filepath = os.path.join(
            self.params.data_path, self.params.dataset_name, f"{split}.json"
        )
        with open(filepath) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def index_context(self, context):
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = "\n".join(
            [f"{index}: {sentence}" for index, sentence in formatted_context]
        )
        return str(indexed_sentences)

    def save_output(self, outputs, file_suffix=None):
        if "llama" in self.params.model_name:
            model_name = "llama"
        else:
            model_name = self.params.model_name
        file_name = (
            f"{self.params.dataset_name}_{model_name}_trans_decompose_no_negation.json"
        )
        file_path = os.path.join(
            self.params.save_path, self.params.dataset_name, file_name
        )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if self.display: 
            print("Saving result with thread lock in path: ", file_path)
        with self.file_lock:
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        file_content = f.read()
                        if file_content.strip():
                            existing_data = json.loads(file_content)
                        else:
                            if self.display: 
                                print(
                                f"File {file_path} is empty. Initializing with an empty list."
                            )
                            existing_data = []
                else:
                    existing_data = []
                if isinstance(outputs, list):
                    existing_data.extend(outputs)
                else:
                    existing_data.append(outputs)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                if self.display: 
                    print(f"Error in saving output: {e}")
