import json
import re


def _conjecture(conjecture):
    if isinstance(conjecture, str):
        if re.search(r"True", conjecture):
            updated_conjecture = re.sub(r"True", "False", conjecture)
        elif re.search(r"False", conjecture):
            updated_conjecture = re.sub(r"False", "True", conjecture)
        else:
            updated_conjecture = "false"
    else:
        updated_conjecture = "Invalid input: conjecture must be a string"
        print(f"Invalid input {conjecture}: conjecture must be a string")
    return updated_conjecture


def negate(dataset_name: str, model_name: str):
    input_path = f"./results/{dataset_name}/{dataset_name}_{model_name}_trans_decompose_no_negation.json"
    with open(input_path, "r") as f:
        data = json.load(f)
    for item in data:
        item["sos_list"] = _conjecture(item["sos_list"])
        item["negated_label"] = "True"

    save_path = f"./results/{dataset_name}/{dataset_name}_{model_name}_trans_decompose_negated_data.json"
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4) 
    return data 


if __name__ == "__main__":
    dataset_name = "LogicNLI" 
    model_name = "Qwen/Qwen2.5-7B-Instruct" 
    out = negate(dataset_name, model_name) 



