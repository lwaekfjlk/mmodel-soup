import os
import jsonlines

subset_names = ['AS', 'R', 'U']
file_dir = '/mustard_data/expert_inference_output/expert_mistral'
check_dir = '/mustard_data/expert_inference_output/expert_blip2'

for name in subset_names:
    file_path = os.path.join(file_dir, f'mustard_{name}_yesno_logits.jsonl')
    check_path = os.path.join(check_dir, f'mustard_{name}_yesno_logits.jsonl')
    
    with jsonlines.open(file_path, 'r') as f, jsonlines.open(check_path, 'r') as g:
        file_lines = list(f)
        check_lines = list(g)
        
        # Ensure both files have the same number of lines
        if len(file_lines) != len(check_lines):
            print(f"Difference in number of lines for {name}")
        else:
            # Compare each line
            for idx, (line_file, line_check) in enumerate(zip(file_lines, check_lines)):
                if line_file != line_check:
                    print(f"Difference found in subset {name} at line {idx + 1}")
                    print("File line:", line_file)
                    print("Check line:", line_check)
