import glob
import re
import json
import os
import argparse

parser = argparse.ArgumentParser()

TYPES = ['train', 'dev', 'test']
parser.add_argument("--dataset_type", type=str, default='test', choices=TYPES)

args = parser.parse_args()

relation_files = (('/home/tjrals/jinseok/Prompting/data/LAMA/single_relations/P*'))

with open(f'{args.dataset_type}_original_relations.json', 'w') as wf:
    line_cnt = 0
    relation_cnt = 0
    for file in glob.glob(relation_files):
        ver = re.findall(r'P\d+', file)[0]
        test_file = '/home/tjrals/jinseok/Prompting/data/LAMA/fact-retrieval/original/' + ver + f'/{args.dataset_type}.jsonl'
        if os.path.isfile(test_file) == False:
            continue
        relation_cnt+=1
        with open(file, 'r') as single:
            relation = json.loads(single.readline())
            template = relation['template']
            with open(test_file, 'r') as fact:
                for data in fact.readlines():
                    data = json.loads(data)
                    sub_label = data['sub_label']
                    obj_label = data['obj_label']
                    if '"' in template or '"' in obj_label or '"' in sub_label:
                        continue
                    text = f'{{"predicate_id": {relation_cnt}, "relation": "{template}", "sub_label": "{sub_label}", "obj_label": "{obj_label}"}}\n'
                    wf.write(text)
                    line_cnt+=1
print(line_cnt)

                
                
        

