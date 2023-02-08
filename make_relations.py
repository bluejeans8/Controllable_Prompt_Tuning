import glob
import re
import json
import os

relation_files = (('/home/tjrals/jinseok/js_p-tuning/data/LAMA/single_relations/P*'))

with open('./test_data/test_original_relations.json', 'w') as wf:
    line_cnt = 0
    for file in glob.glob(relation_files):
        ver = re.findall(r'P\d*', file)[0]
        test_file = '/home/tjrals/jinseok/js_p-tuning/data/LAMA/fact-retrieval/original/' + ver + '/test.jsonl'
        if os.path.isfile(test_file) == False:
            continue
        with open(file, 'r') as single:
            relation = json.loads(single.readline())
            template = relation['template']                
            with open(test_file, 'r') as fact:
                for data in fact.readlines():
                    data = json.loads(data)
                    sub_label = data['sub_label']
                    obj_label = data['obj_label']
                    masked_sentence = template.replace('[X]', sub_label).replace('[Y]', '[MASK]')
                    if '"' in masked_sentence:
                        continue
                    text = f'{{"masked_sentence": "{masked_sentence}", "obj_label": "{obj_label}"}}\n'
                    wf.write(text)
                    line_cnt+=1

                
                
        

