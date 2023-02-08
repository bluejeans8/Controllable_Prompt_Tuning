import glob

dev_files = (('/home/tjrals/jinseok/js_p-tuning/data/LAMA/fact-retrieval/original/P*/test*'))
with open('./test_data/test_original.json', 'w') as wf:
    for file in glob.glob(dev_files):
        with open(file, 'r') as rf:
            for line in rf:
                wf.write(line)


