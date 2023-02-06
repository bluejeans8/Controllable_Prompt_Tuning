import glob

dev_files = (('/home/tjrals/jinseok/js_p-tuning/data/LAMA/fact-retrieval/trex/P*/dev*'))
with open('dev_trex.json', 'w') as wf:
    for file in glob.glob(dev_files):
        with open(file, 'r') as rf:
            for line in rf:
                print(line)
                wf.write(line)
    
    