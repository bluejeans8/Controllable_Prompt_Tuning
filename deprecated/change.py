with open("result.txt", "r") as rf:
    with open("result_bert-base-cased.txt", "w") as wf:
        for line in rf.readlines():
            a = line.split(",")[:-1]
            string = ",".join(a) + f", {float(a[-2])/float(a[-1])}\n"
            wf.write(string)