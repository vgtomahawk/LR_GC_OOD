import sys
from io import open

fine_grained_file = open(sys.argv[1],"r",encoding="utf-8")
coarse_grained_file = open(sys.argv[2],"w",encoding="utf-8")


for line in fine_grained_file.readlines():
    words=line.strip().split("\t")
    fine_label = words[0]
    if "/" in fine_label:
        coarse_label = fine_label.split("/")[0]
        words[0] = coarse_label
    coarse_line = "\t".join(words)+"\n"
    coarse_grained_file.write(coarse_line)

fine_grained_file.close()
coarse_grained_file.close()
