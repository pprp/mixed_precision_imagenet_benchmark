import matplotlib.pyplot as plt


f = open("./exp_4_nohup.out", 'r')

flines = f.readlines()

line_content_train = []
line_content_val = []

for fline in flines:
    if fline.startswith('/') or fline.startswith('Gradient'):
        continue
    if fline.strip().startswith('tensor') or fline.strip().startswith('warning'):
        continue
    if fline.strip() == '':
        continue
    if not fline.startswith("Epoch") and not fline.startswith("Test"):
        continue

    if fline.startswith("Epoch"):
        line_content_train.append(fline)
    if fline.startswith("Test"):
        line_content_val.append(fline)

with open("p4_nohup_train.out", 'w') as fo:
    for line in line_content_train:
        fo.write(line)

with open("p4_nohup_val.out", 'w') as fo:
    for line in line_content_val:
        fo.write(line)

f.close()
