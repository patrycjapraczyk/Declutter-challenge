f = open("./../data/code.txt", "r")

Lines = f.readlines()

for line in Lines:
    # if line.rstrip() != '':
    #line = '"' + line.rstrip() + '"'
    line = line.rstrip() + ','
    # print(line)
    csv_f = open("./../data/code_data_long3.csv", "a")
    csv_f.write(line + '\n')
    csv_f.close()

f.close()
