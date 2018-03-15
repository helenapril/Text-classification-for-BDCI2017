import csv

f1 = open('answer1.csv', 'rb')
f2 = open('answer2.csv', 'rb')
f3 = open('answer3.csv', 'rb')
reader1 = csv.reader(f1)
reader2 = csv.reader(f2)
reader3 = csv.reader(f3)

f4 = open('answer.csv', "wb")
writer = csv.writer(f4)

for row1, row2, row3 in zip(reader1, reader2, reader3):
    sum = int(row1[1]) + int(row2[1]) + int(row3[1])
    print sum
    if sum < 3:
        writer.writerow([row1[0], 'NEGATIVE'])
    else:
        writer.writerow([row1[0], 'POSITIVE'])

