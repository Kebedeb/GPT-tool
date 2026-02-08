import random as rd

file =open('sample_output_final.txt', 'w')
for i in range(5):
    a = rd.randint(0, 9)
    b = rd.randint(0, 9)
    print(a, '+', b, "=", a + b)
    file.write(f"{a}+{b}\n")

for i in range(5):
    a = rd.randint(10, 99)
    b = rd.randint(10, 99)
    file.write(f"{a}+{b}\n")

for i in range(5):
    a = rd.randint(100, 999)
    b = rd.randint(100, 999)
    file.write(f"{a}+{b}\n")

for i in range(5):
    a = rd.randint(1000, 9999)
    b = rd.randint(1000, 9999)
    file.write(f"{a}+{b}\n")
    
   

