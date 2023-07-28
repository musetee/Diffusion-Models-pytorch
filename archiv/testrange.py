from tqdm import tqdm
for i in tqdm(reversed(range(1, 1001)), position=0):
    if i%100 == 0:
        print(i)