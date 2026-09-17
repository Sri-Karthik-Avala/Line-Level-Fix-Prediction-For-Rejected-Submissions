import sys
import math
N = int(input())
array = list(map(int,input().split()))
set_array = set(array)

if not ( 3 <= N <= 10**5 and N % 2 == 1 ): sys.exit()
if len(array) == len(list(set_array)): print(0);sys.exit()

print(len(list(set_array))) if len(list(set_array)) % 2 == 1 else print(len(list(set_array)) - 1)