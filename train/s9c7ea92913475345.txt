n = int(raw_input())
for loop in xrange(n):
    a = []
    while len(a) < 8:
        a += map(lambda i:int(i,16),raw_input().split())
    key = 0
    
    for i in xrange(32):
        s = sum(a[j]^key for j in xrange(8))
        if s&(1 << (i+1)-1) != (a[-1]^key)&(1 << (i+1)-1): key |= (1 << i)
    print "%x"%key