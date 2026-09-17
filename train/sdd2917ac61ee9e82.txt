h, w = map(int, input().split())
max_rect, prev = 0, [0] * (w + 1)
for i in range(h):
    current = [p + 1 if f else 0 for f, p in zip(map(lambda x: int(x) ^ 1, input().split()), prev)] + [0]
    stack = [(0, 0)]
    for j in range(w + 1):
        c_j = current[j]
        if stack[-1][0] < c_j:
            stack.append((c_j, j))
            continue
        if stack[-1][0] > c_j:
            while stack[-1][0] > c_j:
                height, since = stack.pop()
                max_rect = max(max_rect, height * (j - since))
            if c_j:
                stack.append((c_j, j))
    prev = current

print(max_rect)