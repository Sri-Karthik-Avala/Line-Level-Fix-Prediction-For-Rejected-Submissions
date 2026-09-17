n,a = map(int,input().split())
x = list(map(int,input().split()))


dp = [[[0]*(55) for i in range(3000)]for i in range(55)]
dp[0][0][0] = 1

for i in range(n):
  for j in range(n*a):
    for k in range(n):
      dp[i+1][j][k] += dp[i][j][k]
      dp[i+1][j+x[i]][k+1] += dp[i][j][k]
      
    
ans = 0
for k in range(1,n):
  ans += dp[n][k*a][k]
print(ans)