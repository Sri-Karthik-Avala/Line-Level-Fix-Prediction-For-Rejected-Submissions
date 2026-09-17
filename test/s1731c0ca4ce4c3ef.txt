#include <bits/stdc++.h>
using namespace std;
#define rep(i,n) for(int i=0;i<n;i++)
int a[200010], b[200010];
int main() {
	int n, m;
	cin >> n;
	rep(i, n)cin >> a[i];
	cin >> m;
	rep(i, m)cin >> b[i];
	rep(i, n)if (!binary_search(b, b + m, a[i]))cout << a[i] << endl;
	rep(i, m)if (!binary_search(a, a + n, b[i]))cout << b[i] << endl;
	return 0;
}
