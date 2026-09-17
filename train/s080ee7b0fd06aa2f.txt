#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>
#include <valarray>
#include <array>
#include <queue>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <complex>
#include <random>

using namespace std;
using ll = long long;
using ull = unsigned long long;

const ll INF = 1e18 + 100000;

ll mymul(ll a, ll b) {
    if (a == 0) return 0;
    b = min(b, INF);
    ll res = 0;
    if (a % 2) {
        res += b;
    }
    return min(res + mymul(a/2, b*2), INF);
}

const int MN = 110;
const int MM = 2300;

int main() {
    int n, m;
    ll k;
    cin >> n >> m >> k;
    string s[MN];
    int sz[MN];
    for (int i = 0; i < n; i++) {
        cin >> s[i];
        sz[i] = (int)s[i].size();
    }

    ll dp[MM] = {};
    dp[0] = 1;
    for (int i = 1; i <= m; i++) {
        for (int j = 0; j < n; j++) {
            if (i < sz[j]) continue;
            dp[i] += dp[i-sz[j]];
            dp[i] = min(dp[i], INF);
        }
    }

    if (dp[m] < k) {
        cout << "-" << endl;
        return 0;
    }

    bool tabuu[MN][MM] = {};
    ll num[MM] = {};
    num[0] = 1;

    string result;
    for (int fe = 0; fe < m; fe++) {
        for (int i = 0; i < n; i++) {
            if (sz[i] <= fe && !tabuu[i][fe-sz[i]]) {
                num[fe] += num[fe-sz[i]];
                num[fe] = min(num[fe], INF);
            }
        }

        ll cnt[26] = {};
        for (int i = 0; i < n; i++) {
            for (int j = max(0, fe-sz[i]+1); j <= fe; j++) {
                if (tabuu[i][j]) continue;
                int c = s[i][fe-j] - 'a';
                cnt[c] += mymul(num[j], dp[m-(j+sz[i])]);
                cnt[c] = min(cnt[c], INF);
            }
        }
        for (int i = 0; i < 26; i++) {
            if (k <= cnt[i]) {
                result += i+'a';
                break;
            }
            k -= cnt[i];
        }

        for (int i = 0; i < n; i++) {
            for (int j = max(0, fe-sz[i]+1); j <= fe; j++) {
                if (s[i][fe - j] != result[fe]) {
                    tabuu[i][j] = true;
                }
            }
        }
    }

    cout << result << endl;
    return 0;
}