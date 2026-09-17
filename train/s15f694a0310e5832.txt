#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#define rep(i,n) Rep(i,0,n)
#define Rep(i,k,n) for(int i=k ; i<n ; i++)
#define vi vector<int>
#define Sort(v) sort(v.begin(),v.end())

//const int MOD = 1000000007;
//const int INF = 1<<30;
using namespace std;




int main(){
    int r; cin >> r;
    
    double pi = 3.141592653589;
    double s = r*r*pi;
    double l = 2*r*pi;
    
    cout << fixed;
    cout <<setprecision(17) << s << " " << l << endl;
    
    
}







