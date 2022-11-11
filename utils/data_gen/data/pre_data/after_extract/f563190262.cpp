#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <iso646.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>
#include <wctype.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <complex>
#include <csignal>
#include <csetjmp>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <cwctype>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>
#include <complex.h>
#include <fenv.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <tgmath.h>
#include <unistd.h>

using namespace std;


void exact_rhs()
{
double dtemp[5], xi, eta, zeta, dtpp;
int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;
#pragma scop
for (k = 1; k <= grid_points[2]-2; k++) {
for (j = 1; j <= grid_points[1]-2; j++) {
for (i = 1; i <= grid_points[0]-2; i++) {
for (m = 0; m < 5; m++) {
forcing[k][j][i][m] = -1.0 * forcing[k][j][i][m];
}
}
}
}
#pragma endscop

}
