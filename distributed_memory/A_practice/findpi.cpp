#include <math.h>
#include <cstdio>

#define PI25DT 3.141592653589793238462643

int main(int argc, char** argv){
    
    int n;
    double pi, h, sum, x;
    
    printf("Enter the number of intervals: (0 is invalid) ");
    scanf("%d",&n);

    h = 1.0 / (double) n;
    sum = 0.0; 

    for (int i = 0; i <= n; i++) {
        x = h*((double)i-0.5);
        sum += 4.0 / (1.0 + x*x);
    }
    pi = h*sum;
    printf("pi is approximately %.16f, Error is %.16f\n",
      pi, fabs(pi - PI25DT));

    return 0;
}