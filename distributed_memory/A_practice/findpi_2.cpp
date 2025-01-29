#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#define PI25DT 3.141592653589793238462643


int main(int argc, char** argv) {
    const int total_points = 1000000000;
    int local_count = 0;
    
    // Seed for random number generation
    srand(42);

    for (int i = 0; i < total_points; i++){
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }
    double pi = 4.0 * local_count / total_points;
    printf("pi is approximately %.16f, Error is %.16f\n",
      pi, fabs(pi - PI25DT));
}