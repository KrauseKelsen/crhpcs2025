#include <mpi.h>      
#include <iostream> 
#include <cmath>   
#include <chrono>     

#define PI25DT 3.141592653589793238462643 

int main(int argc, char** argv) {
    int N, rank, size;  // cantidad de intervalos y variables para mpix
    double pi, local_sum = 0.0, global_sum = 0.0; // suma local y suma global
    double h, x; // n subintervalos 10^6 y valor de x


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // validar el numero de argumentos
    if (argc < 2) {
        if (rank == 0) { // proceso maestro
            std::cerr << "Usage: " << argv[0] << " <N>\n";
        }
        MPI_Finalize(); // asesinar al mpi
        return -1;
    }
    
    N = std::atoi(argv[1]); // conversion del argumento a int
    h = 1.0 / (double) N;   // calcular el size de cada subintervalo
    
    // tiempo de inicio del proceso 0
    auto start = std::chrono::steady_clock::now();

    // cada proceso calcula una parte de la sumatoria 
    for (int i = rank + 1; i <= N; i += size) { 
        x = h * ((double)i - 0.5);          // calcular la posicion en x
        local_sum += 4.0 / (1.0 + x * x);   // suma local de la funcion
    }
    
    // reducir la suma local de cada proceso en global_sum usando MPI_SUM en el proceso 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // calcular pi en el proceso 0
    if (rank == 0) {
        pi = h * global_sum; // calculando pi
        auto total_time = std::chrono::steady_clock::now() - start; // tiempo transcurrido
        printf("Total time: %.5f seconds\n", std::chrono::duration<double>(total_time).count());
        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    }

    MPI_Finalize(); // Finaliza MPI
    return 0;
}
