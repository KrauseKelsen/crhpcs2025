#include <iostream>
#include <mpi.h>

int main(int argc, char*argv[])
{
    int error;
    error = MPI_Init(&argc, &argv); // error-checking

    std::cout << "Hello World!\n" << std::endl;
    
    error = MPI_Finalize();
    
    return 0;
}