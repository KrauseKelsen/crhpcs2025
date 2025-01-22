#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
// Define the Kokkos array structure

typedef Kokkos::View<double*, 
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::SharedSpace>> realArr; 

// Define Kokkos view

int main(int argv, char *argc[]) {

    MPI_Init(&argc, &argv);

    // read-in simulation parameters 
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int parNx = atoi(argv[3]);
    int parNy = atoi(argv[4]);
    int GpN = atoi(argv[5]);

    // Calculate simulation domain extents
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx / (Nx + 1);
    double dy = Ly / (Ny + 1);

    // Set number of halo (ghost/communication) cells
    
    // Get procid and comm size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::InitializationSettings kokkosSettings;
    Kokkos::Timer timer;

    // Initalize Kokkos

#if KOKKOS_ENABLE_CUDA
    kokkosSettings.set_device_id(rank % GpN);          // choose device id based on number of 
#endif

    Kokkos::initialize(kokkosSettings);

    // Define views for the Laplace solution and temporary storage (in scope)
    {
        int Ntot = (Nx+2)*(Ny+2);  // 2 halo cells at the ends of the dimensions

        realArr u_old("u_old", Ntot);
        realArr u("u", Ntot);
        realArr u_new("u_new", Ntot);


        // Initialize phi with initial guess (e.g., all zeros)
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::fence();

        // Set boundary conditions (e.g., Dirichlet boundary conditions)
        Kokkos::parallel_for("set_boundary", Ntot,
            KOKKOS_LAMBDA(int iGlob) {
                int i,j; 
                i = iGlob / Nx;
                j = iGlob % Nx;
                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                    phi(iGlob) = 1.0; // Set boundary values
                }
            });
        Kokkos::fence();

        // Solve the Laplace equation using Jacobi iteration
        constexpr int max_iterations = 1000;
        
        time1 = timer.seconds();
        for (int iter = 0; iter < max_iterations; ++iter) {
            Kokkos::parallel_for("laplace_iter", Ntot,
                KOKKOS_LAMBDA(int iGlob) {
                    int i,j; 
                    i = iGlob / Nx;
                    j = iGlob % Nx;
                    if (i != 0 && i != Nx - 1 && j != 0 && j != Ny - 1) {
                        phi_new(iGlob) = 0.25 * (phi(iGlob+1) + phi(iGlob-1) + phi(iGlob+(Nx)) + phi(iGlob-(Nx)));
                    }
                    else
                    {
                        phi_new(iGlob) = 1.0;
                    }
                });
            Kokkos::fence();            
            // Swap phi and phi_new
            Kokkos::deep_copy(phi, phi_new);
        }
            runtime = timer.seconds() - time1;

    }
    insitu.Finalize(); 
    Kokkos::finalize();
    MPI_Finalize();

    std::cout << "Elapsed time: " << std::setprecision(5) << runtime << " seconds." << std::endl;
    return 0;
}