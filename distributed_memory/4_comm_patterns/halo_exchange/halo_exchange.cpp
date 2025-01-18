#include <mpi.h>
#include <iostream>
#include <sstream>
#include "write_halo.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the 3D grid dimensions
    int dims[3] = {2, 2, 2};    // 2x2x2 Cartesian grid
    int periods[3] = {1, 1, 1}; // Non-periodic boundaries
    int coords[3];              // Rank's coordinates in the grid
    
    MPI_Comm cart_comm;

    // Create Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Neighbors in each direction
    int nbr_left, nbr_right, nbr_up, nbr_down, nbr_front, nbr_back;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_left, &nbr_right);  // x-direction
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_down, &nbr_up);     // y-direction
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_back, &nbr_front);  // z-direction

    std::cout << "Here" << std::endl;


    // Local 3D grid for each process
    const int N = 4;                        // Local size of the grid in each dimension
    int *grid = new int[N * N * N];         // Allocate 3D grid dynamically

    for (int i = 0; i < N * N * N; ++i) {
        grid[i] = rank;                     // Initialize local grid with the rank
    }

    std::ostringstream filename;
    filename << "output_rank_before_" << rank << ".vtk";
    
    // Determine process coordinates 
    int phys_coords[3] = {rank % 2, (rank / 2) % 2, rank / 4};

    //write_vtk(filename.str(), grid, N, rank, phys_coords);

    // Buffers for halo exchange
    int *send_x = new int[N * N];
    int *recv_x = new int[N * N];
    int *send_y = new int[N * N];
    int *recv_y = new int[N * N];
    int *send_z = new int[N * N];
    int *recv_z = new int[N * N];

    // Prepare send buffers (example: sending first layer of each dimension)
    for (int i = 0; i < N * N; ++i) {
        send_x[i] = grid[i];          // First layer in x-direction
        send_y[i] = grid[i * N];      // First layer in y-direction
        send_z[i] = grid[i * N * N];  // First layer in z-direction
    }

    // Exchange in x-direction
    MPI_Sendrecv(send_x, N * N, MPI_INT, nbr_left, 0,
                 recv_x, N * N, MPI_INT, nbr_right, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Exchange in y-direction
    MPI_Sendrecv(send_y, N * N, MPI_INT, nbr_down, 1,
                 recv_y, N * N, MPI_INT, nbr_up, 1,
                 cart_comm, MPI_STATUS_IGNORE);

    // Exchange in z-direction
    MPI_Sendrecv(send_z, N * N, MPI_INT, nbr_back, 2,
                 recv_z, N * N, MPI_INT, nbr_front, 2,
                 cart_comm, MPI_STATUS_IGNORE);

    // Print results
    std::cout << "Rank " << rank << " completed exchanges." << std::endl;

    // Clean up dynamically allocated memory
    delete[] grid;
    delete[] send_x;
    delete[] recv_x;
    delete[] send_y;
    delete[] recv_y;
    delete[] send_z;
    delete[] recv_z;

    MPI_Finalize();
    return 0;
}
