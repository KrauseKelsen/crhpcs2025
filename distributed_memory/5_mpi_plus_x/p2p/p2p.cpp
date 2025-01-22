#include <mpi.h>
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize the MPI environment
    Kokkos::initialize();
    {
        int rank, size, err;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
        MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes

        if (size < 2 || size > 2) {
            if (rank == 0) {
                std::cout << "This program requires only two processes." << std::endl;
            }
            MPI_Finalize();
            return 1;
        }

        Kokkos::View<double*> send_buf;
        Kokkos::View<double*> recv_buf;

        if (rank == 0) {
            // Process 0 sends a message
            send_buf = Kokkos::View<double*>("send_buf", 10); // Data to send

            Kokkos::parallel_for("populate_data", 10, KOKKOS_LAMBDA (const int i){
                send_buf(i) = i*2.;
            });

            err = MPI_Send(send_buf.data(), 10, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

            printf("Sent data:\n");
            for (int i = 0; i < 10; i++) {
                printf("%f ", send_buf(i));
            }
            printf("\n");

            } else if (rank == 1) {

            // Process 1 receives a message
            recv_buf = Kokkos::View<double*>("recv_buf", 10); // Data to send
            MPI_Status status; // To check the details of the received message
            err = MPI_Recv(recv_buf.data(), 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            
            Kokkos::parallel_for("rescale_data", 10, KOKKOS_LAMBDA (const int i){
                recv_buf(i) = recv_buf(i)/2.;
            });

            printf("Received data:\n");
            for (int i = 0; i < 10; i++) {
                printf("%f ", recv_buf(i));
            }
            printf("\n");
        }
    }
    
    Kokkos::finalize();
    MPI_Finalize(); // Finalize the MPI environment
    return 0;
}
