
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkXMLPolyDataWriter.h>

#include <cstdlib>
#include <ctime>

#define G 1
#define M 1000
#define PI 3.141592653589793238462643

using View = Kokkos::View<double**, Kokkos::Device<Kokkos::DefaultExecutionSpace,Kokkos::SharedSpace>>;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;    // particle force calculation "loop"

void calculate_forces(View& particles, int Np){

  // zero out forces
  Kokkos::parallel_for("zero_out_forces", Np, KOKKOS_LAMBDA(const int i) {
    particles(i, 7) = 0.0;
    particles(i, 8) = 0.0;
    particles(i, 9) = 0.0;
  });
  
  Kokkos::parallel_for("calculate_forces", mdrange_policy({0,0}, {Np,Np}), KOKKOS_LAMBDA(const int j, const int i) {
    if (i != j)
    { 
      // Calculation with other particles
      double dx = particles(i,1) - particles(j,1);
      double dy = particles(i,2) - particles(j,2);
      double dz = particles(i,3) - particles(j,3);
      
      double r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;

      double Fmag = G*(particles(i,0)*particles(j,0))/(r*r);

      particles(i, 7) += -Fmag*dx/r;
      particles(i, 8) += -Fmag*dy/r;
      particles(i, 9) += -Fmag*dz/r;

      // Calculation with central mass
      dx = particles(i,1);
      dy = particles(i,2);   // Central mass is located at (0,0,0)
      dz = particles(i,3);

      r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;

      Fmag = G*(M*particles(i,0))/(r*r);
      
      particles(i, 7) += -Fmag*dx/r;
      particles(i, 8) += -Fmag*dy/r;
      particles(i, 9) += -Fmag*dz/r;
    }
  });
}

int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv);
  {
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

    int Np = atoi(argv[1]);       // Number of particles in simulation
    int Nt = atoi(argv[2]);       // Number of timesteps for simulation
    int N_write = atoi(argv[3]);  // Write frequency for output
    double dt = atof(argv[4]);    // timestep for simulation
    //L = atof(argv[4]);          // length scale (for initial conditions)
    //mass = atof(argv[6]);     // particle mass

    // Initialize data structure to hold particles 
    View particles("particles", Np, 10); // each particle (mass,x,y,z,vx,vy,vz,ax,ay,az)

    // Initialize the starting point for all particles
    Kokkos::parallel_for("initialize_particles", Np, KOKKOS_LAMBDA(const int i) {
      auto generator = random_pool.get_state();

      double r = generator.drand(10., 50.);
      double theta = 2*PI*generator.drand(0., 1.);
      double z = 10*generator.drand(0., 1.);

      particles(i, 0) = generator.drand(0.5, 1.5);

      particles(i, 1) = r*cos(theta); // x
      particles(i, 2) = r*sin(theta); // y
      particles(i, 3) = z; // z

      double eps = generator.drand(-0.01, 0.01);
      double vt = 2.*sqrt(G*M/r)*(1+eps);

      //double r_hat[3]; 
      //r_hat[0] = particles(i, 1); 
      //r_hat[1] = particles(i, 2); 
      //r_hat[2] = particles(i, 3); 

      // calculate tangential velocity (r_hat x k_hat)

      particles(i, 4) = vt*particles(i, 2)/r;
      particles(i, 5) = -vt*particles(i, 1)/r;
      particles(i, 6) = 0.0;

      random_pool.free_state(generator);
    });

    for (int step = 0; step < Nt; step++)
    {
      // Visualize
      if (step % N_write == 0)
      {
        std::cout << "Step: " << step << std::endl;

        // VTK Routines for plotting
        vtkNew<vtkPoints> points;
        
        // Iterate over particles to create VTK structures
        for (int i = 0; i < Np; ++i) {
          points->InsertNextPoint(
              particles(i, 1), 
              particles(i, 2), 
              particles(i, 3)
          );
        }

        vtkNew<vtkDoubleArray> velocities;
        vtkNew<vtkDoubleArray> forces;
        velocities->SetName("Velocity");
        velocities->SetNumberOfComponents(3);
        forces->SetName("Force");
        forces->SetNumberOfComponents(3);

        for (int i = 0; i < Np; i++) {
          velocities->InsertNextTuple3(particles(i,4), particles(i,5), particles(i,6));
          forces->InsertNextTuple3(particles(i,7), particles(i,8), particles(i,9));
        }

        
        
        std::ostringstream filename;
        filename << "nbody_" << step << ".vtp";

        vtkNew<vtkPolyData> polyData;

        polyData->GetPointData()->AddArray(velocities);
        polyData->GetPointData()->AddArray(forces);

        polyData->SetPoints(points);
        vtkNew<vtkXMLPolyDataWriter> writer;
        writer->SetFileName(filename.str().c_str());
        writer->SetInputData(polyData);
        writer->Write();
      }

      calculate_forces(particles, Np);

      Kokkos::parallel_for("kick_leap_1", Np, KOKKOS_LAMBDA(const int i) {
        particles(i,4) += 0.5*dt*particles(i,7) / particles(i,0);
        particles(i,5) += 0.5*dt*particles(i,8) / particles(i,0);
        particles(i,6) += 0.5*dt*particles(i,9) / particles(i,0);

        particles(i,1) += particles(i,4)*dt; 
        particles(i,2) += particles(i,5)*dt;
        particles(i,3) += particles(i,6)*dt;
      }); 

      calculate_forces(particles, Np);

      Kokkos::parallel_for("kick_leap_2", Np, KOKKOS_LAMBDA(const int i) {
        particles(i,4) += 0.5*dt*particles(i,7) / particles(i,0);
        particles(i,5) += 0.5*dt*particles(i,8) / particles(i,0);
        particles(i,6) += 0.5*dt*particles(i,9) / particles(i,0);
      });
    }
  } 
  Kokkos::finalize();
}


