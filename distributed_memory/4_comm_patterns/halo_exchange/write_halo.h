#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkPointData.h>
#include <vtkIntArray.h>

void write_vtk(const std::string &filename, int *grid, int N, int rank, int coords[3]) {
    // Create VTK structured points
    auto structuredPoints = vtkSmartPointer<vtkStructuredPoints>::New();
    structuredPoints->SetDimensions(N, N, N);
    structuredPoints->SetOrigin(coords[0] * N, coords[1] * N, coords[2] * N);
    structuredPoints->SetSpacing(1.0, 1.0, 1.0);

    // Create a data array to store grid values
    auto dataArray = vtkSmartPointer<vtkIntArray>::New();
    dataArray->SetName("GridData");
    dataArray->SetNumberOfComponents(1);
    dataArray->SetNumberOfTuples(N * N * N);

    // Fill the data array
    for (int i = 0; i < N * N * N; ++i) {
        dataArray->SetValue(i, grid[i]);
    }

    // Attach data array to structured points
    structuredPoints->GetPointData()->SetScalars(dataArray);

    // Write the structured points to a file
    auto writer = vtkSmartPointer<vtkStructuredPointsWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(structuredPoints);
    writer->SetFileTypeToBinary(); // Use binary output
    writer->Write();
}