#include <cstdio>

int main(int argc, char **argv){
    int Nrows = 6; 
    int Ncols = 6;

    int *A = new int[Nrows*Ncols];
    int *x = new int[Ncols];
    int *b = new int[Ncols];

    for (int i = 0; i < Ncols; i++) {
        x[i] = i + 1;
    }    

    for (int j = 0; j < Nrows; j++) {
        b[j] = 0;
        for (int i = 0; i < Ncols; i++) {
            A[j*Ncols+i] = 1;
        } 
    }

    // compute A*x

    for (int j = 0; j < Nrows; j++) {
        for (int i = 0; i < Ncols; i++){
            b[j] += A[j*Ncols+i]*x[i];
        }
    }

    printf("Result:\n");
    for (int i = 0; i < Nrows; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
    return 0; 
}