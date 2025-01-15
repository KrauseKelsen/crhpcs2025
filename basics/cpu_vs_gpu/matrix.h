#ifndef MATRIX_H
#define MATRIX_H
#define BLOCK_SIZE 32

float **alloc_matrix(const int n, const int m);
void print_matrix(const int n, const int m, float **matrix);
void free_matrix(const int n, const int m, float **matrix);
void init_matrix(const int n, const int m, int base, float **matrix);

#endif
