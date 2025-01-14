#include <stdio.h>
#include <stdlib.h>

void transpose_matrix_naive(const int n, const int m, const float **origin, float **result)
{
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j)
    {
      result[i][j] = origin[j][i];
    }
  }
}

void print_matrix(const int n, const int m, float **matrix)
{
  for(int i = 0; i < n; ++i)
  {
    for(int j = 0; j < m; ++j)
    {
      printf("%.2f\t", matrix[i][j]);
    }
    printf("\n");
  }
}

float **alloc_matrix(const int n, const int m)
{
  float **matrix = malloc(sizeof(float*) * m);
  for(int j = 0; j < m; ++j)
  {
    matrix[j] = malloc(sizeof(float) * n);
  }
  return matrix;
}

void free_matrix(const int n, const int m, float **matrix)
{
  for(int j = 0; j < m; ++j)
  {
    free(matrix[j]);
  }
  free(matrix);
}

void init_matrix(const int n, const int m, const int base, float **matrix)
{
  for(int i = 0; i < n; ++i)
  {
    for(int j = 0; j < m; ++j)
    {
      matrix[i][j] = base * i + j;
    }
  }
}

int main(void)
{
  printf("Hello, world!\n");
  int n = 4;
  int m = 5;
  float **matA = alloc_matrix(n, m);
  init_matrix(n, m, 3, matA);
  print_matrix(n, m, matA);
  free_matrix(n, m, matA);
  return 0;
}
