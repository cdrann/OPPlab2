//
// Created by Admin on 11.05.2020.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define type runtime //static…

using namespace std;

#define N 15000
#define eps 1e-6

void matrMul(double *A, double *x, double *xnew) {
#pragma omp parallel for schedule (type)
    for (int i = 0; i < N; i++) {
        xnew[i] = 0;
        for (int j = 0; j < N; j++) {
            xnew[i] += A[j + i * N] * x[j];
        }
    }
}
int main() {
    omp_set_num_threads(4);

    double startTime = omp_get_wtime();

    double *x = nullptr, *b = nullptr, *xnew = nullptr, tau = 0.0001, norm = 0, norm_b = 0, sum = 0;

    x = (double *)calloc(N, sizeof(double));
    b = (double *)malloc((N + 1) * sizeof(double));
    xnew = (double *)calloc(N, sizeof(double));

    // инициализация b
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1;
    }

    // инициализация А
    double *A = nullptr;
    A = (double *)malloc(N * N * sizeof(double));
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    // норма b
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        sum += b[i] * b[i];
    norm_b = sqrt(sum); // ||b||

    matrMul(A, x, xnew);
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    double norm_0 = sqrt(sum); // ||Ax_0 - b||
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        xnew[i] = -tau * xnew[i];
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        xnew[i] = x[i] + xnew[i];
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        x[i] = xnew[i];

    sum = 0;
    matrMul(A, x, xnew);
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
#pragma omp parallel for schedule  (type)
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    norm = sqrt(sum); // ||Ax_1 - b||

    // Если с некоторым знаком решение начинает расходиться,
    // то следует сменить его на противоположный
    if (norm_0 <= norm)
        tau = -tau;

    norm /= norm_b; // ||Ax_1 - b|| / ||b||

    while (norm >= eps) {
        sum = 0;
        matrMul(A, x, xnew);

#pragma omp parallel for schedule  (type)
        for (int i = 0; i < N; i++)
            xnew[i] = xnew[i] - b[i]; //(Ax^n - b)

        //  ||Ax_1 - b|| / ||b||
#pragma omp parallel for schedule  (type)
        for (int i = 0; i < N; i++)
            sum += xnew[i] * xnew[i];
        norm = sqrt(sum) / norm_b;

#pragma omp parallel for schedule  (type)
        for (int i = 0; i < N; i++)
            xnew[i] = -tau * xnew[i]; // -tau * (Ax^n - b)

#pragma omp parallel for schedule  (type)
        for (int i = 0; i < N; i++)
            xnew[i] = x[i] + xnew[i]; // x^n - tau * (Ax^n - b)

#pragma omp parallel for schedule  (type)
        for (int i = 0; i < N; i++)
            x[i] = xnew[i]; // x^(n+1) = x^n - tau * (Ax^n - b)
    }

    double endTime = omp_get_wtime( );

    cout << "Res: " << "\n";
    for (int i = 0; i < N; i++) {
        cout << "x[" << i << "] = " << xnew[i] << "\n";
    }
    cout << "Time: "<< (double)(endTime - startTime) << endl;

    return 0;
}
