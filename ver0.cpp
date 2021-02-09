//
// Created by Admin on 11.05.2020.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;

#define N 15000
#define eps 1e-6

void matrMul(double *A, double *x, double *xnew) {
    for (int i = 0; i < N; i++) {
        xnew[i] = 0;
        for (int j = 0; j < N; j++) {
            xnew[i] += A[j + i * N] * x[j];
        }
    }
}

int main() {
    clock_t startTime = clock();

    double *x = nullptr, *b = nullptr, *xnew = nullptr, tau = 0.0001, norm = 0, norm_b = 0, sum = 0;

    x = (double *)calloc(N, sizeof(double));
    b = (double *)malloc((N + 1) * sizeof(double));
    xnew = (double *)calloc(N, sizeof(double));

    // инициализация b
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1;
    }

    // инициализация А
    double *A = nullptr;
    A = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    // норма b
    for (int i = 0; i < N; i++)
        sum += b[i] * b[i];
    norm_b = sqrt(sum);

    matrMul(A, x, xnew);
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    double norm_0 = sqrt(sum); // ||Ax_0 - b||
    for (int i = 0; i < N; i++)
        xnew[i] = -tau * xnew[i];
    for (int i = 0; i < N; i++)
        xnew[i] = x[i] + xnew[i];
    for (int i = 0; i < N; i++)
        x[i] = xnew[i];

    sum = 0;
    matrMul(A, x, xnew);
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    norm = sqrt(sum);

    // Если с некоторым знаком решение начинает расходиться,
    // то следует сменить его на противоположный
    if (norm_0 <= norm)
        tau = -tau;

    norm /= norm_b; // ||Ax_1 - b|| / ||b||

    while (norm >= eps) {
        sum = 0;
        matrMul(A, x, xnew);

        for (int i = 0; i < N; i++)
            xnew[i] = xnew[i] - b[i]; //(Ax^n - b)

        //  ||Ax_1 - b|| / ||b||
        for (int i = 0; i < N; i++)
            sum += xnew[i] * xnew[i];
        norm = sqrt(sum) / norm_b;

        for (int i = 0; i < N; i++)
            xnew[i] = -tau * xnew[i]; // -tau * (Ax^n - b)

        for (int i = 0; i < N; i++)
            xnew[i] = x[i] + xnew[i]; // x^n - tau * (Ax^n - b)

        for (int i = 0; i < N; i++)
            x[i] = xnew[i]; // x^(n+1) = x^n - tau * (Ax^n - b)
    }
    clock_t endTime = clock();

    cout << "Res: " << "\n";
    for (int i = 0; i < N; i++) {
        cout << "x[" << i << "] = " << xnew[i] << "\n";
    }
    cout << "Time: "<< (double)(endTime - startTime) / CLOCKS_PER_SEC << endl;

    free(x);
    free(xnew);
    free(A);
    free(b);
    return 0;
}

