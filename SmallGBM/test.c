#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tree.h"

#define N 100000
#define M 20

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    srand(42);

    double *X = malloc((size_t)N * M * sizeof(double));
    double *y = malloc(N * sizeof(double));
    if (!X || !y) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = 0; j < M; j++) {
            double v = (double)rand() / RAND_MAX;
            X[i * M + j] = v;
            s += v * (j + 1);
        }
        y[i] = s + 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }

    Tree *t = tree_create(4, 20, 1.0, 255, 0.1);
    if (!t) { fprintf(stderr, "tree_create failed\n"); return 1; }

    double t0 = now_sec();
    tree_fit(t, X, y, N, M);
    double t1 = now_sec();

    Predictions *p = tree_predict(t, X, N, M);
    double t2 = now_sec();

    if (p) {
        printf("C: fit = %.3f s, predict = %.3f s\n", t1 - t0, t2 - t1);
        printf("C: first 5 preds = %.4f %.4f %.4f %.4f %.4f\n",
               p->preds[0], p->preds[1], p->preds[2], p->preds[3], p->preds[4]);
        predictions_free(p);
    }

    FILE *f = fopen("data.csv", "w");
    if (!f) { perror("fopen"); return 1; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            fprintf(f, "%.17g,", X[i * M + j]);
        fprintf(f, "%.17g\n", y[i]);
    }
    fclose(f);

    tree_free(t);
    free(X);
    free(y);
    return 0;
}