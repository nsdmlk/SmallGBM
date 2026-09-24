#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "tree.h"

#define NBINS 512

typedef struct {
    uint8_t *Xb;
    double  *edges;
    int      n, m;
} BinnedData;

typedef struct {
    double  *residuals;
    double  *scratch;
    int     *left_idx;
    int     *right_idx;
    int     *hist_cnt;
    double  *hist_sum;
    int     *cum_cnt;
    double  *cum_sum;
    int      capacity;
} Scratch;

typedef struct { double val; int idx; } SortPair;

static Scratch *scratch_create(int cap) {
    Scratch *s = malloc(sizeof(Scratch));
    if (!s) return NULL;
    s->residuals = malloc((size_t)cap * sizeof(double));
    s->scratch   = malloc((size_t)cap * sizeof(double));
    s->left_idx  = malloc((size_t)cap * sizeof(int));
    s->right_idx = malloc((size_t)cap * sizeof(int));
    s->hist_cnt  = malloc((size_t)NBINS * sizeof(int));
    s->hist_sum  = malloc((size_t)NBINS * sizeof(double));
    s->cum_cnt   = malloc((size_t)NBINS * sizeof(int));
    s->cum_sum   = malloc((size_t)NBINS * sizeof(double));
    if (!s->residuals || !s->scratch || !s->left_idx || !s->right_idx ||
        !s->hist_cnt || !s->hist_sum || !s->cum_cnt || !s->cum_sum) {
        free(s->residuals); free(s->scratch);
        free(s->left_idx); free(s->right_idx);
        free(s->hist_cnt); free(s->hist_sum);
        free(s->cum_cnt); free(s->cum_sum);
        free(s); return NULL;
    }
    s->capacity = cap;
    return s;
}

static void scratch_free(Scratch *s) {
    if (!s) return;
    free(s->residuals); free(s->scratch);
    free(s->left_idx); free(s->right_idx);
    free(s->hist_cnt); free(s->hist_sum);
    free(s->cum_cnt); free(s->cum_sum);
    free(s);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int cmp_sortpair(const void *a, const void *b) {
    double x = ((const SortPair *)a)->val, y = ((const SortPair *)b)->val;
    return (x > y) - (x < y);
}

static double median_into(const double *arr, int n, double *scratch) {
    if (n == 0) return 0.0;
    memcpy(scratch, arr, (size_t)n * sizeof(double));
    qsort(scratch, n, sizeof(double), cmp_double);
    return (n % 2 != 0) ? scratch[n / 2]
                        : (scratch[n / 2 - 1] + scratch[n / 2]) / 2.0;
}

static double mad_into(const double *arr, int n, double *scratch) {
    double med = median_into(arr, n, scratch);
    for (int i = 0; i < n; i++) scratch[i] = fabs(arr[i] - med);
    return median_into(scratch, n, scratch);
}

static RobustWeight robust_weight(const double *residuals, int n,
                                  double parent_mu, double sigma_prior,
                                  double *scratch) {
    RobustWeight weight = { .mu = parent_mu, .std = 0.0 };
    if (n == 0) return weight;
    double center, sigma_noise;
    const int ROBUST_THRESHOLD = 30;
    if (n <= ROBUST_THRESHOLD) {
        center = median_into(residuals, n, scratch);
        double m = mad_into(residuals, n, scratch);
        sigma_noise = (m > 0.0) ? 1.4826 * m : sigma_prior;
    } else {
        double med = median_into(residuals, n, scratch);
        double wsum = 0.0;
        for (int i = 0; i < n; i++) {
            double w = 1.0 / (fabs(residuals[i] - med) + sigma_prior);
            scratch[i] = w; wsum += w;
        }
        for (int i = 0; i < n; i++) scratch[i] /= wsum;
        center = 0.0;
        for (int i = 0; i < n; i++) center += scratch[i] * residuals[i];
        double var = 0.0;
        for (int i = 0; i < n; i++) {
            double d = residuals[i] - center;
            var += scratch[i] * d * d;
        }
        sigma_noise = sqrt(var);
    }
    const double K = 1.5;
    double ratio = (sigma_noise * sigma_noise) / (n * sigma_prior * sigma_prior);
    double shrinkage = 1.0 / (1.0 + pow(ratio, K));
    weight.mu  = shrinkage * center + (1.0 - shrinkage) * parent_mu;
    weight.std = sigma_noise / sqrt((double)n);
    return weight;
}

static BinnedData *binned_create(const double *X, int n, int m) {
    BinnedData *bd = malloc(sizeof(BinnedData));
    if (!bd) return NULL;
    bd->n = n; bd->m = m;
    bd->Xb    = malloc((size_t)n * m * sizeof(uint8_t));
    bd->edges = malloc((size_t)m * (NBINS + 1) * sizeof(double));
    if (!bd->Xb || !bd->edges) { free(bd->Xb); free(bd->edges); free(bd); return NULL; }

    SortPair *col = malloc((size_t)n * sizeof(SortPair));
    if (!col) { free(bd->Xb); free(bd->edges); free(bd); return NULL; }

    for (int f = 0; f < m; f++) {
        for (int i = 0; i < n; i++) { col[i].val = X[(size_t)i*m+f]; col[i].idx = i; }
        qsort(col, n, sizeof(SortPair), cmp_sortpair);

        double *edges = &bd->edges[(size_t)f * (NBINS + 1)];
        for (int b = 0; b <= NBINS; b++) {
            int pos = (int)((double)b * (n - 1) / NBINS);
            edges[b] = col[pos].val;
        }
        for (int i = 0; i < n; i++) {
            double v = X[(size_t)i*m+f];
            int lo = 0, hi = NBINS;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (edges[mid + 1] <= v) lo = mid + 1;
                else hi = mid;
            }
            bd->Xb[(size_t)i*m+f] = (uint8_t)(lo > 255 ? 255 : lo);
            /* NBINS=512 требует uint16_t! */
        }
    }
    free(col);
    return bd;
}

static void binned_free(BinnedData *bd) {
    if (!bd) return;
    free(bd->Xb); free(bd->edges); free(bd);
}

static BestSplit best_split(const BinnedData *bd, const double *y,
                            const int *idx, int n, const Tree *t, Scratch *s) {
    BestSplit best = { .best_gain = -INFINITY, .best_threshold = 0.0,
                       .best_feature = -1, .best_bin = -1 };

    double total_sum = 0.0;
    for (int i = 0; i < n; i++) total_sum += y[idx[i]];

    double lambda = t->sigma_prior * t->sigma_prior;
    double parent_score = (total_sum * total_sum) / (n + lambda);

    int m = bd->m;
    int n_cols = m;
    if (t->colsample_bytree < 1.0 && m > 1) {
        n_cols = (int)(m * t->colsample_bytree);
        if (n_cols < 1) n_cols = 1;
    }

    int    *hist_cnt = s->hist_cnt;
    double *hist_sum = s->hist_sum;
    int    *cum_cnt  = s->cum_cnt;
    double *cum_sum  = s->cum_sum;

    int n_splits = t->n_splits;
    if (n_splits < 1) n_splits = 1;

    for (int f = 0; f < n_cols; f++) {
        memset(hist_cnt, 0, (size_t)NBINS * sizeof(int));
        memset(hist_sum, 0, (size_t)NBINS * sizeof(double));

        for (int i = 0; i < n; i++) {
            int k = idx[i];
            uint16_t b = (uint16_t)bd->Xb[(size_t)k*m+f];
            hist_cnt[b]++;
            hist_sum[b] += y[k];
        }

        double ssum = 0.0; int scnt = 0;
        for (int b = 0; b < NBINS; b++) {
            ssum += hist_sum[b]; scnt += hist_cnt[b];
            cum_sum[b] = ssum; cum_cnt[b] = scnt;
        }

        int b_lo = 0, b_hi = NBINS - 2;
        while (b_lo <= b_hi && cum_cnt[b_lo] < t->min_samples_leaf) b_lo++;
        while (b_hi >= b_lo && (n - cum_cnt[b_hi]) < t->min_samples_leaf) b_hi--;
        if (b_lo > b_hi) continue;

        int range = b_hi - b_lo + 1;
        int tries = (n_splits < range) ? n_splits : range;

        for (int sp = 0; sp < tries; sp++) {
            int b = b_lo + rand() % range;
            int left_cnt  = cum_cnt[b];
            int right_cnt = n - left_cnt;
            if (left_cnt  < t->min_samples_leaf) continue;
            if (right_cnt < t->min_samples_leaf) continue;

            double left_sum  = cum_sum[b];
            double right_sum = total_sum - left_sum;

            double left_score  = (left_sum  * left_sum)  / (left_cnt  + lambda);
            double right_score = (right_sum * right_sum) / (right_cnt + lambda);
            double gain = left_score + right_score - parent_score;

            if (gain > best.best_gain) {
                best.best_gain    = gain;
                best.best_feature = f;
                best.best_bin     = b;
                double e0 = bd->edges[(size_t)f * (NBINS + 1) + b];
                double e1 = bd->edges[(size_t)f * (NBINS + 1) + b + 1];
                best.best_threshold = (e0 + e1) / 2.0;
            }
        }
    }
    return best;
}

static Node *make_leaf(const double *y, const int *idx, int n,
                       double parent_mu, double sigma_prior, Scratch *s) {
    for (int i = 0; i < n; i++) s->residuals[i] = y[idx[i]];
    RobustWeight w = robust_weight(s->residuals, n, parent_mu, sigma_prior, s->scratch);
    Node *node = malloc(sizeof(Node));
    if (!node) return NULL;
    node->type = LEAF; node->weight = w.mu; node->uncertainty = w.std;
    node->n_samples = n; node->feature = -1; node->threshold = 0.0;
    node->left = node->right = NULL;
    return node;
}

static Node *build_tree(const Tree *t, const BinnedData *bd, const double *y,
                        const int *idx, int n, int depth,
                        double parent_mu, Scratch *s) {
    int stop = (depth >= t->max_depth) || (n < t->min_samples_leaf * 2);
    if (!stop) {
        int same = 1;
        for (int i = 1; i < n; i++) if (y[idx[i]] != y[idx[0]]) { same = 0; break; }
        if (same) stop = 1;
    }
    if (stop) return make_leaf(y, idx, n, parent_mu, t->sigma_prior, s);

    BestSplit split = best_split(bd, y, idx, n, t, s);
    if (split.best_feature == -1 || split.best_gain <= 0.0)
        return make_leaf(y, idx, n, parent_mu, t->sigma_prior, s);

    for (int i = 0; i < n; i++) s->residuals[i] = y[idx[i]];
    RobustWeight node_w = robust_weight(s->residuals, n, parent_mu,
                                        t->sigma_prior, s->scratch);

    int *li = s->left_idx, *ri = s->right_idx;
    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
        int k = idx[i];
        uint16_t b = (uint16_t)bd->Xb[(size_t)k*m + split.best_feature];
        if ((int)b <= split.best_bin) li[nl++] = k; else ri[nr++] = k;
    }

    int *left_copy  = malloc((size_t)nl * sizeof(int));
    int *right_copy = malloc((size_t)nr * sizeof(int));
    if (!left_copy || !right_copy) {
        free(left_copy); free(right_copy);
        return make_leaf(y, idx, n, parent_mu, t->sigma_prior, s);
    }
    memcpy(left_copy, li, (size_t)nl * sizeof(int));
    memcpy(right_copy, ri, (size_t)nr * sizeof(int));

    Node *node = malloc(sizeof(Node));
    if (!node) { free(left_copy); free(right_copy);
                 return make_leaf(y, idx, n, parent_mu, t->sigma_prior, s); }
    node->type = NODE; node->feature = split.best_feature;
    node->threshold = split.best_threshold;
    node->weight = node_w.mu; node->uncertainty = node_w.std;
    node->n_samples = n;

    node->left  = build_tree(t, bd, y, left_copy,  nl, depth + 1, node_w.mu, s);
    node->right = build_tree(t, bd, y, right_copy, nr, depth + 1, node_w.mu, s);
    free(left_copy); free(right_copy);
    return node;
}

static void free_node(Node *node) {
    if (!node) return;
    free_node(node->left); free_node(node->right);
    free(node);
}

Tree *tree_create(int max_depth, int min_samples_leaf, double sigma_prior,
                  int n_splits, double colsample_bytree) {
    Tree *t = malloc(sizeof(Tree));
    if (!t) return NULL;
    t->max_depth = max_depth; t->min_samples_leaf = min_samples_leaf;
    t->sigma_prior = sigma_prior; t->n_splits = n_splits;
    t->colsample_bytree = colsample_bytree; t->tree = NULL;
    return t;
}

void tree_free(Tree *t) {
    if (!t) return;
    free_node(t->tree); free(t);
}

void tree_fit(Tree *t, const double *X, const double *y, int n, int m) {
    if (!t || !X || !y || n <= 0 || m <= 0) return;
    if (t->tree) { free_node(t->tree); t->tree = NULL; }
    Scratch *s = scratch_create(n);
    if (!s) return;
    BinnedData *bd = binned_create(X, n, m);
    if (!bd) { scratch_free(s); return; }
    int *idx = malloc((size_t)n * sizeof(int));
    if (!idx) { binned_free(bd); scratch_free(s); return; }
    for (int i = 0; i < n; i++) idx[i] = i;
    t->tree = build_tree(t, bd, y, idx, n, 0, 0.0, s);
    free(idx); binned_free(bd); scratch_free(s);
}

static double predict_one(const Node *node, const double *x, double *uncertainty) {
    while (node->type != LEAF)
        node = (x[node->feature] <= node->threshold) ? node->left : node->right;
    if (uncertainty) *uncertainty = node->uncertainty;
    return node->weight;
}

Predictions *tree_predict(const Tree *t, const double *X, int n, int m) {
    if (!t || !t->tree || !X || n <= 0) return NULL;
    Predictions *p = malloc(sizeof(Predictions));
    if (!p) return NULL;
    p->preds         = malloc((size_t)n * sizeof(double));
    p->uncertainties = malloc((size_t)n * sizeof(double));
    if (!p->preds || !p->uncertainties) {
        free(p->preds); free(p->uncertainties); free(p);
        return NULL;
    }
    p->n = n;
    for (int i = 0; i < n; i++)
        p->preds[i] = predict_one(t->tree, &X[(size_t)i*m], &p->uncertainties[i]);
    return p;
}

void predictions_free(Predictions *p) {
    if (!p) return;
    free(p->preds); free(p->uncertainties); free(p);
}