#ifndef TREE_H
#define TREE_H

typedef enum { LEAF, NODE } NodeType;

typedef struct Node {
    NodeType type;
    int feature;
    double threshold;
    double weight;
    double uncertainty;
    int n_samples;
    struct Node *left;
    struct Node *right;
} Node;

typedef struct {
    double mu;
    double std;
} RobustWeight;

typedef struct {
    double best_gain;
    double best_threshold;
    int    best_feature;
    int    best_bin;
} BestSplit;

typedef struct {
    double *preds;
    double *uncertainties;
    int n;
} Predictions;

typedef struct {
    int max_depth;
    int min_samples_leaf;
    double sigma_prior;
    int n_splits;
    double colsample_bytree;
    Node *tree;
} Tree;

Tree *tree_create(int max_depth, int min_samples_leaf, double sigma_prior,
                  int n_splits, double colsample_bytree);
void tree_free(Tree *t);
void tree_fit(Tree *t, const double *X, const double *residuals, int n, int m);
Predictions *tree_predict(const Tree *t, const double *X, int n, int m);
void predictions_free(Predictions *p);

#endif