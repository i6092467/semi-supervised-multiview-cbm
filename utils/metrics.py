"""
Utility functions for model evaluation metrics
"""
import os
from pathlib import Path
from shutil import rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, balanced_accuracy_score, det_curve)

matplotlib.use("Agg")


class ROC:
    """
    A class for visualising ROC curves over several data folds
    """

    def __init__(self, folds, epochs):
        # Store all information and figures in dictionaries
        self.folds = folds
        self.mean_fpr = np.linspace(0, 1, 100)
        self.tprs = {}
        self.aucs = {}
        self.axes = {}
        self.figs = {}

        for epoch in epochs:
            fig, ax = plt.subplots()
            self.figs[epoch] = fig
            self.axes[epoch] = ax
            self.tprs[epoch] = []
            self.aucs[epoch] = []

    def update(self, epoch, fold, y_true, y_pred):
        """
        Compute ROC and AUROC and store the results globally
        """
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pred,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=self.axes[epoch]
        )
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.tprs[epoch].append(interp_tpr)
        self.aucs[epoch].append(viz.roc_auc)

    def save(self, roc_dir):
        """
        Generate ROC figures and save them
        """
        print("Writing ROC curve files")
        if roc_dir.exists() and roc_dir.is_dir():
            rmtree(roc_dir)
        try:
            os.mkdir(roc_dir)
        except OSError:
            print("Could not create ROC directory...")
            quit()

        f = open(Path(roc_dir, "summary.csv"), "w")
        f.write("Epoch,AUROC, std\n")

        for epoch in self.tprs.keys():
            # Plot mean true positive rate and area under curve
            mean_tpr = np.mean(self.tprs[epoch], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(self.mean_fpr, mean_tpr)
            std_auc = np.std(self.aucs[epoch])
            self.axes[epoch].plot(
                self.mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )
            f.write("{},{},{}\n".format(epoch, mean_auc, std_auc))

            # Plot standard deviation
            std_tpr = np.std(self.tprs[epoch], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            self.axes[epoch].fill_between(
                self.mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            # Plot diagonal
            self.axes[epoch].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)

            self.axes[epoch].set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title=f"ROC {self.folds}-fold for epoch {epoch}",
            )

            # Plot legend
            self.axes[epoch].legend(loc="lower right")
            self.figs[epoch].savefig(os.path.join(roc_dir, f"ROC_epoch{epoch}.png"))


class PRC:
    """
    A class for visualising precision-recall curves (PRC) over several data folds
    """

    def __init__(self, folds, epochs):
        # Store all information and figures in dictionaries
        self.folds = folds
        self.mean_recall = np.linspace(0, 1, 100)
        self.prec = {}
        self.auc = {}
        self.axes = {}
        self.figs = {}

        for epoch in epochs:
            fig, ax = plt.subplots()
            self.figs[epoch] = fig
            self.axes[epoch] = ax
            self.prec[epoch] = []
            self.auc[epoch] = []

    def update(self, epoch, fold, y_true, y_pred):
        """
        Compute PRC and AUPRC and store the results globally
        """
        viz = PrecisionRecallDisplay.from_predictions(
            y_true,
            y_pred,
            name=f"PR fold {fold}",
            alpha=0.3,
            lw=1,
            ax=self.axes[epoch]
        )
        interp_prec = np.interp(self.mean_recall, viz.recall[::-1], viz.precision[::-1])
        interp_prec[0] = 1.0
        self.prec[epoch].append(interp_prec)
        self.auc[epoch].append(auc(viz.recall, viz.precision))

    def save(self, prc_dir):
        """
        Generate PRC figures and save them
        """
        print("Writing PRC curve files")
        if prc_dir.exists() and prc_dir.is_dir():
            rmtree(prc_dir)
        try:
            os.mkdir(prc_dir)
        except OSError:
            print("Could not create PRC directory...")
            quit()

        f = open(Path(prc_dir, "summary.csv"), "w")
        f.write("Epoch,AUPR, std\n")

        for epoch in self.prec.keys():
            # Plot mean true positive rate and area under curve
            mean_prec = np.mean(self.prec[epoch], axis=0)
            mean_avg_p = np.mean(self.auc[epoch])
            std_auc = np.std(self.auc[epoch])
            self.axes[epoch].plot(
                self.mean_recall,
                mean_prec,
                color="b",
                label=r"Mean PRC (AUPR = %0.2f $\pm$ %0.2f)" % (mean_avg_p, std_auc),
                lw=2,
                alpha=0.8,
            )

            f.write("{},{},{}\n".format(epoch, mean_avg_p, std_auc))

            # Plot standard deviation
            std_tpr = np.std(self.prec[epoch], axis=0)
            tprs_upper = np.minimum(mean_prec + std_tpr, 1)
            tprs_lower = np.maximum(mean_prec - std_tpr, 0)
            self.axes[epoch].fill_between(
                self.mean_recall,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            # Plot the diagonal
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = self.axes[epoch].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                self.axes[epoch].annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

            self.axes[epoch].set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title=f"PRC {self.folds}-fold for epoch {epoch}",
            )

            # Plot legend
            self.axes[epoch].legend(loc="lower left")
            self.figs[epoch].savefig(os.path.join(prc_dir, f"PRC_epoch{epoch}.png"))

        f.close()


class TMetrics:
    """
    Metrics of interest for evaluating target variable predictions
    """

    def __init__(self, ppv, npv, sensitivity, specificity, accuracy, balanced_accuracy, f1_1, f1_0, f1_macro, auroc, aupr,
                 fpr_at_k=None, brier_score=None):
        self.ppv = ppv
        self.npv = npv
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.accuracy = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.f1_1 = f1_1
        self.f1_0 = f1_0
        self.f1_macro = f1_macro
        self.auroc = auroc
        self.aupr = aupr
        if fpr_at_k is not None:
            self.fpr_at_k = fpr_at_k
        if brier_score is not None:
            self.brier_score = brier_score

    def get_tMetrics(self):
        tmp = [self.ppv, self.npv, self.sensitivity, self.specificity, self.accuracy, self.balanced_accuracy, self.f1_1,
               self.f1_0, self.f1_macro, self.auroc, self.aupr]
        if hasattr(self, 'fpr_at_k'):
            tmp.append(self.fpr_at_k)
        if hasattr(self, 'brier_score'):
            tmp.append(self.brier_score)
        metrics_array = np.array(tmp)
        return metrics_array


def calc_tMetrics(y_true, y_score):
    """
    Computes metrics for evaluating target variable predictions
    """
    y_true = y_true.cpu().numpy()
    n_classes = len(np.unique(y_true))
    y_score = y_score.detach().cpu().numpy()
    if n_classes == 2:
        y_pred = np.where(y_score > 0.5, 1, 0)
    else:
        y_pred = np.argmax(y_score, 1)

    accuracy = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        ppv = precision_score(y_true, y_pred, average=None, zero_division=0)[1]
        npv = precision_score(y_true, y_pred, average=None, zero_division=0)[0]

        sensitivity = recall_score(y_true, y_pred, average=None, zero_division=0)[1]
        specificity = recall_score(y_true, y_pred, average=None, zero_division=0)[0]

        f1_1 = f1_score(y_true, y_pred, average=None, zero_division=0)[1]
        f1_0 = f1_score(y_true, y_pred, average=None, zero_division=0)[0]
        f1_macro = (f1_1 + f1_0) / 2

        balanced_accuracy = (sensitivity + specificity) / 2

        auroc = roc_auc_score(y_true, y_score)

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        aupr = auc(recall, precision)

        fpr, fnr, _ = det_curve(y_true, y_score)
        tpr = 1 - fnr
        fpr_at_k = {}
        ks = [0.75, 0.80, 0.90, 0.95, 0.99]
        for k in ks:
            ind = np.argmin(np.abs(tpr - k))
            fpr_at_k['FPR at ' + str(k)] = np.round(fpr[ind], 3)

        brier_score = calc_brier_score(y_true=y_true, y_prob=y_score)

    else:
        ppv = precision_score(y_true, y_pred, average='macro', zero_division=0)
        npv = precision_score(y_true, y_pred, average='macro', zero_division=0)

        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = recall_score(y_true, y_pred, average='macro', zero_division=0)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_1 = -1.0
        f1_0 = -1.0

        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')

        # TODO: implement multiclass versions of the metrics below
        aupr = -1.0

        fpr_at_k = None

        brier_score = None

    return TMetrics(ppv, npv, sensitivity, specificity, accuracy, balanced_accuracy, f1_1, f1_0, f1_macro, auroc, aupr,
                    fpr_at_k, brier_score)


class CMetrics:
    """
    Metrics of interest for evaluating concept predictions
    """

    def __init__(self, accuracy, f1_macro, auroc, aupr, concept_name, brier_score=None):
        self.accuracy = accuracy
        self.f1_macro = f1_macro
        self.auroc = auroc
        self.aupr = aupr
        self.concept_name = concept_name
        if brier_score is not None:
            self.brier_score = brier_score

    def get_cMetrics(self):
        tmp = [self.accuracy, self.f1_macro, self.auroc, self.aupr]
        if hasattr(self, 'brier_score'):
            tmp.append(self.brier_score)
        cmetrics_array = np.array(tmp)
        return cmetrics_array


def calc_cMetrics(y_true, y_score, concept_name):
    """
    Calculates metrics for a single binary concept
    """
    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    y_pred = np.where(y_score > 0.5, 1, 0)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    brier_score = calc_brier_score(y_true=y_true, y_prob=y_score)
    return CMetrics(accuracy, f1_macro, auroc, aupr, concept_name, brier_score)


def calc_confusion(y_true, y_score, image_file_names):
    """
    Constructs the confusion matrix
    """
    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        y_pred = np.where(y_score > 0.5, 1, 0)
    else:
        y_pred = np.argmax(y_score, 1)

    conf_matrix = confusion_matrix(y_true, y_pred)

    if n_classes == 2:
        file_names = np.array(image_file_names, dtype=object)
        FP = np.intersect1d(np.argwhere(y_pred == 1).flatten(), np.argwhere(y_true == 0).flatten())
        FN = np.intersect1d(np.argwhere(y_pred == 0).flatten(), np.argwhere(y_true == 1).flatten())
        tn, fp, fn, tp = conf_matrix.ravel()
        assert len(FP) == fp
        assert len(FN) == fn
        FP_names = file_names[FP]
        FN_names = file_names[FN]
    else:
        FP_names = np.array([])
        FN_names = np.array([])

    return conf_matrix, FP_names, FN_names


def calc_brier_score(y_true, y_prob, rescale=False):
    if rescale:
        y_prob_ = (y_prob - np.min(y_prob)) / (np.max(y_prob) - np.min(y_prob))
    else:
        y_prob_ = y_prob
    # NOTE: assumes a binary classification task
    return np.mean((y_prob_ - y_true)**2)
