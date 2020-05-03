from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
from inspect import signature

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class oodMetrics():
    def __init__(self):
        #dictionary to collect metric names
        self.metrics={}

    def resetMetrics():
        self.metrics={}

    def compute_all_metrics(self,y_gt,y_pred,choose_thresholds=True, fixed_threshold=None):
        precision, recall, pr_thresholds = precision_recall_curve(y_gt, y_pred)
        average_precision = average_precision_score(y_gt, y_pred)
        f1_scores = [2*x*y/(x+y) for x,y in zip(precision,recall)]

        precision_reverse, recall_reverse, pr_reverse_thresholds =  precision_recall_curve(1-y_gt, -y_pred)

        fpr, tpr, thresholds = roc_curve(y_gt, y_pred)
        auc_area = auc(fpr,tpr)
        aupr_in_area = auc(recall,precision)
        aupr_out_area = auc(recall_reverse, precision_reverse)

        precision_at_half, recall_at_half, f1_at_half = 0,0,0
        min_distance_to_half = 1
        for thIndex,threshold in enumerate(pr_thresholds):
            if abs(threshold-0.5)<=min_distance_to_half:
                min_distance_to_half = abs(threshold-0.5)
                precision_at_half = precision[thIndex]
                recall_at_half = recall[thIndex]
                f1_at_half = f1_scores[thIndex]
        print(pr_thresholds)
        self.metrics["PR@0.5"] = precision_at_half
        self.metrics["REC@0.5"] = recall_at_half
        self.metrics["F1@0.5"] = f1_at_half

        if choose_thresholds:
            fpr_at_95_tpr, fpr_at_80_tpr = 1.0, 1.0
            fpr_at_95_tpr_thresh, fpr_at_80_tpr_thresh = max(thresholds), max(thresholds)
            for f,t,th in zip(fpr,tpr,thresholds):
                if t>=0.95:
                    fpr_at_95_tpr = f
                    fpr_at_95_tpr_thresh = th
                    break
            for f,t,th in zip(fpr,tpr,thresholds):
                if t>=0.80:
                    fpr_at_80_tpr = f
                    fpr_at_80_tpr_thresh = th
                    break
            recall_at_95_pr, recall_at_80_pr = 0.0 , 0.0
            recall_at_95_pr_thresh, recall_at_95_pr_thresh = 1.0 , 1.0
            for r, p , prth in zip(recall,precision,pr_thresholds):
                if p>=0.95:
                    recall_at_95_pr = r
                    recall_at_95_pr_thresh = prth
                    break
            for r, p, prth in zip(recall,precision,pr_thresholds):
                if p>=0.80:
                    recall_at_80_pr = r
                    recall_at_80_pr_thresh = prth
                    break

        elif fixed_threshold!=None:
            fpr_at_95_tpr, fpr_at_80_tpr = 1.0, 1.0
            fpr_at_95_tpr_thresh, fpr_at_80_tpr_thresh = max(thresholds), max(thresholds)
            for f,t,th in zip(fpr,tpr,thresholds):
                if th==fixed_threshold:
                    fpr_at_95_tpr = f
                    fpr_at_95_tpr_thresh = th
                    break
            for f,t,th in zip(fpr,tpr,thresholds):
                if th==fixed_threshold:
                    fpr_at_80_tpr = f
                    fpr_at_80_tpr_thresh = th
                    break
            recall_at_95_pr, recall_at_80_pr = 0.0 , 0.0
            recall_at_95_pr_thresh, recall_at_95_pr_thresh = 1.0 , 1.0
            for r, p , prth in zip(recall,precision,pr_thresholds):
                if prth==fixed_threshold:
                    recall_at_95_pr = r
                    recall_at_95_pr_thresh = prth
                    break
            for r, p, prth in zip(recall,precision,pr_thresholds):
                if prth==fixed_threshold:
                    recall_at_80_pr = r
                    recall_at_80_pr_thresh = prth
                    break


        self.metrics["recall"] = recall
        self.metrics["precision"] = precision
        self.metrics["pr_thresholds"] = pr_thresholds
        self.metrics["average_precision"] = average_precision
        self.metrics["recall_reverse"] = recall_reverse
        self.metrics["precision_reverse"] = precision_reverse
        self.metrics["pr_thresholds_reverse"] = pr_reverse_thresholds
        self.metrics["f1_scores"] = f1_scores
        self.metrics["fpr"] = fpr
        self.metrics["tpr"] = tpr
        self.metrics["thresholds"] = thresholds

        self.metrics["FPR@80%TPR"] = fpr_at_80_tpr
        self.metrics["FPR@95%TPR"] = fpr_at_95_tpr
        self.metrics["REC@80%PR"] = recall_at_80_pr
        self.metrics["REC@95%PR"] = recall_at_95_pr
        self.metrics["AUROC"] = auc_area
        self.metrics["AUPRIn"] = aupr_in_area
        self.metrics["AUPROut"] = aupr_out_area
        self.metrics["F1 Scores OOD Threshold"]=max(f1_scores)


    def plot_PR_curve(self,plotName):
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(self.metrics["recall"], self.metrics["precision"], color='b', alpha=0.2, where='post')
        plt.fill_between(self.metrics["recall"], self.metrics["precision"], alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0,1.05])
        plt.xlim([0.0,1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(self.metrics["average_precision"]))
        plt.savefig(plotName)


    def pretty_print_metrics(self):
        print("-----OOD.METRICS.SUMMARY.BEGIN-----")
        keyList = ["PR@0.5","REC@0.5","F1@0.5","FPR@80%TPR","FPR@95%TPR","REC@80%PR","REC@95%PR","AUROC","AUPRIn","AUPROut","F1 Scores OOD Threshold"]
        for key in keyList:
            print(key,":",self.metrics[key])
        print("-----OOD.METRICS.SUMMARY.END-----")

