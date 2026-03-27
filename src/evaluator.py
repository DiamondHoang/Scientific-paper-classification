import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from . import config


def plot_confusion_matrix(y_true, y_pred, label_list, figure_name, save_path):
    """Plot confusion matrix with counts and percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = np.unique(y_true)
    class_names = [label_list[i] for i in labels]
    
    # Create annotations
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            raw = cm[i, j]
            norm = cm_normalized[i, j]
            annotations[i, j] = f"{raw}\n({norm:.2%})"
    
    # Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(figure_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_classification_report(report, model_name, vec_name):
    """Save classification report to JSON and TXT files"""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(
        config.REPORTS_DIR,
        f"{model_name}_{vec_name}_report.json"
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    
    # Save as readable TXT
    txt_path = os.path.join(
        config.REPORTS_DIR,
        f"{model_name}_{vec_name}_report.txt"
    )
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report - {model_name.upper()} ({vec_name.upper()})\n")
        f.write("="*70 + "\n\n")
        
        # Per-class metrics
        f.write(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        
        for label, metrics in report.items():
            if label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                f.write(f"{label:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                       f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}\n")
        
        f.write("\n" + "-"*70 + "\n")
        
        # Overall metrics
        f.write(f"\nAccuracy: {report['accuracy']:.4f}\n\n")
        
        if 'macro avg' in report:
            f.write("Macro Average:\n")
            f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall:    {report['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}\n\n")
        
        if 'weighted avg' in report:
            f.write("Weighted Average:\n")
            f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")


def print_classification_report(report, model_name, vec_name):
    """Print classification report to console"""
    print(f"\n{model_name.upper()} - {vec_name.upper()}:")
    print("-"*60)
    print(f"{'Category':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*60)
    
    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        if isinstance(metrics, dict):
            print(f"{label:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1-score']:<10.4f}")
    
    print(f"\nAccuracy: {report['accuracy']:.4f}")


def visualize_results(results, y_test, id_to_label):
    """Generate all confusion matrices"""
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    sorted_labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    for model_name, vec_results in results.items():
        print(f"\n{model_name.upper()}:")
        for vec_name, data in vec_results.items():
            y_pred = data['predictions']
            figure_name = f"{model_name.upper()} - {vec_name.upper()}"
            save_path = os.path.join(
                config.FIGURES_DIR,
                f"{model_name}_{vec_name}_confusion_matrix.pdf"
            )
            
            plot_confusion_matrix(y_test, y_pred, sorted_labels, figure_name, save_path)
            print(f"  Saved: {save_path}")


def save_all_reports(results):
    """Save all classification reports"""
    print("\n" + "="*60)
    print("Saving Classification Reports")
    print("="*60)
    
    for model_name, vec_results in results.items():
        print(f"\n{model_name.upper()}:")
        for vec_name, data in vec_results.items():
            report = data['report']
            save_classification_report(report, model_name, vec_name)
            print(f"  Saved: {model_name}_{vec_name}_report.txt/json")


def print_all_reports(results):
    """Print all classification reports to console"""
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    for model_name, vec_results in results.items():
        for vec_name, data in vec_results.items():
            report = data['report']
            print_classification_report(report, model_name, vec_name)


def print_summary(results):
    """Print accuracy summary table"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'BoW':<10} {'TF-IDF':<10} {'Embeddings':<10}")
    print("-" * 50)
    
    for model_name, vec_results in results.items():
        bow_acc = vec_results['bow']['accuracy']
        tfidf_acc = vec_results['tfidf']['accuracy']
        emb_acc = vec_results['embeddings']['accuracy']
        
        print(f"{model_name:<15} {bow_acc:<10.4f} {tfidf_acc:<10.4f} {emb_acc:<10.4f}")
    
    print("\n" + "="*60)