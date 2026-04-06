"""
Phase 6 (Updated): Improved Phenotype Naming
Standard Monitoring vs Hemodynamic Monitoring vs Advanced Multimodal
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def define_phenotypes_improved(df: pd.DataFrame) -> pd.Series:
    """Define monitoring phenotypes with improved naming.
    
    Categories:
    - Standard Monitoring: ECG + SpO2 + RESP (basic physiological)
    - Hemodynamic Monitoring: Standard + invasive ABP (cardiovascular focus)
    - Advanced Hemodynamic: Hemodynamic + CVP/PAP (full hemodynamic profile)
    - Neurological Monitoring: Advanced + ICP/BIS (neuro-critical care)
    - Ventilated Monitoring: Any level + CO2/ventilator (mechanical ventilation)
    - Advanced Multimodal: Multiple advanced systems (comprehensive ICU)
    """
    phenotypes = []
    
    for _, row in df.iterrows():
        # Check presence of key signals
        has_ecg = row.get('has_ecg', False)
        has_ppg = row.get('has_ppg', False)
        has_resp = row.get('has_resp', False)
        has_abp = row.get('has_abp_invasive', False)
        has_cvp = row.get('has_cvp', False)
        has_pap = row.get('has_pap', False)
        has_icp = row.get('has_icp', False)
        has_bis = row.get('has_bis_eeg', False)
        has_co2 = row.get('has_co2', False)
        has_vent = row.get('has_ventilation', False)
        
        # Count total signal categories
        total_signals = sum([has_ecg, has_ppg, has_resp, has_abp, has_cvp, 
                           has_pap, has_icp, has_bis, has_co2, has_vent])
        
        # Define phenotypes (priority order matters)
        if has_icp and has_abp:
            phenotype = 'Neurological_Monitoring'
        elif has_cvp and has_pap and has_abp:
            phenotype = 'Advanced_Hemodynamic'
        elif has_abp and has_cvp:
            phenotype = 'Advanced_Hemodynamic'
        elif has_abp and (has_co2 or has_vent):
            phenotype = 'Ventilated_Hemodynamic'
        elif has_abp:
            phenotype = 'Hemodynamic_Monitoring'
        elif has_ecg and has_ppg and has_resp:
            phenotype = 'Standard_Monitoring'
        elif total_signals >= 6:
            phenotype = 'Advanced_Multimodal'
        elif total_signals <= 3:
            phenotype = 'Minimal_Monitoring'
        else:
            phenotype = 'Intermediate_Monitoring'
        
        phenotypes.append(phenotype)
    
    return pd.Series(phenotypes, index=df.index)


def run_phenotype_modeling_improved(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                                   output_dir: Path):
    """Run Phase 6 with improved phenotype naming."""
    print("\n" + "="*70)
    print("PHASE 6: MONITORING PHENOTYPES (IMPROVED NAMING)")
    print("="*70)
    
    # Add dataset labels
    mimic_df = mimic_df.copy()
    uciq_df = uciq_df.copy()
    mimic_df['dataset'] = 'MIMIC'
    uciq_df['dataset'] = 'UCIQ'
    
    # Define phenotypes with improved naming
    print("\nDefining monitoring phenotypes...")
    print("  - Standard_Monitoring: ECG + SpO2 + RESP (basic ICU monitoring)")
    print("  - Hemodynamic_Monitoring: Standard + invasive ABP")
    print("  - Advanced_Hemodynamic: Hemodynamic + CVP/PAP")
    print("  - Neurological_Monitoring: Hemodynamic + ICP/BIS")
    print("  - Ventilated_Hemodynamic: Hemodynamic + CO2/ventilator")
    print("  - Advanced_Multimodal: 6+ signal categories")
    print("  - Minimal_Monitoring: ≤3 signal categories")
    
    mimic_df['phenotype'] = define_phenotypes_improved(mimic_df)
    uciq_df['phenotype'] = define_phenotypes_improved(uciq_df)
    
    # Combine datasets
    combined = pd.concat([mimic_df, uciq_df], ignore_index=True)
    
    # Phenotype distribution
    print("\n" + "-"*70)
    print("PHENOTYPE DISTRIBUTION")
    print("-"*70)
    
    phenotype_dist = pd.crosstab(combined['phenotype'], combined['dataset'])
    phenotype_pct = phenotype_dist.div(phenotype_dist.sum(axis=0), axis=1) * 100
    
    print("\nCounts:")
    print(phenotype_dist.to_string())
    print("\nPercentages:")
    print(phenotype_pct.round(1).to_string())
    
    # Save distribution
    phenotype_dist.to_csv(output_dir / 'phenotype_distribution_v2.csv')
    phenotype_pct.to_csv(output_dir / 'phenotype_percentages_v2.csv')
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    chi2, p_val, dof, expected = chi2_contingency(phenotype_dist)
    
    print(f"\nChi-square test for phenotype independence:")
    print(f"  Chi2 = {chi2:.2f}, p-value = {p_val:.2e}")
    
    # Classification: Predict dataset from signal composition
    print("\n" + "-"*70)
    print("CLASSIFICATION: Dataset Prediction from Signals")
    print("-"*70)
    
    signal_cols = [c for c in combined.columns if c.startswith('has_')]
    
    X = combined[signal_cols].fillna(False).astype(int)
    y = (combined['dataset'] == 'UCIQ').astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    cv_scores = cross_val_score(clf, X, y, cv=5)
    y_pred = clf.predict(X_test)
    
    print(f"\nClassification Performance:")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test accuracy: {clf.score(X_test, y_test):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': signal_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop predictive signals:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature'].replace('has_', '')}: {row['importance']:.3f}")
    
    feature_importance.to_csv(output_dir / 'phenotype_feature_importance_v2.csv', index=False)
    
    # Visualization
    print("\nGenerating phenotype visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Stacked bar chart
    phenotype_pct.T.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_title('Phenotype Distribution by Dataset (%)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Percentage')
    axes[0].legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].tick_params(axis='x', rotation=0)
    
    # Feature importance
    top_features = feature_importance.head(10)
    axes[1].barh(top_features['feature'].str.replace('has_', ''), 
                top_features['importance'], color='steelblue')
    axes[1].set_title('Top Signal Predictors of Dataset', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phenotype_analysis_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phenotype_analysis_v2.png")
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['MIMIC', 'UCIQ'], yticklabels=['MIMIC', 'UCIQ'])
    ax.set_title('Confusion Matrix: Dataset Classification', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'phenotype_confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phenotype_confusion_matrix_v2.png")
    
    print("\n" + "="*70)
    print("Phase 6 (Improved) Complete")
    print("="*70)
    
    return {
        'phenotype_distribution': phenotype_dist,
        'feature_importance': feature_importance,
        'cv_accuracy': cv_scores.mean(),
        'test_accuracy': clf.score(X_test, y_test)
    }


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_phenotype_modeling_improved(mimic_df, uciq_df, output_dir)
