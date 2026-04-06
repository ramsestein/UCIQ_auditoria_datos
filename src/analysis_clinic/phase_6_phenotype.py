"""
Phase 6: Monitoring Phenotype Modeling
Implements phenotype classification using signal composition and demographics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def define_phenotypes(df: pd.DataFrame) -> pd.Series:
    """Define monitoring phenotypes based on signal composition."""
    phenotypes = []
    
    for _, row in df.iterrows():
        signals = []
        
        # Basic monitoring
        if row.get('has_ecg', False):
            signals.append('ECG')
        if row.get('has_ppg', False):
            signals.append('PPG')
        
        # Hemodynamic
        if row.get('has_abp_invasive', False):
            signals.append('ABP')
        if row.get('has_cvp', False):
            signals.append('CVP')
        if row.get('has_pap', False):
            signals.append('PAP')
        
        # Neurological
        if row.get('has_icp', False):
            signals.append('ICP')
        if row.get('has_bis_eeg', False):
            signals.append('BIS')
        
        # Ventilation
        if row.get('has_co2', False):
            signals.append('CO2')
        if row.get('has_ventilation', False):
            signals.append('VENT')
        
        # Define phenotype based on signal set
        if set(['ICP', 'ABP', 'CVP']).issubset(signals):
            phenotypes.append('Neuro_Hemodynamic')
        elif set(['ABP', 'CVP', 'PAP']).issubset(signals):
            phenotypes.append('Cardiac_Hemodynamic')
        elif set(['ECG', 'PPG', 'ABP']).issubset(signals):
            phenotypes.append('Basic_Hemodynamic')
        elif set(['CO2', 'VENT', 'ABP']).issubset(signals):
            phenotypes.append('Ventilated_Hemodynamic')
        elif len(signals) <= 3:
            phenotypes.append('Minimal_Monitoring')
        else:
            phenotypes.append('Comprehensive_Monitoring')
    
    return pd.Series(phenotypes, index=df.index)


def run_phenotype_modeling(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                           output_dir: Path):
    """Run Phase 6: Monitoring phenotype modeling."""
    print("\n" + "="*70)
    print("PHASE 6: MONITORING PHENOTYPE MODELING")
    print("="*70)
    
    # Add dataset labels
    mimic_df = mimic_df.copy()
    uciq_df = uciq_df.copy()
    mimic_df['dataset'] = 'MIMIC'
    uciq_df['dataset'] = 'UCIQ'
    
    # Define phenotypes
    print("\nDefining monitoring phenotypes...")
    mimic_df['phenotype'] = define_phenotypes(mimic_df)
    uciq_df['phenotype'] = define_phenotypes(uciq_df)
    
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
    phenotype_dist.to_csv(output_dir / 'phenotype_distribution.csv')
    phenotype_pct.to_csv(output_dir / 'phenotype_percentages.csv')
    
    # Chi-square test for phenotype differences
    from scipy.stats import chi2_contingency
    chi2, p_val, dof, expected = chi2_contingency(phenotype_dist)
    
    print(f"\nChi-square test for phenotype independence:")
    print(f"  Chi2 = {chi2:.2f}, p-value = {p_val:.2e}")
    
    # Prepare features for classification
    signal_cols = [c for c in combined.columns if c.startswith('has_')]
    
    # Classification: Predict dataset from signal composition
    print("\n" + "-"*70)
    print("CLASSIFICATION: Dataset Prediction from Signals")
    print("-"*70)
    
    X = combined[signal_cols].fillna(False).astype(int)
    y = (combined['dataset'] == 'UCIQ').astype(int)  # 1 = UCIQ, 0 = MIMIC
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    
    # Predictions
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
    
    # Save results
    feature_importance.to_csv(output_dir / 'phenotype_feature_importance.csv', index=False)
    
    # Generate phenotype visualization
    print("\nGenerating phenotype visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stacked bar chart
    phenotype_pct.T.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_title('Phenotype Distribution by Dataset (%)', fontweight='bold')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Percentage')
    axes[0].legend(title='Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Feature importance
    top_features = feature_importance.head(10)
    axes[1].barh(top_features['feature'].str.replace('has_', ''), 
                top_features['importance'], color='steelblue')
    axes[1].set_title('Top Signal Predictors of Dataset', fontweight='bold')
    axes[1].set_xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phenotype_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phenotype_analysis.png")
    
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['MIMIC', 'UCIQ'], yticklabels=['MIMIC', 'UCIQ'])
    ax.set_title('Confusion Matrix: Dataset Classification', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'phenotype_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phenotype_confusion_matrix.png")
    
    print("\n" + "="*70)
    print("Phase 6 Complete")
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
    
    results = run_phenotype_modeling(mimic_df, uciq_df, output_dir)
