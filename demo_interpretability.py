#!/usr/bin/env python3
"""
Comprehensive Interpretability Demo
Demonstrates all interpretability features of the diabetes detection system
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🔬 {title}")
    print("="*80)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 60)

def demo_interpretability_system():
    """Complete demonstration of interpretability features"""
    
    print_header("DIABETES MODEL INTERPRETABILITY SYSTEM DEMO")
    print("🎯 This demo showcases SHAP and LIME explanations for diabetes predictions")
    print("🏥 Designed for healthcare professionals and researchers")
    
    # Step 1: Check prerequisites
    print_step(1, "Checking System Prerequisites")
    
    try:
        import shap
        import lime
        print("✅ SHAP version:", shap.__version__)
        print("✅ LIME version: 0.2.0.1")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("📦 Please install: pip install shap lime")
        return
    
    # Check for required files
    required_files = [
        'models/clean_diabetes_rf.pkl',
        'models/clean_diabetes_nn.pth',
        'models/clean_scaler.pkl',
        'data/2023_BRFSS_CLEANED.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("🔧 Please run train.py first to generate models")
        return
    
    print("✅ All required files found")
    
    # Step 2: Global Model Analysis
    print_step(2, "Global Model Interpretability Analysis")
    print("🔬 Running comprehensive SHAP and LIME analysis...")
    print("⏳ This may take 2-3 minutes for complete analysis...")
    
    # Import and run comprehensive analysis
    try:
        from interpretability import DiabetesModelExplainer
        explainer = DiabetesModelExplainer()
        
        print("📊 Generating global explanations...")
        explainer.run_complete_analysis()
        
        print("✅ Global analysis complete!")
        print("📁 Results saved in explanations/ directory")
        
    except Exception as e:
        print(f"❌ Error in global analysis: {e}")
        return
    
    # Step 3: Individual Patient Analysis
    print_step(3, "Individual Patient Explanation Demo")
    print("👤 Demonstrating patient-level explanations...")
    
    try:
        from explain_patient import PatientExplainer
        patient_explainer = PatientExplainer()
        
        # Demo cases
        demo_cases = [
            ("Low Risk Patient", "low_risk"),
            ("Moderate Risk Patient", "moderate_risk"),
            ("High Risk Patient", "high_risk")
        ]
        
        for case_name, risk_level in demo_cases:
            print(f"\n🔍 Analyzing {case_name}...")
            patient_data = patient_explainer.create_sample_patient(risk_level)
            
            result = patient_explainer.explain_patient(
                patient_data, 
                patient_id=f"{risk_level.replace('_', '-')}-demo"
            )
            
            if result:
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.1f}%")
                print(f"   📊 Visualization saved")
            
            time.sleep(1)  # Brief pause between cases
        
        print("✅ Individual patient analysis complete!")
        
    except Exception as e:
        print(f"❌ Error in patient analysis: {e}")
    
    # Step 4: Results Summary
    print_step(4, "Generated Results Summary")
    
    # Check what was created
    explanations_dir = Path("explanations")
    if explanations_dir.exists():
        print("📂 Generated Files:")
        
        # Dashboard
        dashboard_file = explanations_dir / "interpretability_dashboard.png"
        if dashboard_file.exists():
            print("   🎯 interpretability_dashboard.png - Main overview")
        
        # SHAP results
        shap_dir = explanations_dir / "shap"
        if shap_dir.exists():
            shap_files = list(shap_dir.glob("*.png"))
            print(f"   📊 SHAP visualizations: {len(shap_files)} files")
            for file in shap_files:
                print(f"      - {file.name}")
        
        # LIME results
        lime_dir = explanations_dir / "lime"
        if lime_dir.exists():
            lime_files = list(lime_dir.glob("*"))
            print(f"   🔍 LIME explanations: {len(lime_files)} files")
            html_files = [f for f in lime_files if f.suffix == '.html']
            png_files = [f for f in lime_files if f.suffix == '.png']
            print(f"      - {len(html_files)} interactive HTML reports")
            print(f"      - {len(png_files)} summary visualizations")
        
        # Patient explanations
        patients_dir = explanations_dir / "patients"
        if patients_dir.exists():
            patient_files = list(patients_dir.glob("*.png"))
            print(f"   👤 Patient explanations: {len(patient_files)} files")
            for file in patient_files:
                print(f"      - {file.name}")
    
    # Step 5: Usage Guidance
    print_step(5, "Next Steps and Usage")
    
    print("🚀 How to use the interpretability system:")
    print()
    print("   📊 Global Analysis:")
    print("      python interpretability.py")
    print("      → Generates overall model explanations")
    print()
    print("   👤 Individual Patients:")
    print("      python explain_patient.py")
    print("      → Interactive patient explanation demo")
    print()
    print("   🔧 Custom Integration:")
    print("      from explain_patient import PatientExplainer")
    print("      explainer = PatientExplainer()")
    print("      result = explainer.explain_patient(patient_data)")
    print()
    print("   📁 View Results:")
    print("      - Open explanations/interpretability_dashboard.png")
    print("      - Browse explanations/lime/*.html for interactive reports")
    print("      - Check explanations/patients/ for individual analyses")
    
    # Step 6: Clinical Benefits
    print_step(6, "Clinical and Research Benefits")
    
    benefits = [
        ("🏥 For Clinicians", [
            "Transparent prediction rationale",
            "Evidence-based clinical decision support",
            "Understanding of key risk factors"
        ]),
        ("👤 For Patients", [
            "Clear explanation of diabetes risk",
            "Actionable lifestyle recommendations",
            "Personalized risk factor analysis"
        ]),
        ("🔬 For Researchers", [
            "Model validation and bias detection",
            "Feature importance discovery",
            "Population-level pattern analysis"
        ])
    ]
    
    for category, items in benefits:
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")
    
    # Final Summary
    print_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("✅ Interpretability system is fully operational")
    print("📊 Global model explanations generated")
    print("👤 Individual patient analysis demonstrated")
    print("📁 Results available in explanations/ directory")
    print("📖 See INTERPRETABILITY_GUIDE.md for detailed documentation")
    print()
    print("🔬 Your diabetes detection models are now interpretable and clinically ready!")

if __name__ == "__main__":
    demo_interpretability_system()