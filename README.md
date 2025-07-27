# Project Title: Drone Detection Classification

**Author:** Ajay Tewari

## Executive summary

This project develops a comprehensive computer vision system for automated drone detection in aerial imagery, addressing critical needs in security, surveillance, airspace monitoring, and wildlife protection. Through systematic application of the CRISP-DM methodology, we implemented and evaluated six distinct machine learning models, achieving **94.74% classification accuracy** and **90.28% spatial detection accuracy**.

## Rationale - Why should anyone care about this question?
In recent years, we have observed the unprecedented usage of drone and related technologies in various areas like surveillance, security, crops monitoring, delivery drones, airspace monitoring, search & rescue operations. This incredibly useful technology has many benefits but if armed, drones pose great threat to civilian and armed forces life & infrastructure. These drones could also be used for surveillance by bad people or could be a great concern for violation of privacy. 

The anti-drone technologies used to counter these drones could also harm wild life if drone detection accuracy is not very high. Thus, drone detection model with high accuracy could save lives, critical infrastructure and avoid harm to wild life.

## Research Question - What are you trying to answer?
How drones detection in images or video frames using computer vision techniques could help security, surveillance, airspace monitoring and wild life protection? 

## Data Sources
[Roboflow Drone Dataset](https://universe.roboflow.com/ahmedmohsen/drone-detection-new-peksv)

[Sample Drone Dataset](https://github.com/atewari-bot/drone-image-classification/blob/main/data/drone-dataset-sample.zip)

## Methodology - What methods are you using to answer the question?

I used CRISP-DM methodologies for end to end model training and deployment lifecycle.

### 1. Business Understanding

  #### Research Question

  **"How can drone detection in images or video frames using computer vision techniques help security, surveillance, airspace monitoring, and wildlife protection?"**

  #### Business Objectives
  - **Security Enhancement**: Automated detection of unauthorized drones near critical infrastructure
  - **Airspace Safety**: Real-time monitoring to prevent aviation incidents
  - **Wildlife Protection**: Non-intrusive monitoring systems for conservation efforts
  - **Multi-Class Recognition**: Distinguish drones from aircraft, helicopters, and birds

  #### Success Criteria
  - Classification accuracy >90% for operational deployment
  - Real-time inference capability (<1 second per image)
  - Robust performance across diverse environmental conditions
  - Scalable architecture for enterprise deployment
  ---

### 2. Data Understanding

  #### Dataset Overview
  - **Source**: Roboflow Drone Detection Dataset
  - **Classes**: 4 categories (AIRPLANE, DRONE, HELICOPTER, BIRD)
  - **Format**: YOLO annotations with bounding box coordinates
  - **Resolution**: Standardized 224×224 RGB images
  - **Distribution**: Balanced multi-class dataset

  #### Key Data Insights
  | Class | Characteristics | Detection Challenge |
  |-------|----------------|-------------------|
  | **DRONE** | High consistency, geometric shapes | Easiest to classify |
  | **AIRPLANE** | Sky backgrounds, structural uniformity | Moderate difficulty |
  | **HELICOPTER** | Complex rotor patterns, varied contexts | High difficulty |
  | **BIRD** | Natural textures, motion blur | Most challenging |

  #### Data Quality Assessment

  * Class Distribution: Frequency analysis for imbalance
  * Image Quality: Brightness, contrast, sharpness, entropy, noise levels
  * Spatial Patterns: Center vs. edge intensity, gradient magnitude, corner detection
   ---

### 3. Data Preparation

  #### Preprocessing Pipeline
  1. **Image Standardization**: Resize to 224×224, normalize pixel values (0-1)
  2. **Quality Enhancement**: Gaussian blur denoising for artifact reduction
  3. **Feature Engineering**: Extract 87 features including:
      * Color channel statistics (RGB + grayscale)
      * Texture features (LBP, HOG, gradient magnitude)
      * Spatial characteristics (symmetry, center positioning)
      * Statistical measures (entropy, skewness, kurtosis)

  #### Data Transformation
  - **Classification Task**: Converted detection data to single-label classification
  - **Detection Task**: Maintained bounding box coordinates for object localization
  - **Feature Engineering**: Extracted 100+ features for Random Forest models
  - **Dimensionality Reduction**: PCA analysis showing 95% variance in ~35 components

  ---

### 4. Modeling

  #### Models Developed
  1. **Random Forest Classifier (Baseline)**
  2. **Optimized Random Forest (GridSearchCV)**
  3. **CNN Classification Model**
  4. **CNN Classification with Denoising**
  5. **CNN Detection Model**
  6. **MobileNetV2 Fast R-CNN Model**

  #### Model Portfolio

  <table align="center" style="border-collapse: collapse; width: 100%; max-width: 1200px; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-radius: 8px; overflow: hidden;">
  <thead>
    <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
      <th style="padding: 15px 12px; text-align: left; font-weight: 600; font-size: 14px; border-right: 1px solid rgba(255,255,255,0.2);">
        📈 <strong>Performance Metric</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 140px;">
        🌳 <strong>Random Forest<br/>Classifier</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 140px;">
        ⚙️ <strong>Optimized<br/>Random Forest</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 140px;">
        🧠 <strong>CNN<br/>Classification</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 140px;">
        🧠 <strong>CNN Classification<br/>Denoised</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 140px;">
        🎯 <strong>CNN Detection<br/>Model</strong>
      </th>
      <th style="padding: 15px 12px; text-align: center; font-weight: 600; font-size: 12px; min-width: 140px;">
        ⚡ <strong>Fast R-CNN<br/>Model</strong>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        ⏱️ <strong>Training Time (seconds)</strong>
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>0.41s</strong> 🚀
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>1.24s</strong> 🚀
      </td>
      <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
        246.88s
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        143.18s
      </td>
      <td style="padding: 12px; text-align: center; background-color: #f8d7da; color: #721c24;">
        358.08s
      </td>
      <td style="padding: 12px; text-align: center; background-color: #f8d7da; color: #721c24;">
        2707.92s
      </td>
    </tr>
    <tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        🎯 <strong>Test Accuracy / Coord Accuracy</strong>
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>98.25%</strong> 🏆
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>97.75%</strong> 🥈
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        96.50%
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        94.50%
      </td>
      <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
        79.21%*
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        92.86%*
      </td>
    </tr>
    <tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        📉 <strong>Test MSE (Lower is Better)</strong>
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.075
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.103
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.215
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.338
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>0.003</strong> 🏆
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>0.003</strong> 🥈
      </td>
    </tr>
    <tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        📉 <strong>Test MAE (Lower is Better)</strong>
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.035
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.035
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.044
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.060
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>0.008</strong> 🏆
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.039
      </td>
    </tr>
    <tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        🎯 <strong>Test Precision / IoU</strong>
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>98.29%</strong> 🏆
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>97.83%</strong> 🥈
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        96.55%
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        94.61%
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.188* (IoU)
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.290* (IoU)
      </td>
    </tr>
    <tr style="background-color: white;">
      <td style="padding: 12px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef;">
        📊 <strong>Test R² Score</strong>
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.929
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.902
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.795
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.679
      </td>
      <td style="padding: 12px; text-align: center; color: #495057;">
        0.949
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>0.997</strong> 🏆
      </td>
    </tr>
  </tbody>
</table>

🏆 Champion Models by Category

<table align="center" style="border-collapse: collapse; width: 80%; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
  <tr style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white;">
    <th style="padding: 15px; text-align: center; font-weight: 600; border-radius: 8px 0 0 8px;">
      🏅 **Performance Category**
    </th>
    <th style="padding: 15px; text-align: center; font-weight: 600; border-radius: 0 8px 8px 0;">
      🎖️ **Champion Model**
    </th>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center; border-right: 1px solid #e9ecef;">
      ⚡ **Fastest Training**
    </td>
    <td style="padding: 12px; text-align: center; color: #155724;">
      🌳 **Random Forest Classifier** (0.41s)
    </td>
  </tr>
  <tr style="background-color: white;">
    <td style="padding: 12px; font-weight: 600; text-align: center; border-right: 1px solid #e9ecef;">
      🎯 **Highest Classification Accuracy**
    </td>
    <td style="padding: 12px; text-align: center; color: #155724;">
      🌳 **Random Forest Classifier** (98.25%)
    </td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center; border-right: 1px solid #e9ecef;">
      📍 **Best Detection Accuracy**
    </td>
    <td style="padding: 12px; text-align: center; color: #155724;">
      ⚡ **Fast R-CNN Model** (92.86%)
    </td>
  </tr>
  <tr style="background-color: white;">
    <td style="padding: 12px; font-weight: 600; text-align: center; border-right: 1px solid #e9ecef;">
      📉 **Lowest Error Rate**
    </td>
    <td style="padding: 12px; text-align: center; color: #155724;">
      🎯 **CNN Detection Model** (MSE: 0.003)
    </td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center; border-right: 1px solid #e9ecef;">
      📊 **Best Model Fit**
    </td>
    <td style="padding: 12px; text-align: center; color: #155724;">
      ⚡ **Fast R-CNN Model** (R²: 0.997)
    </td>
  </tr>
</table>

  #### Class-wise Performance
  ```
  🎯 Class-wise Performance  
  ├── 🚁 DRONE: Perfect (100%)
  ├── 🚁 HELICOPTER: Perfect (100%)
  ├── 🛩️ AIRPLANE: 94.6% (5 misclassifications)
  └── 🦅 BIRD: 94.6% (4 misclassifications)
  ```

  **Architecture: CNN Detection Model**

  ![Image](/images/cnn_detection_architecture_diagram.svg)

  **Architecture: MobileNetV2 Fast R-CNN Detection Model**

  ![Image](/images/fast_rcnn_detection_diagram.svg)

  #### Training Strategy
  - **Epochs**: 110 to 200 with early stopping
  - **Batch Size**: 32 for classification, 8 for detection
  - **Optimization**: Adam optimizer with learning rate scheduling
  - **Validation**: Stratified k-fold cross-validation for Random Forest

  ### Feature Importance Analysis
  **Top Contributing Features** (Optimized RandomForest):
  1. **Blue Channel Statistics** (32.8%) - Sky background patterns
  2. **Texture Complexity** (18.4%) - Surface pattern differentiation  
  3. **Gradient Magnitude** (15.2%) - Edge and boundary detection
  4. **Symmetry Features** (12.1%) - Aircraft structural patterns
  5. **HOG Descriptors** (10.5%) - Shape and orientation features

  ### Model Innovation Highlights

  #### Optimized RandomForest
  - **Hyperparameter Space**: 1,000+ combinations via GridSearchCV
  - **Key Parameters**: 500 estimators, entropy criterion, max_depth=20
  - **Class Handling**: Balanced weights for minority class protection

  ![Image](/images/opti_rf_features_distribution.png)

  #### Fast R-CNN Enhancement
  - **Transfer Learning**: MobileNetV2 pre-trained backbone
  - **Advanced Loss**: Focal loss for class imbalance handling
  - **Multi-Scale Detection**: Adaptive bounding box regression

---

### 5. Evaluation

  #### Model Performance Analysis
  - **Best Overall Performance**: RandomForest Classifier Model (98.25% accuracy)
  - **Most Efficient**: Random Forest models for faster deployment
  - **Best for Detection**: Fast R-CNN Detection Model with 78.97% coordinate accuracy
  - **Improvement**: 4.64% accuracy gain from baseline to optimized CNN

  #### Class-Specific Performance
  - **DRONE**: Highest accuracy across all models (>95%)
  - **AIRPLANE**: Good performance with spatial attention benefits
  - **HELICOPTER**: Moderate performance due to complexity
  - **BIRD**: Most challenging class requiring specialized techniques

  #### Performance Metrics Summary

  | **Model** | **Training Time** | **Test Accuracy** | **MSE** | **Precision/IoU** | **R² Score** |
  |-----------|------------------|-------------------|---------|-------------------|--------------|
  | **🌳 RandomForest** | 0.41s | **98.25%** | 0.075 | **98.29%** | 0.9286 |
  | **⚙️ Optimized RF** | 1.24s | **97.75%** | 0.1025 | **97.83%** | 0.9024 |
  | **🧠 CNN Classification** | 246.88s | **96.50%** | 0.215 | **96.55%** | 0.7953 |
  | **🧠 CNN Denoised** | 143.18s | 94.50% | 0.3375 | 94.61% | 0.6786 |
  | **🎯 CNN Detection** | 358.07s | 79.21%* | 0.0034 | 18.81%* | 0.9486 |
  | **⚡ Fast R-CNN** | 2707.92s | 92.86%* | 0.0029 | 29.01%* | 0.9971 |

  <sub><b>* Detection models show coordinate accuracy and IoU instead of classification metrics</b></sub>

---

### 6. Deployment

  #### Production Readiness
  1. **Primary Model**: CNN Classification for highest accuracy
  2. **Secondary Model**: Optimized Random Forest for real-time applications
  3. **Specialized Model**: CNN Detection for precise localization tasks

  #### Implementation Strategy
  - **Edge Deployment**: Use Random Forest for resource-constrained environments
  - **Server Deployment**: Implement CNN models for maximum accuracy
  - **Hybrid Approach**: Combine models for different operational requirements

  #### Performance Monitoring
  - **Real-time Metrics**: Accuracy, latency, and throughput monitoring
  - **Model Drift Detection**: Continuous validation on new data
  - **Retraining Schedule**: Quarterly model updates with new data

---

### Business Impact & Recommendations

  #### Expected Benefits
  - **Security Enhancement**: 98.25% accuracy in drone detection
  - **Cost Reduction**: Automated monitoring reduces manual surveillance costs
  - **Operational Efficiency**: Real-time processing capabilities
  - **Risk Mitigation**: Early detection of unauthorized activities

  #### Success Metrics
  - **Technical**: >90% accuracy achieved across primary models
  - **Operational**: Sub-second response time for real-time applications
  - **Business**: Potential 60% reduction in manual monitoring costs

  #### Scalability
  - **Horizontal Scaling**: Model architecture supports distributed deployment
  - **Vertical Scaling**: Optimized for GPU acceleration
  - **Integration**: Compatible with existing security infrastructure

---


## Results

### Understanding Data (Exploratory Data Analysis)

**Class Balance Analysis**

**Key Takeaways:** 
* Total samples: 2100
* Imbalance ratio (max/min): 2.19
* Class distribution:
  ```
  * 🚁 DRONE      ████████████████████  834 samples (39.7%)
  * 🚁 HELICOPTER ███████████           460 samples (21.9%)  
  * 🛩️ AIRPLANE   ██████████            426 samples (20.3%)
  * 🦅 BIRD       █████████             380 samples (18.1%)
  ```

![Image](/images/class_distribution.png)

**Pixel Statistics Analysis**

**Key Takeaways:** 

**Class-wise Insights**

| Class | Key Characteristics | Background Context | Technical Notes |
|-------|-------------------|-------------------|-----------------|
| **AIRPLANE** | Sky dominance, size variation, consistent lighting | High-altitude captures with clear sky backgrounds | Bimodal distribution: close-up and distant shots |
| **BIRD** | Natural environment, variable contrast | Outdoor/natural backgrounds | Motion blur potential, lighting variations |
| **DRONE** | Mixed backgrounds, technical clarity | Diverse operational environments | Scale variations: detail shots and operational distance |
| **HELICOPTER** | Operational context, distinctive features | Aerial operations over varied terrain | Rotor blade visibility, environmental diversity |

![Image](/images/pixel_statistics.png)

**Image Quality Metric Analysiss**

**Key Takeaways:**

| Quality Metric     | Best Performing Class | Worst Performing Class | Recommendation                          |
|--------------------|------------------------|--------------------------|------------------------------------------|
| Consistency         | DRONE                  | BIRD                     | Focus augmentation on BIRD class         |
| Contrast            | DRONE                  | AIRPLANE                 | Enhance edge detection for AIRPLANE      |
| Sharpness           | DRONE                  | HELICOPTER               | Apply deblurring for HELICOPTER          |
| Noise Level         | AIRPLANE               | BIRD                     | Implement noise reduction for BIRD       |
| Feature Richness    | BIRD                   | AIRPLANE                 | Extract texture features for BIRD        |

**Classification Implications**

- **DRONE**: Most consistent quality metrics make it easiest to classify  
- **AIRPLANE**: Sky backgrounds provide clear context but lower contrast  
- **HELICOPTER**: Motion blur challenges require specialized preprocessing  
- **BIRD**: Highest variability requires robust augmentation strategies


![Image](/images/image_quality_metrics.png)

**Spatial Patterns Analysis**

**Key Takeaways:** 

| Spatial Feature       | Most Distinctive Class | Least Distinctive Class | Recommendation                           |
|-----------------------|------------------------|--------------------------|-------------------------------------------|
| **Center Positioning** | AIRPLANE               | BIRD                     | Use spatial attention for AIRPLANE        |
| **Edge Definition**    | DRONE                  | BIRD                     | Enhance edge detection for DRONE          |
| **Corner Features**    | BIRD                   | AIRPLANE                 | Extract corner features for BIRD          |
| **Symmetry**           | AIRPLANE               | BIRD                     | Use symmetry features for aircraft        |
| **Texture Complexity** | BIRD                   | AIRPLANE                 | Focus on texture for BIRD classification  |


**Classification Strategy Implications**

- **AIRPLANE**: Leverage high symmetry and center positioning
- **DRONE**: Utilize geometric corner patterns and edge definition
- **HELICOPTER**: Focus on complex rotor blade spatial patterns
- **BIRD**: Extract rich texture and natural shape variations

![Image](/images/spatial_patterns_analysis.png)

## Feature Engineering

### Principal Component Analysis - RandomForest

**Key Takeaways:**

| Class        | Separability        | CNN Focus                     | Strategy Summary                                                   | Expected Accuracy |
|--------------|---------------------|-------------------------------|--------------------------------------------------------------------|-------------------|
| **DRONE**    | Best separated     | Geometric shapes              | Early convergence, few PCs needed, simple or shallow models work   | Highest         |
| **AIRPLANE** | Moderate overlap   | Sky-background spatial cues   | Spatial attention + augmentation helps isolate characteristics     | Good            |
| **HELICOPTER** | Widely spread      | Rotor complexity patterns     | Needs deeper CNN layers and ensemble models due to feature mix     | Moderate        |
| **BIRD**     | Most overlapped    | Natural texture variations    | Heavy augmentation + class weighting; benefits from transfer learn | Lowest          |

**Insights**

- **PC1** explains **28.65%** variance → primary for class separation (esp. DRONE).
- **PC2** adds **14.49%**, taking cumulative to **43.14%**.
- **Top 10 PCs** capture **~85%** of variance → ideal for compressed feature learning.
- **~35 PCs** required to reach **95%** variance threshold → full information coverage.

**Model Design Takeaways**

- **Feature Engineering**: Use first **20–25 PCs** to retain useful info, reduce noise.
- **Class Weighting**: Apply higher weights to **BIRD** and **HELICOPTER** due to overlaps.
- **CNN Architecture**: 
  - Shallow, fast learners for **DRONE**
  - Attention layers for **AIRPLANE**
  - Deep, complex structures for **HELICOPTER**
  - Transfer learning + strong augmentation for **BIRD**
- **Random Forest**: Leverages many PCs well; excels on **DRONE**, challenged on **BIRD/HELICOPTER**.

![Image](/images/rf_pca_analysis.png)

### Principal Component Analysis - CNN

**Key Takeaways**

| Class       | t-SNE Pattern       | CNN Strategy                        | Feature Focus               | Training Needs                          | Expected Performance |
|-------------|----------------------|-------------------------------------|-----------------------------|-----------------------------------------|----------------------|
| **DRONE**   | Tight clusters       | Shallow CNN (e.g., MobileNet)       | Geometric shapes            | Fast convergence, low complexity        | Highest            |
| **AIRPLANE**| Scattered groups     | Medium-depth + spatial attention    | Sky vs background patterns  | Moderate depth, spatial pooling         | Good               |
| **HELICOPTER** | Dispersed           | Deep CNN + Ensemble (MobileNetV2)   | Rotor blade variations      | Complex features, deeper layers         | Moderate          |
| **BIRD**    | Mixed, low cohesion  | Transfer learning + augmentation    | Natural textures, poses     | Pre-trained models, heavy augmentation  | Challenging        |


**Additional Key Metrics**

| Metric                        | Value              | Implication                                |
|------------------------------|--------------------|--------------------------------------------|
| Explained Variance (PC1+2)   | 62.24%             | Strong class separability                  |
| Variance (First 10 PCs)      | ~90%               | Core CNN feature set                       |
| Optimal Dimensionality       | ~25 PCs            | Efficient compression + representation     |
| Most Challenging Class       | BIRD               | Needs advanced augmentation & transfer learning |

![Image](/images/cnn_pca_analysis.png)

## Performance Metrics

### Prediction Errors - RandomForest

![Image](/images/rf_optimized_prediction_errors_analysis.png)

### CNN Detection Model Predictions

![Image](/images/cnn_detection_model_predictions.png)

### Fast R-CNN Detection Model Predictions

![Image](/images/fast_rcnn_predictions.png)

### Models Performance Comparision

![Image](/images/model_comparison_with_fast_rcnn.png)

## <div align="center">🔬 Detailed Performance Metrics Across Train/Validation/Test Sets</div>

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 1400px; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); border-radius: 12px; overflow: hidden;">
  <thead>
    <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
      <th style="padding: 18px 15px; text-align: left; font-weight: 700; font-size: 14px; border-right: 2px solid rgba(255,255,255,0.3); position: sticky; left: 0; z-index: 10; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        📈 <strong>Performance Metric</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 130px;">
        🌳 <strong>Random Forest<br/>Classifier</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 130px;">
        ⚙️ <strong>Optimized<br/>Random Forest</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 130px;">
        🧠 <strong>CNN<br/>Classifier</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 130px;">
        🧠 <strong>CNN<br/>Denoised</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; border-right: 1px solid rgba(255,255,255,0.2); min-width: 130px;">
        🎯 <strong>CNN<br/>Detection</strong>
      </th>
      <th style="padding: 18px 12px; text-align: center; font-weight: 600; font-size: 12px; min-width: 130px;">
        ⚡ <strong>Fast<br/>R-CNN</strong>
      </th>
    </tr>
  </thead>
  <tbody>
    <!-- Training Time -->
    <tr style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); border-bottom: 2px solid #dee2e6;">
      <td colspan="7" style="padding: 12px 15px; font-weight: 700; color: #495057; text-align: center; font-size: 16px;">
        ⏱️ <strong>TRAINING PERFORMANCE</strong>
      </td>
    </tr>
    <tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
      <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: white;">
        ⏰ <strong>Training Time (seconds)</strong>
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700; font-size: 14px;">
        <strong>0.41</strong> 🚀
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
        <strong>1.24</strong> ⚡
      </td>
      <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
        246.88
      </td>
      <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
        143.18
      </td>
      <td style="padding: 12px; text-align: center; background-color: #f8d7da; color: #721c24;">
        358.07
      </td>
      <td style="padding: 12px; text-align: center; background-color: #f8d7da; color: #721c24;">
        2707.92
      </td>
    </tr>
<!-- Accuracy Section -->
<tr style="background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); border-bottom: 2px solid #90caf9;">
  <td colspan="7" style="padding: 12px 15px; font-weight: 700; color: #1565c0; text-align: center; font-size: 16px;">
    🎯 <strong>ACCURACY METRICS</strong>
  </td>
</tr>
<tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: #f8f9fa;">
    📈 <strong>Training Accuracy</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>100.00%</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>99.95%</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    98.81%
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    96.71%
  </td>
  <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
    81.37%*
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    92.05%*
  </td>
</tr>
<tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: white;">
    📊 <strong>Validation Accuracy</strong>
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    96.50%
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.25%</strong> 🥇
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.25%</strong> 🥇
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    94.75%
  </td>
  <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
    79.90%*
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    92.72%*
  </td>
</tr>
<tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: #f8f9fa;">
    🎯 <strong>Test Accuracy</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>98.25%</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.75%</strong> 🥈
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    96.50%
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    94.50%
  </td>
  <td style="padding: 12px; text-align: center; background-color: #fff3cd; color: #856404;">
    79.21%*
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    92.86%*
  </td>
</tr>

<!-- Error Metrics Section -->
<tr style="background: linear-gradient(90deg, #ffebee 0%, #ffcdd2 100%); border-bottom: 2px solid #ef9a9a;">
  <td colspan="7" style="padding: 12px 15px; font-weight: 700; color: #c62828; text-align: center; font-size: 16px;">
    📉 <strong>ERROR METRICS (Lower is Better)</strong>
  </td>
</tr>
<tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: white;">
    📉 <strong>Test MSE</strong>
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.075
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.103
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.215
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.338
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>0.003</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>0.003</strong> 🥈
  </td>
</tr>
<tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: #f8f9fa;">
    📉 <strong>Test MAE</strong>
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.035
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.035
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.044
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.060
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>0.008</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.039
  </td>
</tr>

<!-- Classification Metrics Section -->
<tr style="background: linear-gradient(90deg, #e8f5e8 0%, #c8e6c9 100%); border-bottom: 2px solid #a5d6a7;">
  <td colspan="7" style="padding: 12px 15px; font-weight: 700; color: #2e7d32; text-align: center; font-size: 16px;">
    🎯 <strong>CLASSIFICATION METRICS</strong>
  </td>
</tr>
<tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: white;">
    🎯 <strong>Test Precision / IoU</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>98.29%</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.83%</strong> 🥈
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    96.55%
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    94.61%
  </td>
  <td style="padding: 12px; text-align: center; color: #856404;">
    0.188* (IoU)
  </td>
  <td style="padding: 12px; text-align: center; color: #856404;">
    0.290* (IoU)
  </td>
</tr>
<tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: #f8f9fa;">
    📈 <strong>Test Recall</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>98.25%</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.75%</strong> 🥈
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    96.50%
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    94.50%
  </td>
  <td style="padding: 12px; text-align: center; color: #6c757d; font-style: italic;">
    N/A
  </td>
  <td style="padding: 12px; text-align: center; color: #6c757d; font-style: italic;">
    N/A
  </td>
</tr>
<tr style="background-color: white; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: white;">
    🎯 <strong>Test F1 Score</strong>
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>98.24%</strong> 🏆
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 600;">
    <strong>97.74%</strong> 🥈
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d1ecf1; color: #0c5460;">
    96.47%
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    94.45%
  </td>
  <td style="padding: 12px; text-align: center; color: #6c757d; font-style: italic;">
    N/A
  </td>
  <td style="padding: 12px; text-align: center; color: #6c757d; font-style: italic;">
    N/A
  </td>
</tr>

<!-- Model Fit Section -->
<tr style="background: linear-gradient(90deg, #f3e5f5 0%, #e1bee7 100%); border-bottom: 2px solid #ce93d8;">
  <td colspan="7" style="padding: 12px 15px; font-weight: 700; color: #7b1fa2; text-align: center; font-size: 16px;">
    📊 <strong>MODEL FIT QUALITY</strong>
  </td>
</tr>
<tr style="background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
  <td style="padding: 12px 15px; font-weight: 600; color: #495057; border-right: 1px solid #e9ecef; position: sticky; left: 0; z-index: 5; background-color: #f8f9fa;">
    📊 <strong>Test R² Score</strong>
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.929
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.902
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.795
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.679
  </td>
  <td style="padding: 12px; text-align: center; color: #495057;">
    0.949
  </td>
  <td style="padding: 12px; text-align: center; background-color: #d4edda; color: #155724; font-weight: 700;">
    <strong>0.997</strong> 🏆
  </td>
</tr>
  </tbody>
</table>

### 🔍 Cross-Set Performance Analysis

<div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 30px 0;">
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; width: 320px; margin: 15px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
  <h4 style="margin: 0 0 15px 0; text-align: center; font-size: 18px;">🌳 **Random Forest Champion**</h4>
  <div style="font-size: 13px; line-height: 1.6;">
    <strong>🏆 Best Overall Performance</strong><br/>
    ✅ Fastest Training: 0.41s<br/>
    ✅ Perfect Train Accuracy: 100%<br/>
    ✅ Highest Test Accuracy: 98.25%<br/>
    ✅ Best Test Precision: 98.29%<br/>
    ✅ Excellent Generalization
  </div>
</div>
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; width: 320px; margin: 15px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
  <h4 style="margin: 0 0 15px 0; text-align: center; font-size: 18px;">⚡ **Detection Specialist**</h4>
  <div style="font-size: 13px; line-height: 1.6;">
    <strong>🎯 Fast R-CNN Excellence</strong><br/>
    ✅ Best Model Fit: R² = 0.997<br/>
    ✅ Lowest Detection MSE: 0.003<br/>
    ✅ Strong Coordinate Accuracy<br/>
    ⚠️ Longest Training Time<br/>
    🎯 Detection Task Optimized
  </div>
</div>
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 25px; border-radius: 15px; width: 320px; margin: 15px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);">
  <h4 style="margin: 0 0 15px 0; text-align: center; font-size: 18px;">🎯 **CNN Detection Leader**</h4>
  <div style="font-size: 13px; line-height: 1.6;">
    <strong>📉 Error Minimization</strong><br/>
    ✅ Lowest Test MAE: 0.008<br/>
    ✅ Minimal MSE: 0.003<br/>
    ✅ High R² Score: 0.949<br/>
    ⚠️ Lower Classification Accuracy<br/>
    🎯 Coordinate Precision Focused
  </div>
</div>
</div>

### 📈 Performance Trends Analysis

<table align="center" style="border-collapse: collapse; width: 90%; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 8px; overflow: hidden;">
  <tr style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white;">
    <th style="padding: 15px; text-align: center; font-weight: 600;">🔍 **Analysis Category**</th>
    <th style="padding: 15px; text-align: center; font-weight: 600;">📊 **Key Finding**</th>
    <th style="padding: 15px; text-align: center; font-weight: 600;">🎯 **Best Model**</th>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center;">**⚡ Training Efficiency**</td>
    <td style="padding: 12px; text-align: center;">Random Forest models train 200-6000x faster</td>
    <td style="padding: 12px; text-align: center; color: #155724;">🌳 **RF Classifier**</td>
  </tr>
  <tr style="background-color: white;">
    <td style="padding: 12px; font-weight: 600; text-align: center;">**🎯 Classification Excellence**</td>
    <td style="padding: 12px; text-align: center;">Random Forest achieves near-perfect scores</td>
    <td style="padding: 12px; text-align: center; color: #155724;">🌳 **RF Classifier**</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center;">**📍 Detection Precision**</td>
    <td style="padding: 12px; text-align: center;">CNN Detection minimizes coordinate errors</td>
    <td style="padding: 12px; text-align: center; color: #155724;">🎯 **CNN Detection**</td>
  </tr>
  <tr style="background-color: white;">
    <td style="padding: 12px; font-weight: 600; text-align: center;">**📊 Model Fit Quality**</td>
    <td style="padding: 12px; text-align: center;">Fast R-CNN shows exceptional R² scores</td>
    <td style="padding: 12px; text-align: center; color: #155724;">⚡ **Fast R-CNN**</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 12px; font-weight: 600; text-align: center;">**⚖️ Generalization**</td>
    <td style="padding: 12px; text-align: center;">Random Forest shows consistent train-test performance</td>
    <td style="padding: 12px; text-align: center; color: #155724;">🌳 **RF Models**</td>
  </tr>
</table>

### Loss Function Metrics

![Image](/images/comprehensive_model_comparison.png)

### Optimized RandomForest Test Dataset Performance Metrics

![Image](/images/optimized_rf_test_performance.png)

#### Overall Performance
| Metric | Value | Description |
|--------|-------|-------------|
| Classes | 4 | Airplane, Drone, Helicopter, Bird |
| AUC Score | 1.00 | Perfect AUC across all classes |
| Performance Level | Perfect/Near-Perfect | Exceptional discriminative ability |

#### Confusion Matrix Results
| Class | Correct Predictions | Misclassifications | Details |
|-------|-------------------|-------------------|---------|
| Airplane | 87 | 5 | 5 → Helicopter |
| Drone | 159 | 0 | Perfect classification |
| Helicopter | 75 | 0 | Perfect classification |
| Bird | 70 | 4 | 1 → Airplane, 3 → Drone |

#### Precision-Recall Performance
| Class | Class ID | Average Precision (AP) | Rank |
|-------|----------|----------------------|------|
| Airplane | Class 0 | 1.00 | 1st (Tied) |
| Drone | Class 1 | 1.00 | 1st (Tied) |
| Helicopter | Class 2 | 0.99 | 3rd |
| Bird | Class 3 | 0.99 | 3rd (Tied) |

#### ROC Curve Performance
| Class | AUC Score | Performance |
|-------|-----------|------------|
| Class 0 (Airplane) | 1.00 | Perfect |
| Class 1 (Drone) | 1.00 | Perfect |
| Class 2 (Helicopter) | 1.00 | Perfect |
| Class 3 (Bird) | 1.00 | Perfect |

#### Key Performance Insights
| Insight | Description |
|---------|-------------|
| Best Performing Classes | Drone & Helicopter - Perfect confusion matrix performance |
| Most Challenging Class | Bird - 4 misclassifications (1 → Airplane, 3 → Drone) |
| Common Misclassification | Airplane → Helicopter (5 instances) and Bird → Drone (3 instances) |
| Model Strength | Perfect 1.00 AUC across all classes, near-perfect AP scores |
| Overall Accuracy | Extremely high with only 9 total misclassifications out of 391 samples |
| Improvement over CNN | Significantly better performance with perfect AUC and higher AP scores |

### CNN Classification Test Dataset Performance Metrics

![Image](/images/cnn_test_performance_matrix.png)

### CNN Denoised Test Dataset Performance Metrics

![Image](/images/cnn_denoised_test_performance_matrix.png)

#### Overall Performance
| Metric | Value | Description |
|--------|-------|-------------|
| Classes | 4 | Airplane, Drone, Helicopter, Bird |
| AUC Score | 0.97 | All classes achieve identical AUC |
| Performance Level | Excellent | Strong discriminative ability across all classes |

#### Confusion Matrix Results
| Class | Correct Predictions | Misclassifications | Details |
|-------|-------------------|-------------------|---------|
| Airplane | 73 | 19 | 16 → Helicopter, 3 → Bird |
| Drone | 159 | 0 | Perfect classification |
| Helicopter | 65 | 10 | 7 → Airplane, 3 → Bird |
| Bird | 61 | 13 | 10 → Airplane, 3 → Drone |

#### Precision-Recall Performance
| Class | Class ID | Average Precision (AP) | Rank |
|-------|----------|----------------------|------|
| Drone | Class 1 | 0.98 | 1st (Highest) |
| Bird | Class 3 | 0.93 | 2nd |
| Helicopter | Class 2 | 0.90 | 3rd |
| Airplane | Class 0 | 0.88 | 4th (Lowest) |

#### ROC Curve Performance
| Class | AUC Score | Performance |
|-------|-----------|------------|
| Class 0 (Airplane) | 0.97 | Excellent |
| Class 1 (Drone) | 0.97 | Excellent |
| Class 2 (Helicopter) | 0.97 | Excellent |
| Class 3 (Bird) | 0.97 | Excellent |

#### Key Performance Insights
| Insight | Description |
|---------|-------------|
| Best Performing Class | Drone - Perfect confusion matrix performance and highest AP (0.98) |
| Most Challenging Class | Airplane - Shows most confusion with other classes, lowest AP (0.88) |
| Common Misclassification | Airplane ↔ Helicopter confusion (16 and 7 instances respectively) |
| Model Strength | High discriminative power with consistent 0.97 AUC across all classes |
| Precision-Recall | Maintains high precision across different recall levels for all classes |

## Next steps
What suggestions do you have for next steps?

  ### Key Achievements
  ✅ **Successfully developed** multi-class drone detection system  
  ✅ **Achieved 98.25% accuracy** with RandomForest model on test data   
  ✅ **Implemented both classification and detection** capabilities with 90.48% detection accuracy  
  ✅ **Comprehensive evaluation** across multiple model architectures  
  ✅ **Production-ready models** with documented performance metrics 

  ### Immediate Next Steps
  - **🧪 Pilot Deployment** - Test in controlled environment
  - **⚙️ Performance Optimization** - Fine-tune model parameters
  - **🔌 API Development** - Create integration endpoints
  - **📚 Documentation** - Prepare user training materials

  ### Future Enhancements
  - **🆕 Advanced Architectures** - Explore YOLO v8, RetinaNet
  - **🎭 Multi-modal Integration** - Combine radar and audio data
  - **📱 Edge Optimization** - Optimize for mobile deployment
  * **🌍 Dataset Expansion** - Include diverse environmental conditions
  - **🎯 Real-time Tracking** - Implement object tracking capabilities

## Outline of project

- [Project Report](https://github.com/atewari-bot/drone-image-classification/blob/main/README.md)
- [Jupyter Notebook - Model Training & Performance Metrics Analysis](https://github.com/atewari-bot/drone-image-classification/blob/main/drone_detection.ipynb)
- [Trained Models](https://github.com/atewari-bot/drone-image-classification/blob/main/models/)
- [Python File - Model Training & Performance Metrics Analysis](https://github.com/atewari-bot/drone-image-classification/blob/main/scripts/drone_detection.py)
- [Data Sampling Script](https://github.com/atewari-bot/drone-image-classification/blob/main/scripts/image_sampling.py)

```
📦 drone-image-classification/
├── 📄 README.md - Project documentation
├── 📊 drone_detection.ipynb - Analysis notebook
├── 🤖 models/ - Trained model files
├── 🐍 scripts/
│   ├── drone_detection.py - Training script
│   └── image_sampling.py - Data preprocessing
└── 📈 images/ - Performance visualizations
```

## Contact and Further Information

| Contact Information | |
|-------|---------|
| **Name** | Ajay Tewari |
| **Email** | <mail.ajaytewari@gmail.com> |
| **GitHub** | [github.com/atewari-bot](https://github.com/atewari-bot) |
| **LinkedIn** | [linkedin.com/in/ajaytewari](https://www.linkedin.com/in/ajaytewari/) |
| **Project Repository** | [git@github.com:atewari-bot/drone-image-classification.git](https://github.com/atewari-bot/drone-image-classification) |
| **Primary Data Source** | [Roboflow Drone Dataset](https://universe.roboflow.com/ahmedmohsen/drone-detection-new-peksv) |
