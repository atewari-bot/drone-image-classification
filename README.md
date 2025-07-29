# Project Title: Drone Detection Classification

**Author:** Ajay Tewari

## Executive summary

This project develops a comprehensive computer vision system for automated drone detection in aerial imagery, addressing critical needs in security, surveillance, airspace monitoring, and wildlife protection. Through systematic application of the CRISP-DM methodology, we implemented and evaluated six distinct machine learning models, achieving **98.25% classification accuracy** and **90.5% spatial detection accuracy**.

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
  
  ![Image](/images/metrics/model_portfolio.svg)

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
  | **⚡ Fast R-CNN** | 2707.92s | 90.5%* | 0.0029 | 29.01%* | 0.9971 |

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

### 🔬 Detailed Performance Metrics Across Train/Validation/Test Sets

![Image](/images/metrics/model_performance_metrics.svg) 

### 🔍 Cross-Set Performance Analysis

![Image](/images/metrics/cross_set_performance_analysis.svg)

### 📈 Performance Trends Analysis

![Image](/images/metrics/performance_trend_analysis.png) 

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
  ✅ **Implemented both classification and detection** capabilities with 90.5% detection accuracy  
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
