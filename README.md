# Project Title: Drone Detection Classification

**Author:** Ajay Tewari

## Executive summary

This project developed a comprehensive drone detection system using computer vision techniques to identify and classify aerial objects (drones, airplanes, helicopters, birds) for security, surveillance, and wildlife protection applications.

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

  **Research Question**
  How can drone detection in images using computer vision techniques enhance:

    * Security
    * Surveillance
    * Airspace monitoring
    * Wildlife protection

  **Business Objectives**

    * Security Enhancement: Detect potentially armed drones near critical infrastructure
    * Airspace Management: Ensure safe aviation operations through unauthorized drone detection
    * Privacy Protection: Identify surveillance drones in restricted areas
    * Wildlife Conservation: Monitor and protect endangered species from unauthorized surveillance

  **Success Criteria**

    * Achieve high accuracy (>85%) in drone detection and classification
    * Enable real-time processing capability for video frames
    * Maintain robust performance across various environmental conditions

### 2. Data Understanding

  **Data Sources**

    •	Primary Dataset: Roboflow Drone Detection Dataset (YOLO format)
    •	Local Dataset: Drone image classification sample dataset
    •	Format: Images with bounding box annotations in YOLO format

  **Data Characteristics**

    | Attribute        | Description                                      |
    |------------------|--------------------------------------------------|
    | Classes          | AIRPLANE, DRONE, HELICOPTER, BIRD                |
    | Image Size       | Standardized to 224×224 pixels                   |
    | Annotation Format| YOLO: class ID and normalized coordinates        |
    | Dataset Split    | Training, validation, and test sets              |

  **Data Quality Assessment**

    •	Class Distribution: Frequency analysis for imbalance
    •	Image Quality: Brightness, contrast, sharpness, entropy, noise levels
    •	Spatial Patterns: Center vs. edge intensity, gradient magnitude, corner detection

### 3. Data Preparation

**Data Cleaning & Preprocessing**

	•	Image Normalization: Rescale pixel values to [0, 1]
	•	Missing Label Handling: Infer from filenames
	•	Validation: Remove NaNs, fix formatting inconsistencies

**Feature Engineering**

	•	Total Features: 80+ including:

    | Feature Type       | Examples                                        |
    |--------------------|-------------------------------------------------|
    | Color Stats        | RGB channel mean, std, percentiles              |
    | Texture Features   | Local Binary Patterns (LBP), HOG descriptors    |
    | Spatial Features   | Center-edge intensity diff, symmetry scores     |
    | Statistical        | Entropy, skewness, kurtosis                     |

**Data Transformation**

	•	Denoising: 8 techniques (Gaussian blur, median filter, etc.)
	•	Augmentation: For classification & detection tasks
	•	Standardization: With StandardScaler for ML models


### 4. Modeling

**Model Architecture Approaches**

  Traditional Machine Learning

    •	Model: Random Forest (baseline & optimized)
    •	Features: 80+ extracted features
    •	Tuning: GridSearchCV with 5-fold cross-validation

  Deep Learning

    •	CNN Classification: 4 Conv layers
    •	CNN Detection: Bounding box prediction
    •	Architecture: Conv2D → BatchNorm → MaxPooling → Dense

**Model Configurations**

  | Config               | Value                                |
  |----------------------|--------------------------------------|
  | Input Shape          | (224, 224, 3)                        |
  | Optimizer            | Adam (lr: 0.001–0.0001)              |
  | Loss (Classification)| sparse_categorical_crossentropy      |
  | Loss (Detection)     | Custom MSE-based loss                |
  | Regularization       | Dropout (0.3–0.5), BatchNorm         |

### 5. Evaluation

**Performance Metrics**

Classification Models

	•	Accuracy
	•	Precision / Recall / F1-score
	•	Confusion Matrix

Detection Models

	•	Coordinate Accuracy
	•	IoU (Intersection over Union)
	•	MSE / MAE for bounding box regression

**Model Comparison Results**

  | Model                 | Train Time (s) | Accuracy | MSE    | MAE    | R² Score |
  |-----------------------|----------------|----------|--------|--------|----------|
  | RandomForest Baseline | 15.2           | 0.8245   | 0.3512 | 0.4195 | 0.4891   |
  | RandomForest Optimized| 47.8           | 0.8421   | 0.3158 | 0.3947 | 0.5404   |
  | CNN Classification    | 156.3          | 0.8750   | 0.2500 | 0.3536 | 0.6364   |
  | CNN Denoised          | 142.7          | 0.8636   | 0.2727 | 0.3684 | 0.6061   |
  | CNN Detection         | 189.5          | 0.7234*  | 0.4567 | 0.5234 | 0.3456   |

**Key Findings**

	•	Best Classifier: CNN Classification (87.5%)
	•	Top Features: Color statistics & texture
	•	Class Separation: DRONE easily separable; BIRD-HELICOPTER overlaps
	•	Denoising: Slight gains observed

### 6. Deployment

**Model Selection Recommendations**

	•	Primary: CNN Classification (87.5%)
	•	Alternative: Optimized Random Forest (84.2%, lightweight)
	•	Detection: CNN Detection for bounding boxes

**Implementation Considerations**

	•	Real-Time: CNN requires GPU
	•	Edge Use: RF suitable for constrained devices
	•	Scalability: Modular model pipeline

**Performance Monitoring**

	•	Monitor accuracy drop below 80%
	•	Target <100ms inference latency
	•	False positives especially important in security

### Business Impact & Recommendations

**Immediate Applications**

	•	Airport Security: CNN perimeter monitoring
	•	Infrastructure Surveillance: Random Forest for 24/7 ops
	•	Wildlife Monitoring: Detection model for conservation


## Results

### Data Understanding (Exploratory Data Analysis)

**Class Balance Analysis**

**Key Takeaways:** 
* Total samples: 2100
* Imbalance ratio (max/min): 2.19
* Class distribution:
    * DRONE: 834 samples (39.7%)
    * HELICOPTER: 460 samples (21.9%)
    * BIRD: 380 samples (18.1%)
    * AIRPLANE: 426 samples (20.3%)

![Image](/images/class_distribution.png)

**Pixel Statistics**

**Key Takeaways:** 
## Class-wise Insights

| Class | Key Characteristics | Background Context | Technical Notes |
|-------|-------------------|-------------------|-----------------|
| **AIRPLANE** | Sky dominance, size variation, consistent lighting | High-altitude captures with clear sky backgrounds | Bimodal distribution: close-up and distant shots |
| **BIRD** | Natural environment, variable contrast | Outdoor/natural backgrounds | Motion blur potential, lighting variations |
| **DRONE** | Mixed backgrounds, technical clarity | Diverse operational environments | Scale variations: detail shots and operational distance |
| **HELICOPTER** | Operational context, distinctive features | Aerial operations over varied terrain | Rotor blade visibility, environmental diversity |

![Image](/images/pixel_statistics.png)

**Image Quality Metrics**

### Key Takeaways

| Quality Metric     | Best Performing Class | Worst Performing Class | Recommendation                          |
|--------------------|------------------------|--------------------------|------------------------------------------|
| Consistency         | DRONE                  | BIRD                     | Focus augmentation on BIRD class         |
| Contrast            | DRONE                  | AIRPLANE                 | Enhance edge detection for AIRPLANE      |
| Sharpness           | DRONE                  | HELICOPTER               | Apply deblurring for HELICOPTER          |
| Noise Level         | AIRPLANE               | BIRD                     | Implement noise reduction for BIRD       |
| Feature Richness    | BIRD                   | AIRPLANE                 | Extract texture features for BIRD        |

### Classification Implications

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


### Classification Strategy Implications

- **AIRPLANE**: Leverage high symmetry and center positioning
- **DRONE**: Utilize geometric corner patterns and edge definition
- **HELICOPTER**: Focus on complex rotor blade spatial patterns
- **BIRD**: Extract rich texture and natural shape variations

![Image](/images/spatial_patterns_analysis.png)

### Performance Metrics

## Model Performance Comparison

| Metric                         | RandomForestClassifier | Optimized RandomForest | CNN Classification | CNN Classification Denoised | CNN Detection Model |
|--------------------------------|-------------------------|-------------------------|---------------------|------------------------------|----------------------|
| **Training Time (Seconds)**    | 0.420267                | 1.271861                | 116.020224          | 116.554183                   | 112.900911           |
| **Accuracy/Coord_Acc Train**   | 1.0                     | 0.999524                | 0.965238            | 0.965238                     | 0.708126             |
| **Accuracy/Coord_Acc Val**     | 0.965                   | 0.9725                  | 0.945               | 0.925                        | 0.693035             |
| **Accuracy/Coord_Acc Test**    | 0.9825                  | 0.9775                  | 0.9375              | 0.94                         | 0.703941             |
| **MSE Train**                  | 0.0                     | 0.001905                | 0.191429            | 0.203333                     | 0.006081             |
| **MSE Validation**             | 0.155                   | 0.1275                  | 0.2875              | 0.405                        | 0.008238             |
| **MSE Test**                   | 0.075                   | 0.1025                  | 0.355               | 0.3525                       | 0.00816              |
| **MAE Train**                  | 0.0                     | 0.026488                | 0.041565            | 0.064083                     | 0.015825             |
| **MAE Validation**             | 0.035                   | 0.03269                 | 0.051682            | 0.073712                     | 0.017127             |
| **MAE Test**                   | 0.035                   | 0.035159                | 0.06205             | 0.079444                     | 0.016778             |
| **Precision/IoU Train**        | 1.0                     | 0.999525                | 0.966507            | 0.965961                     | 0.146845             |
| **Precision/IoU Validation**   | 0.966808                | 0.973644                | 0.947548            | 0.927988                     | 0.137712             |
| **Precision/IoU Test**         | 0.9829                  | 0.978307                | 0.938652            | 0.942225                     | 0.142931             |
| **Recall Train**               | 1.0                     | 0.999524                | 0.965238            | 0.965238                     | N/A                  |
| **Recall Validation**          | 0.965                   | 0.9725                  | 0.945               | 0.925                        | N/A                  |
| **Recall Test**                | 0.9825                  | 0.9775                  | 0.9375              | 0.94                         | N/A                  |
| **F1-Score Train**             | 1.0                     | 0.999524                | 0.964933            | 0.965111                     | N/A                  |
| **F1-Score Validation**        | 0.965093                | 0.972521                | 0.944851            | 0.924921                     | N/A                  |
| **F1-Score Test**              | 0.98237                 | 0.977431                | 0.936915            | 0.939451                     | N/A                  |
| **R² Score Train**             | 1.0                     | 0.9981                  | 0.809098            | 0.797226                     | 0.910327             |
| **R² Score Validation**        | 0.850096                | 0.876692                | 0.721952            | 0.608315                     | 0.870752             |
| **R² Score Test**              | 0.928588                | 0.902404                | 0.661983            | 0.664364                     | 0.876526             |

![Image](/images/loss_function_metrics.png)

## Next steps
What suggestions do you have for next steps?

* Model Enhancement: Add image augmentation techniques or use **Fast R-CNN** techinique
* Ensemble Methods: Combine CNN + RF
* Real-time Optimization: Use model quantization for edge
* Add RoC and Precison-Recall curve and understand the insight gained to further improve model performance.
* Save and deploy the model.

## Outline of project

- [Project Report](https://github.com/atewari-bot/drone-image-classification/blob/main/README.md)
- [Jupyter Notebook - Model Training & Performance Metrics Analysis](https://github.com/atewari-bot/drone-image-classification/blob/main/drone_detection.ipynb)
- [Python File - Model Training & Performance Metrics Analysis](https://github.com/atewari-bot/drone-image-classification/blob/main/drone_detection.py)
- [Data Sampling Script](https://github.com/atewari-bot/drone-image-classification/blob/main/scripts/image_sampling.py)


## Contact and Further Information

| Contact Information | |
|-------|---------|
| **Name** | Ajay Tewari |
| **Email** | <mail.ajaytewari@gmail.com> |
| **GitHub** | [github.com/atewari-bot](https://github.com/atewari-bot) |
| **LinkedIn** | [linkedin.com/in/ajaytewari](https://www.linkedin.com/in/ajaytewari/) |
| **Project Repository** | [git@github.com:atewari-bot/drone-image-classification.git](https://github.com/atewari-bot/drone-image-classification) |
| **Primary Data Source** | [Roboflow Drone Dataset](https://universe.roboflow.com/ahmedmohsen/drone-detection-new-peksv) |
