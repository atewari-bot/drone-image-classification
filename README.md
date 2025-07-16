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

**Class Distribution**

**Key Takeaways:** 

![Image](/images/class_distribution.png)

**Pixel Statistics**

**Key Takeaways:** 
![Image](/images/pixel_statistics.png)

**Image Quality Metrics**

**Key Takeaways:** 

![Image](/images/image_quality_metrics.png)

**Spatial Patterns Analysis**

**Key Takeaways:** 

![Image](/images/spatial_patterns_analysis.png)

## Next steps
What suggestions do you have for next steps?

* Model Enhancement: Add diverse samples
* Ensemble Methods: Combine CNN + RF
* Real-time Optimization: Use model quantization for edge
* Use **Fast R-CNN** techinique to train the model and compare the performance.
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
