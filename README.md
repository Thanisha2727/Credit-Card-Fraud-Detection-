# Credit Card Fraud Detection App

A mobile-friendly application built with KivyMD that detects credit card fraud using machine learning. The app provides an intuitive interface for loading CSV datasets, training Random Forest models, and visualizing classification results.

## Features

- **File Management**: Easy CSV file loading with built-in file browser
- **Data Processing**: Automatic data balancing and feature scaling
- **Machine Learning**: Random Forest classifier for fraud detection
- **Visualization**: Interactive plots showing classification metrics
- **Dark Theme**: Modern dark UI optimized for mobile devices

## Screenshots

The app consists of three main screens:
- **Main Screen**: File loading and model training controls
- **Results Screen**: Classification report with precision, recall, and F1-scores
- **Plot Screen**: Bar chart visualization of model performance metrics

## Requirements

### Python Dependencies
```
kivymd>=1.1.1
kivy>=2.1.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

### System Requirements
- Python 3.8 or higher
- Android 5.0+ (for mobile deployment)
- 2GB RAM minimum
- 100MB storage space

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
```

2. Install required dependencies:
```bash
pip install kivymd pandas scikit-learn matplotlib
```

3. Run the application:
```bash
python main.py
```

## Usage

### 1. Load Dataset
- Tap "Load CSV File" to open the file manager
- Navigate to your fraud detection dataset
- Select a CSV file with the following structure:
  - Features: V1, V2, ..., V28 (PCA transformed features)
  - Time: Transaction time
  - Amount: Transaction amount
  - Class: Target variable (0 = Genuine, 1 = Fraud)

### 2. Train Model
- After loading data, tap "Train Model"
- The app will automatically:
  - Scale the Time and Amount features
  - Balance the dataset (equal fraud/genuine samples)
  - Split data into training/testing sets (80/20)
  - Train a Random Forest classifier
  - Generate classification metrics

### 3. View Results
- Review precision, recall, and F1-scores for both classes
- Tap "Show Plot" to visualize the metrics
- Use "Back" buttons to navigate between screens

## Dataset Format

The app expects a CSV file with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| Time | Seconds elapsed since first transaction | Float |
| V1-V28 | PCA transformed features | Float |
| Amount | Transaction amount | Float |
| Class | 0 = Genuine, 1 = Fraud | Integer |

### Example Dataset
You can use the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Data Balancing**: Undersampling of majority class
- **Feature Scaling**: StandardScaler for Time and Amount
- **Train/Test Split**: 80/20 ratio
- **Random State**: 42 (for reproducibility)

## Architecture

```
FraudDetectionApp/
├── MainScreen (File loading and training)
├── ResultScreen (Classification metrics)
├── PlotScreen (Visualization)
└── ScreenManager (Navigation)
```

## Error Handling

The app includes error handling for:
- Invalid CSV file formats
- Missing required columns
- Memory issues with large datasets
- Model training failures
- File access permissions

## Performance

- **Training Time**: ~5-30 seconds (depending on dataset size)
- **Memory Usage**: ~50-200MB (depending on dataset)
- **Supported Dataset Size**: Up to 1M transactions
- **Model Accuracy**: Typically 85-95% on balanced data

## Troubleshooting

### Common Issues

**"Error loading file"**
- Ensure CSV has required columns (Time, Amount, Class, V1-V28)
- Check file permissions
- Verify file is not corrupted

**"Error training model"**
- Ensure dataset has both fraud (Class=1) and genuine (Class=0) samples
- Check available memory (large datasets may cause issues)
- Verify numeric data types in all feature columns

**Blank plot screen**
- Ensure model training completed successfully
- Check matplotlib backend compatibility
- Restart app if visualization fails

## Mobile Deployment

To deploy on Android:
1. Install Buildozer: `pip install buildozer`
2. Initialize: `buildozer init`
3. Build APK: `buildozer android debug`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KivyMD team for the excellent UI framework
- Scikit-learn contributors for machine learning tools
- Credit card fraud detection research communit
