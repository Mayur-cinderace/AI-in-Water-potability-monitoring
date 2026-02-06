# 💧 AI in Water Potability Monitoring

An intelligent machine learning system for predicting water potability using advanced water quality analysis. This project combines data science with a user-friendly Streamlit dashboard to assess whether water sources are safe for consumption.

## 🎯 Project Overview

Water potability assessment is critical for public health. This project uses machine learning (Random Forest Classifier) to predict water safety based on multiple water quality parameters collected from monitoring stations across India. The system can:

- **Predict water safety** in real-time for single water samples
- **Batch process** multiple water samples from CSV files
- **Provide confidence scores** for each prediction
- **Track prediction history** during a session
- **Generate detailed reports** with full water quality metrics

## ✨ Features

- 🤖 **Machine Learning Model**: Trained Random Forest Classifier with 200 estimators
- 📊 **Interactive Dashboard**: Built with Streamlit for intuitive user experience
- 📈 **Real-time Predictions**: Instant water potability assessment
- 📁 **Batch Processing**: Upload CSV files for multiple predictions
- 📥 **Export Results**: Download prediction results and detailed records
- 🎨 **Visual Analytics**: Charts, graphs, and session history tracking
- ⚠️ **Safety Alerts**: Audio and visual alerts for unsafe water conditions
- 🌍 **Metadata Tracking**: Station code, location, season, water source type, and more

## 🏗️ Project Structure

```
├── model.py                      # Model training and evaluation
├── implement.py                  # Streamlit dashboard application
├── requirements.txt              # Python dependencies
├── sample_batch.csv              # Sample data for batch predictions
├── rf_model_indianwater.pkl      # Pre-trained Random Forest model
├── scaler_indianwater.pkl        # Feature scaling preprocessor
├── imputer_indianwater.pkl       # Missing value imputation object
├── features_indianwater.pkl      # Feature list
├── alert.mp3                     # Alert sound file (optional)
└── README.md                     # Project documentation
```

## 📋 Water Quality Parameters

The model analyzes 8 key water quality indicators:

| Parameter | Unit | Range | Safety Threshold |
|-----------|------|-------|------------------|
| Temperature (Temp) | °C | 5-50 | Flexible |
| Dissolved Oxygen (DO) | mg/l | 0-15 | ≥ 5.0 |
| pH | - | 0-14 | 6.5-8.5 |
| Conductivity | µS/cm | 50-2500 | ≤ 2500 |
| Biochemical Oxygen Demand (BOD) | mg/l | 0-20 | ≤ 3.0 |
| Nitrate/Nitrite | mg/l | 0-50 | ≤ 50 |
| Fecal Coliform | MPN/100ml | 0-10000 | ≤ 2500 |
| Total Coliform | MPN/100ml | 0-10000 | ≤ 10000 |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mayur-cinderace/AI-in-Water-potability-monitoring.git
   cd AI-in-Water-potability-monitoring
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify required model files**
   Ensure the following files are in the project root directory:
   - `rf_model_indianwater.pkl`
   - `scaler_indianwater.pkl`
   - `imputer_indianwater.pkl`
   - `features_indianwater.pkl`

### Running the Application

Start the Streamlit dashboard:

```bash
streamlit run implement.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Single Sample Prediction

1. **Enter Location Information**
   - Select State (Karnataka, Maharashtra, Delhi, Tamil Nadu, or Other)
   - Enter Station Code and Location name
   - Specify Year and Water Source Type (River, Groundwater, Lake, etc.)
   - Select Season (Summer, Monsoon, Winter)
   - Indicate if water is pre-treated

2. **Input Water Quality Parameters**
   - Adjust slider values for each of the 8 parameters
   - Default values are pre-populated based on typical ranges

3. **Get Prediction**
   - View instant prediction: SAFE ✅ or UNSAFE ⚠️
   - See confidence score as a percentage
   - Visual alerts for unsafe water (audio, flashing banner)

4. **Analyze Results**
   - View bar chart of input feature values
   - Check prediction history and counts
   - Download the last record as CSV

### Batch Prediction

1. **Prepare CSV File**
   - Format: Must have these column headers:
     ```
     Temp,D.O. (mg/l),PH,CONDUCTIVITY,B.O.D. (mg/l),NITRATENAN N+ NITRITENANN (mg/l),FECAL COLIFORM (MPN/100ml),TOTAL COLIFORM (MPN/100ml)Mean
     ```
   - Use the provided `sample_batch.csv` as a template

2. **Upload File**
   - Click "Upload CSV" in the sidebar
   - Select your CSV file

3. **View & Download Results**
   - Results table shows predictions and confidence scores
   - Download processed results with predictions

## 📊 Model Architecture

The machine learning pipeline consists of:

**Data Preprocessing**
- Feature selection from raw water quality data
- Missing value imputation using median strategy
- Feature scaling with StandardScaler

**Model Training**
- Algorithm: Random Forest Classifier
- Parameters:
  - Number of estimators: 200
  - Random state: 42
  - Test-train split: 80-20

**Potability Classification**
- **SAFE**: Water meets all safety criteria
  - pH: 6.5 - 8.5
  - DO: ≥ 5.0 mg/l
  - BOD: ≤ 3.0 mg/l
  - Fecal Coliform: ≤ 2500 MPN/100ml
- **UNSAFE**: Water fails one or more criteria

## 📦 Dependencies

```
streamlit>=1.25.0      # Web framework
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scikit-learn>=1.2.0    # Machine learning
joblib>=1.2.0          # Model serialization
matplotlib>=3.6.0      # Visualization
```

See `requirements.txt` for complete dependency list with versions.

## 🔄 Workflow

1. **Training Phase** (`model.py`)
   - Loads raw water quality data
   - Cleans and normalizes columns
   - Imputes missing values
   - Trains Random Forest classifier
   - Saves model and preprocessors

2. **Deployment Phase** (`implement.py`)
   - Loads pre-trained model and preprocessors
   - Accepts single samples or batch data
   - Applies same preprocessing pipeline
   - Makes predictions with confidence scores
   - Presents results via interactive UI

## 📊 Sample Data

Example prediction using `sample_batch.csv`:

| Temperature | DO | pH | Conductivity | BOD | NitrateNitrite | FecalColiform | Prediction | Confidence |
|-------------|----|----|--------------|-----|----------------|---------------|------------|-----------|
| 19.4°C | 5.2 | 6.1 | 357 | 3.4 | 7.9 | 3890 | UNSAFE | 0.8234 |
| 33.8°C | 4.5 | 8.9 | 526 | 1.6 | 2 | 646 | SAFE | 0.9156 |
| 28.3°C | 10.9 | 8.5 | 835 | 2.2 | 5.1 | 2888 | SAFE | 0.8792 |

## 🛠️ Configuration

Key features in `implement.py` can be customized:

- **Supported States**: Modify the state selectbox options
- **Station Metadata**: Add/remove location fields
- **Feature Ranges**: Adjust slider min/max values in the feature input section
- **Alert Sound**: Replace `alert.mp3` with your custom audio file
- **Styling**: Modify CSS and HTML in st.markdown() calls

## ⚠️ Alerts and Notifications

**Visual Feedback**
- ✅ Green success banner for SAFE predictions
- ⚠️ Red error banner for UNSAFE predictions
- 📈 Real-time confidence score display

**Audio & Behavioral Alerts** (When unsafe)
- Plays `alert.mp3` on loop (if file exists)
- Flashing red banner animation
- Warning message with remediation suggestions

## 📈 Session Analytics

The dashboard automatically tracks:
- **Prediction History**: Line chart of confidence scores
- **Prediction Counts**: Bar chart of SAFE vs UNSAFE predictions
- **Session State**: Persistent data across user interactions

## 🔒 Data Privacy & Usage

- **No Data Storage**: Predictions are not permanently stored locally
- **Batch Processing**: Download results for your records
- **CSV Export**: All results can be exported for reporting

## 🚦 Future Enhancements

- Database integration for historical tracking
- Advanced anomaly detection
- Multi-region model comparison
- API endpoint deployment
- Mobile app integration
- Real-time monitoring station feeds

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Streamlit for rapid development
- Scikit-learn for robust machine learning models
- Indian water quality monitoring data research
- Water quality assessment standards from WHO and EPA

---

**Last Updated**: February 2026
**Version**: 1.0.0
**Status**: Active Development