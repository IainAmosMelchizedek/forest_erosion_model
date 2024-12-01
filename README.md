# forest_erosion_model
A comprehensive application for predicting forest cover types and assessing erosion vulnerability using advanced machine learning techniques in R. Includes an interactive Shiny app for real-time predictions and insights.


# Forest Erosion Model

This repository hosts a machine learning-based application for predicting forest cover types and assessing erosion vulnerability based on environmental attributes. The project is implemented in **R**, leveraging advanced statistical techniques, clustering, and classification models.

## Features
- **Erosion Risk Score Calculation**: Computes a risk score using environmental factors like slope, hydrology distance, and soil type.
- **Machine Learning Models**: Implements Random Forest for classification and KMeans for clustering.
- **Shiny Application**: Provides an interactive interface for users to predict forest cover types based on environmental inputs.
- **Data Preprocessing**: Includes normalization, feature engineering, and oversampling with SMOTE for balanced datasets.

## Getting Started

### Prerequisites
- **R** version >= 4.0
- The following R libraries:
  - `dplyr`
  - `caret`
  - `randomForest`
  - `smotefamily`
  - `cluster`
  - `factoextra`
  - `shiny`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/forest_erosion_model.git
2. Open the R project in RStudio.
3. Install the required libraries: install.packages(c("dplyr", "caret", "randomForest", "smotefamily", "cluster", "factoextra", "shiny"))

Running the Shiny App
1. Run the following command in RStudio: shiny::runApp("app.R")
2. Access the application at http://127.0.0.1:7866 in your browser.

File Structure
app.R: The main Shiny application file.
model_pipeline.rds: Saved model pipeline for predictions.
data: Folder for storing sample datasets.
Usage
Input the environmental attributes (e.g., slope, hydrology distance) into the Shiny app.
View the predicted forest cover type and interpret the erosion risk score.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Inspired by the UCI ML Covertype dataset.
Thanks to the open-source R community for providing powerful machine learning libraries.


MIT License

Copyright (c) 2024 Iain Amos Melchizedek 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

