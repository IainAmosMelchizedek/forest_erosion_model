# Load required libraries
library(dplyr)
library(caret)
library(randomForest)
library(smotefamily)
library(cluster)
library(factoextra)
library(shiny)

# Logging function
log_message <- function(message) {
  cat(paste(Sys.time(), "- INFO -", message, "\n"))
}

# Fetch dataset
fetch_data <- function() {
  log_message("Fetching dataset...")
  # Simulated data similar to the Covertype dataset
  set.seed(42)
  n <- 1000
  data <- data.frame(
    Slope = runif(n, 0, 45),
    Horizontal_Distance_To_Hydrology = runif(n, 0, 500),
    Vertical_Distance_To_Hydrology = runif(n, -300, 300),
    Elevation = runif(n, 2000, 4000),
    Cover_Type = sample(1:7, n, replace = TRUE)
  )
  for (i in 1:40) {
    data[[paste0("Soil_Type", i)]] <- sample(0:1, n, replace = TRUE)
  }
  log_message("Dataset successfully fetched.")
  return(data)
}

# Normalize a column to the range [0, 1]
normalize <- function(column) {
  return((column - min(column)) / (max(column) - min(column)))
}

# Calculate Erosion Risk Score
calculate_erosion_risk_score <- function(df) {
  log_message("Calculating Erosion Risk Score...")
  
  # Normalize key features
  df <- df %>%
    mutate(
      Slope_Norm = normalize(Slope),
      Hydrology_Ratio_Norm = normalize(Horizontal_Distance_To_Hydrology / abs(Vertical_Distance_To_Hydrology + 1e-5)),
      Elevation_Slope_Norm = normalize(Elevation * Slope)
    )
  
  # Assign soil type weights and calculate soil factor
  set.seed(42)
  soil_weights <- runif(40, 0.1, 0.8) # Random weights for soil types
  soil_factors <- sapply(1:40, function(i) df[[paste0("Soil_Type", i)]] * soil_weights[i])
  df$Soil_Type_Factor <- rowSums(soil_factors)
  
  # Compute erosion risk score
  df$Erosion_Risk_Score <- (
    0.4 * df$Slope_Norm +
      0.3 * df$Hydrology_Ratio_Norm +
      0.2 * df$Soil_Type_Factor +
      0.1 * df$Elevation_Slope_Norm
  )
  
  log_message("Erosion Risk Score calculated successfully.")
  return(df)
}

# Assign KMeans clusters
assign_kmeans_clusters <- function(df, features, n_clusters = 3) {
  log_message("Assigning KMeans clusters...")
  scaled_features <- scale(df[, features])
  kmeans_model <- kmeans(scaled_features, centers = n_clusters, nstart = 25)
  df$KMeans_Cluster <- kmeans_model$cluster
  log_message("Clusters assigned successfully.")
  return(df)
}

# Validate clustering
validate_clustering <- function(df, cluster_column, features) {
  log_message("Validating clustering with Silhouette Score...")
  scaled_features <- scale(df[, features])
  score <- silhouette(df[[cluster_column]], dist(scaled_features))
  log_message(paste("Silhouette Score:", mean(score[, 3])))
}

# Optimize model hyperparameters
optimize_hyperparameters <- function(X_train, y_train) {
  log_message("Starting hyperparameter optimization...")
  
  # Define parameter grid for mtry (number of variables tried at each split)
  param_grid <- expand.grid(mtry = c(1, 2, 3))
  
  # Control object for cross-validation
  control <- trainControl(method = "cv", number = 3)
  
  # Train Random Forest model
  model <- train(
    x = X_train, 
    y = as.factor(y_train),
    method = "rf",
    tuneGrid = param_grid,
    trControl = control,
    ntree = 200
  )
  
  log_message(paste("Best Parameters:", paste(model$bestTune, collapse = ", ")))
  return(model)
}

# Save model pipeline
save_model <- function(model, scaler, filename) {
  log_message(paste("Saving model pipeline to", filename, "..."))
  saveRDS(list(model = model, scaler = scaler), file = filename)
  log_message("Model pipeline saved successfully.")
}

# Main execution
main <- function() {
  # Fetch and preprocess data
  data <- fetch_data()
  features <- c("Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology")
  
  # Calculate Erosion Risk Score
  data <- calculate_erosion_risk_score(data)
  
  # Assign KMeans clusters
  data <- assign_kmeans_clusters(data, features)
  validate_clustering(data, "KMeans_Cluster", features)
  
  # Split data
  X <- data[, features]
  y <- data$Cover_Type
  smote_result <- SMOTE(X, as.factor(y), K = 5, dup_size = 2)
  X_resampled <- smote_result$data[, -ncol(smote_result$data)]
  y_resampled <- smote_result$data[, ncol(smote_result$data)]
  
  train_idx <- createDataPartition(y_resampled, p = 0.8, list = FALSE)
  X_train <- X_resampled[train_idx, ]
  X_test <- X_resampled[-train_idx, ]
  y_train <- y_resampled[train_idx]
  y_test <- y_resampled[-train_idx]
  
  # Hyperparameter tuning
  best_model <- optimize_hyperparameters(X_train, y_train)
  
  # Evaluate model
  predictions <- predict(best_model, X_test)
  log_message("Classification Report:")
  print(confusionMatrix(predictions, as.factor(y_test)))
  
  # Save model
  save_model(best_model, scale(X_resampled), "model_pipeline.rds")
}

# Run the main function
main()

############### Deploy Shiny Application

# Load the saved model pipeline
load_model <- function(filename) {
  pipeline <- readRDS(filename)
  return(pipeline)
}

model_pipeline <- load_model("model_pipeline.rds")

# Check if the model pipeline contains both the model and scaler
if (!("model" %in% names(model_pipeline))) {
  stop("The loaded pipeline does not contain a trained model.")
}

# Define a fallback for scaler if it is not properly structured
if (!("scaler" %in% names(model_pipeline)) || !is.list(model_pipeline$scaler)) {
  warning("Scaler not found or improperly structured. Using unscaled inputs.")
  scale_data <- function(data) data
} else {
  # Define a function to scale data
  scale_data <- function(data) {
    scale(data, center = model_pipeline$scaler$center, scale = model_pipeline$scaler$scale)
  }
}

# Define the Shiny app
ui <- fluidPage(
  titlePanel("Forest Cover Type Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("slope", "Slope:", value = 25, min = 0, max = 90, step = 1),
      numericInput("horiz_dist", "Horizontal Distance to Hydrology:", value = 200, min = 0, max = 1000, step = 10),
      numericInput("vert_dist", "Vertical Distance to Hydrology:", value = 100, min = -500, max = 500, step = 10),
      actionButton("predict", "Predict")
    ),
    
    mainPanel(
      h3("Prediction Result"),
      verbatimTextOutput("prediction_output")
    )
  )
)

server <- function(input, output, session) {
  # Function to make predictions
  predict_cover_type <- reactive({
    req(input$predict) # Ensure prediction only runs when button is clicked
    
    # Create a data frame from user input
    input_data <- data.frame(
      Slope = input$slope,
      Horizontal_Distance_To_Hydrology = input$horiz_dist,
      Vertical_Distance_To_Hydrology = input$vert_dist
    )
    
    # Scale the input data using the saved scaler or fallback
    scaled_data <- as.data.frame(scale_data(input_data))
    
    # Make prediction
    prediction <- predict(model_pipeline$model, newdata = scaled_data)
    
    # Return prediction
    paste("Predicted Cover Type:", prediction)
  })
  
  # Display the prediction
  output$prediction_output <- renderText({
    predict_cover_type()
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
