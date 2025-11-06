using Microsoft.ML;
using Microsoft.ML.Data;

namespace StockPricingForecasting
{
    class Program
    {
        // Data model class to represent stock data from CSV file
        public class StockData
        {
            [LoadColumn(0)]
            public string Date { get; set; }

            [LoadColumn(1)]
            public float Open { get; set; }

            [LoadColumn(2)]
            public float High { get; set; }

            [LoadColumn(3)]
            public float Low { get; set; }

            [LoadColumn(4)]
            public float Close { get; set; }
        }

        // Prediction model class to hold the predicted closing price
        public class StockPrediction
        {
            [ColumnName("Score")]
            public float PredictedClose { get; set; }
        }

        static void Main(string[] args)
        {
            // Initialize ML.NET context for machine learning operations
            var mlContext = new MLContext();
            
            // Load stock data from CSV file into a data view
            var dataView = mlContext.Data.LoadFromTextFile<StockData>("stock_data.csv", separatorChar: ',');
            
            // Preview the loaded data to verify it was loaded correctly
            var preview = dataView.Preview();
            foreach(var row in preview.RowView)
            {
                Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            }
         
            // Create ML pipeline:
         // 1. Combine Open, High, Low prices as features
          // 2. Set Close price as the target label to predict
            // 3. Use FastTree regression algorithm for training
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(StockData.Open), nameof(StockData.High), nameof(StockData.Low))
                .Append(mlContext.Transforms.CopyColumns("Label", nameof(StockData.Close)))
                .Append(mlContext.Regression.Trainers.FastTree());

            // Split data into training (80%) and testing (20%) sets
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
     
            // Train the model using the training data
            var model = pipeline.Fit(trainTestData.TrainSet);
            
            // Make predictions on the test data
            var predictions = model.Transform(trainTestData.TestSet);
      
            // Evaluate model performance using regression metrics
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
      
            // Convert predictions back to enumerable format for comparison
            var predictionResults = mlContext.Data.CreateEnumerable<StockPrediction>(predictions, reuseRowObject: false);
            var testData = mlContext.Data.CreateEnumerable<StockData>(trainTestData.TestSet, reuseRowObject: false);
    
            // Display actual vs predicted closing prices for each test data point
            foreach (var (predicted, actual) in predictionResults.Zip(testData, (p,a) => (p, a)))
            {
                Console.WriteLine($"Actual Close Price: {actual.Close}, Predicted Close Price: {predicted.PredictedClose}");
            }
        }
    }
}