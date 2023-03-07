
namespace Quantum.LogRegQ {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Primitive.IO;

    // Function to split the data into training and test sets
    function SplitData(features : Double[][], labels : Double[]) : (Double[][], Double[], Double[][], Double[]) {
        let numExamples = features.Count;
        let numTrainExamples = numExamples * 4 / 5;
        let trainFeatures = features[0 .. numTrainExamples - 1];
        let trainLabels = labels[0 .. numTrainExamples - 1];
        let testFeatures = features[numTrainExamples ..];
        let testLabels = labels[numTrainExamples ..];
        return (trainFeatures, trainLabels, testFeatures, testLabels);
    }

    // Function to train the logistic regression model
    function TrainModel(trainFeatures : Double[][], trainLabels : Double[]) : Double[] {
        return LogisticRegression(trainFeatures, trainLabels);
    }

    // Function to test the logistic regression model
    function TestModel(model : Double[], testFeatures : Double[][], testLabels : Double[]) : Double {
        let numCorrect = 0;
        let numExamples = testFeatures.Count;
        for i in 0 .. numExamples - 1 {
            let predictedLabel = PredictLabel(model, testFeatures[i]);
            let actualLabel = testLabels[i];
            if (predictedLabel > 0.5 && actualLabel == 1.0) {
                set numCorrect += 1;
            } elif (predictedLabel <= 0.5 && actualLabel == 0.0) {
                set numCorrect += 1;
            }
        }
        return Double(numCorrect) / Double(numExamples);
    }

    // Entry point function for the program
    @EntryPoint()
    operation Main() : Unit {
        // Load the data from the CSV file
        let (features, labels) = LoadData("testreg.csv");

        // Split the data into training and test sets
        let (trainFeatures, trainLabels, testFeatures, testLabels) = SplitData(features, labels);

        // Train the logistic regression model
        let model = TrainModel(trainFeatures, trainLabels);

        // Test the logistic regression model
        let accuracy = TestModel(model, testFeatures, testLabels);

        Message($"Accuracy: {accuracy}");
    }
    // Function to load data from a CSV file
    function LoadData(filename : String) : (Double[][] , Double[]) {
        // Load the file and parse its contents
        let file = File.ReadAllLines(filename);
        let data = file[1..];
        let features : Double[][] = [];
        let labels : Double[] = [];

        // Parse the features and labels from the data
        for row in data {
            let items = row.Split(',');
            let featureRow = [ Double.Parse(items[i]) | i in 0 .. items.Length - 2 ];
            let label = Double.Parse(items[items.Length - 1]);
            set features += [featureRow];
            set labels += [label];
        }

        return (features, labels);
    }

    // Function to encode the input data using quantum gates
    function EncodeInput(features : Double[][]) : (Qubit[][], Qubit[]) {
        // Initialize qubits for the encoding
        let numQubits = features[0].Length;
        let inputQubits = [ Qubit() | i in 0 .. numQubits - 1 ];
        let labelQubit = Qubit();

        // Encode the input features using quantum gates
        for i in 0 .. numQubits - 1 {
            let feature = [ features[j][i] | j in 0 .. features.Count - 1 ];
            let amplitude = Sqrt(feature.Sum(x => x*x) / features.Count);
            let angle = 2.0 * ArcSin(amplitude);
            Ry(angle, inputQubits[i]);
            (ControlledOnBitString(feature, X))(labelQubit, inputQubits[i]);
        }

        // Encode the label using a quantum gate
        let label = labels[0];
        let labelAngle = 2.0 * ArcSin(Sqrt(label));
        Ry(labelAngle, labelQubit);

        return (inputQubits, [labelQubit]);
    }

    // Function to perform logistic regression using quantum gates
    operation LogisticRegression(features : Double[][], labels : Double[]) : Double[] {
        // Encode the input data using quantum gates
        let (inputQubits, labelQubits) = EncodeInput(features);

        // Initialize parameters for the logistic regression model
        let numQubits = features[0].Length;
        mutable theta = new Double[numQubits + 1];
        let learningRate = 0.1;
        let numIterations = 100;

        // Perform gradient descent to train the model
        for i in 0 .. numIterations - 1 {
            // Calculate the predicted label using the current parameters
            let predictedLabel = 0.0;
            for j in 0 .. numQubits - 1 {
                let amplitude = Ry(theta[j], inputQubits[j]);
                predictedLabel += amplitude;
            }
            predictedLabel += theta[numQubits];

            // Update the parameters based on the gradient of the cost function
            let error = predictedLabel - labels[0];
            for j in 0 .. numQubits - 1 {
                let feature = [ features[k][j] | k in 0 .. features.Count - 1 ];
                let gradient = error * feature.Sum(x => x*x) / features.Count;
                theta[j] -= learningRate * gradient * inputQubits[j].GetPauliZ();
            }
            let gradient = error / features.Count;
            theta[numQubits] -= learningRate * gradient;
        }
        // Release the qubits
        ResetAll(inputQubits);
        ResetAll(label
    }
}

