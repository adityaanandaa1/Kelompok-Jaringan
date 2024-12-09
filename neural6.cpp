#include <iostream>
#include <vector>
#include <string>
#include <cmath> // Untuk fungsi matematika
#include <iomanip>
#include <fstream> // Untuk file input
#include <sstream> // Untuk string stream

using namespace std;

// Struktur neural network sederhana
class NeuralNetwork {
    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;
    vector<double> hiddenLayer;
    vector<double> outputLayer;
    double learningRate;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double lr)
        : weightsInputHidden(hiddenSize, vector<double>(inputSize)),
          weightsHiddenOutput(outputSize, vector<double>(hiddenSize)),
          hiddenLayer(hiddenSize, 0.0),
          outputLayer(outputSize, 0.0),
          learningRate(lr) {
        // Inisialisasi bobot dengan nilai acak kecil
        for (auto& row : weightsInputHidden)
            for (auto& w : row)
                w = (rand() % 1000 - 500) / 1000.0;

        for (auto& row : weightsHiddenOutput)
            for (auto& w : row)
                w = (rand() % 1000 - 500) / 1000.0;
    }

    vector<double> predict(const vector<double>& inputs) {
        // Hitung nilai layer tersembunyi
        for (size_t i = 0; i < hiddenLayer.size(); ++i) {
            hiddenLayer[i] = 0.0;
            for (size_t j = 0; j < inputs.size(); ++j) {
                hiddenLayer[i] += inputs[j] * weightsInputHidden[i][j];
            }
            hiddenLayer[i] = tanh(hiddenLayer[i]); // Aktivasi
        }

        // Hitung nilai output
        for (size_t i = 0; i < outputLayer.size(); ++i) {
            outputLayer[i] = 0.0;
            for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[i][j];
            }
            outputLayer[i] = tanh(outputLayer[i]); // Aktivasi
        }

        return outputLayer;
    }

    void train(const vector<vector<double>>& trainingData,
               const vector<vector<double>>& targetData, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < trainingData.size(); ++i) {
                // Forward pass
                auto inputs = trainingData[i];
                auto targets = targetData[i];

                // Hitung hidden layer
                for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                    hiddenLayer[j] = 0.0;
                    for (size_t k = 0; k < inputs.size(); ++k) {
                        hiddenLayer[j] += inputs[k] * weightsInputHidden[j][k];
                    }
                    hiddenLayer[j] = tanh(hiddenLayer[j]);
                }

                // Hitung output layer
                for (size_t j = 0; j < outputLayer.size(); ++j) {
                    outputLayer[j] = 0.0;
                    for (size_t k = 0; k < hiddenLayer.size(); ++k) {
                        outputLayer[j] += hiddenLayer[k] * weightsHiddenOutput[j][k];
                    }
                    outputLayer[j] = tanh(outputLayer[j]);
                }

                // Backpropagation
                vector<double> outputErrors(outputLayer.size());
                for (size_t j = 0; j < outputLayer.size(); ++j) {
                    outputErrors[j] = targets[j] - outputLayer[j];
                }

                vector<double> hiddenErrors(hiddenLayer.size(), 0.0);
                for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                    for (size_t k = 0; k < outputLayer.size(); ++k) {
                        hiddenErrors[j] += outputErrors[k] * weightsHiddenOutput[k][j];
                    }
                }

                // Update weights hidden-output
                for (size_t j = 0; j < outputLayer.size(); ++j) {
                    for (size_t k = 0; k < hiddenLayer.size(); ++k) {
                        weightsHiddenOutput[j][k] += learningRate * outputErrors[j] * hiddenLayer[k];
                    }
                }

                // Update weights input-hidden
                for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                    for (size_t k = 0; k < inputs.size(); ++k) {
                        weightsInputHidden[j][k] += learningRate * hiddenErrors[j] * inputs[k];
                    }
                }
            }
        }
    }
};

vector<vector<double>> readCSV(const string& filename) {
    ifstream file(filename);
    string line;
    vector<vector<double>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        
        data.push_back(row);
    }

    return data;
}

int main() {
    // Load dataset
    auto trainingData = readCSV("weather_data1.csv");

    // Neural network setup
    NeuralNetwork nn(3, 2, 3, 0.1);

    // Input baru dari pengguna
    double newTemp, newHum, newWind;
    cout << "Masukkan suhu hari ini (C): ";
    cin >> newTemp;
    cout << "Masukkan kelembapan hari ini (%): ";
    cin >> newHum;
    cout << "Masukkan kecepatan angin hari ini (km/h): ";
    cin >> newWind;

    // Normalisasi data input
    vector<double> newInput = {newTemp / 50, newHum / 100, newWind / 30};

    // Prediksi cuaca
    vector<double> prediction = nn.predict(newInput);

    // Denormalisasi dan tampilkan hasil prediksi
      double scaleFactor = 0.9;
      double windFactor = 5.0;
      double humidityOffset = 0.68;
    cout << fixed << setprecision(2);
    cout << "\nPrediksi Cuaca:\n";
    cout << "Suhu: " << prediction[0] * 50 << " C\n";
    cout << "Kelembapan: " << max(prediction[1], humidityOffset) * 100 << "%\n";
    cout << "Kecepatan Angin: " << prediction[2] * 30 * windFactor << " km/h\n";

    return 0;
}