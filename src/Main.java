import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Main {
    static PrintWriter p;
    int maxIdentifiedTest = 0;
    int maxIdentifiedTrain = 0;
    final int numIterations = 1000;
    final int NumHiddenNeurons = 30;
    final int NumOutputNeurons = 5;
    final int numHiddenLayerInputs = 105;
    final double eta = 0.01; //learning rate
    final int bias = -1;
    final double accuracy = 0.5;
    //   Hidden layer Weights  + 1 for bias
    double[][] hiddenLayerWeights = new double[NumHiddenNeurons][numHiddenLayerInputs + 1];
    //    Hidden layer Output
    double[] hiddenLayerFnet = new double[NumHiddenNeurons];
    // Output layer weights + 1 for bias
    double[][] outputLayerWeights = new double[NumOutputNeurons][NumHiddenNeurons + 1];
    // Output layer weights
    double[] outputLayerFnet = new double[NumOutputNeurons];

    public Main() throws IOException {
        p = new PrintWriter("output.csv");
        List<Character> characters = readCharactersFromFile("/char_profiles_alphabet.txt");
        List<Character> testingSet = readCharactersFromFile("/test_char_profiles_alphabet.txt");

        initializeEmptyWeights();

        double previousTestingSSE = Double.MAX_VALUE,
                trainingSSE = 0.0,
                currentTestingSSE = Double.MAX_VALUE;
        int iterationCount = 0;

        StringBuilder printString = new StringBuilder();

        System.out.println("Starting training:");
        while (iterationCount < numIterations &&
                previousTestingSSE >= currentTestingSSE) {
            previousTestingSSE = currentTestingSSE;
            int testCorrect = 0, trainCorrect = 0;
            trainingSSE = 0.0;
            currentTestingSSE = 0.0;

            System.out.println("Iteration: " + iterationCount);
            int trainCharCorrect = 0;
            printString.append(String.valueOf(iterationCount)).append(",");

            for (int i = 0; i < characters.size(); i++) {
                Character c = characters.get(i);
                trainWithCharacter(c);

                // Calculate SSE for pattern p
                trainCorrect = 0;
                for (int j = 0; j < outputLayerFnet.length; j++) {
                    double neuronError = Math.abs(c.getBinary()[j] - outputLayerFnet[j]);
                    if (c.getBinary()[j] == 0.0 && neuronError <= accuracy) {
                        trainCorrect++;
                    } else if (c.getBinary()[j] == 1.0 && (1-neuronError) <= accuracy) {
                        trainCorrect++;
                    }
                    trainingSSE += Math.pow(neuronError, 2);
                }
                if (trainCorrect == NumOutputNeurons) {
                    trainCharCorrect++;
                }
            }

            if (trainCharCorrect > maxIdentifiedTrain)
                maxIdentifiedTrain = trainCharCorrect;

            if (trainCharCorrect == characters.size()) {
                System.out.println("All training characters identified");
            } else {
                System.out.println("Identified (Train): " + trainCharCorrect);
            }

            printString.append(String.valueOf(trainingSSE)).append(",");
            printString.append(String.valueOf(trainCharCorrect)).append(",");

            int testCharCorrect = 0;
            for (int i = 0; i < testingSet.size(); i++) {
                // fnet of inputs to hidden   20 * 106              20
                double[] arrHidden = new double[NumHiddenNeurons];
                computeFnet(testingSet.get(i).getVectors(), hiddenLayerWeights, arrHidden);

                // fnet of hidden to output
                double[] arrOutput = new double[NumOutputNeurons];
                computeFnet(arrHidden, outputLayerWeights, arrOutput);

                testCorrect = 0;
                for (int j = 0; j < arrOutput.length; j++) {
                    double neuronError = Math.abs(testingSet.get(i).getBinary()[j] - arrOutput[j]);
                    if (testingSet.get(i).getBinary()[j] == 0.0 && neuronError < accuracy) {
                        testCorrect++;
                    } else if (testingSet.get(i).getBinary()[j] == 1.0 && (1-neuronError) < accuracy) {
                        testCorrect++;
                    }
                    currentTestingSSE = Math.pow(neuronError, 2);
                }
                if (testCorrect == NumOutputNeurons) {
                    testCharCorrect++;
                }
            }

            if (testCharCorrect > maxIdentifiedTest)
                maxIdentifiedTest = testCharCorrect;

            if (testCharCorrect == testingSet.size()) {
                System.out.println("All testing characters identified");
            } else {
                System.out.println("Identified (Test): " + testCharCorrect);
            }

            printString.append(String.valueOf(currentTestingSSE)).append(",");
            printString.append(String.valueOf(testCharCorrect)).append(",");
            p.write(printString.toString().concat("\n"));
            p.flush();
            printString = new StringBuilder();

            iterationCount++;
        }
        System.out.println("Max identified train: " + maxIdentifiedTrain);
        System.out.println("Max identified test: " + maxIdentifiedTest);
    }

    private void trainWithCharacter(Character c) {
        // fnet of inputs to hidden   20 * 106              20
        computeFnet(c.getVectors(), hiddenLayerWeights, hiddenLayerFnet);

        // fnet of hidden to output
        computeFnet(hiddenLayerFnet, outputLayerWeights, outputLayerFnet);

        // delta k (error signal of hidden to output)
        double[] deltaK = computeDeltaK(outputLayerFnet, c.getBinary());

        // delta J will be calculated while doing updates, as a summation is to be done using the deltaK error signal calculated above

        // Adjust weights output
        updateWeightsK(outputLayerWeights, deltaK, hiddenLayerFnet);

        // Adjust weights hidden
        updateWeightsJ(hiddenLayerWeights, deltaK, c.getVectors());
    }

    public static void main(String[] args) {
        try {
            new Main();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Character> readCharactersFromFile(String resource) throws IOException {
        List<Character> characters = new ArrayList<>();
        //  Read iterator from characters text file
        InputStream resourceAsStream = getClass().getResourceAsStream(resource);
        BufferedReader br = new BufferedReader(new InputStreamReader(resourceAsStream));
        String line = "";
        Character c = null;
        while ((line = br.readLine()) != null) {
            if (line.length() > 0) {
                if (line.length() == 1) {
                    if (c != null) {
                        characters.add(c);
                    }

                    // Create new char since we have a first letter
                    c = new Character();

                    // Set character desired value
                    c.desired = line;
                } else {
                    c.vectors.addAll(Arrays.stream(line.split(",")).map(Double::parseDouble).collect(Collectors.toList()));
                }
            }
        }
        // Add last letter to character map
        characters.add(c);

        // Return characters
        return characters;
    }

    private void updateWeightsJ(double[][] weightsToUpdate, double[] delta, double[] inputVectors) {
        // 1 offset for bias as bias is first weight
        for (int i = 0; i < weightsToUpdate.length; i++) { // all neurons, where i is a specific neuron
            for (int j = 1; j < weightsToUpdate[i].length; j++) { // all weights of specific neuron, j being a specific weight of the i'th neuron
                double weightChange = 0.0;
                for (int i1 = 0; i1 < delta.length; i1++) {
                    weightChange -= -1.0 * delta[i1] * outputLayerWeights[i1][i] * hiddenLayerFnet[i] * (1-hiddenLayerFnet[i]) * inputVectors[j - 1];
                }
                weightsToUpdate[i][j] = weightsToUpdate[i][j] - (eta * weightChange);
            }
            // update bias
            int j = 0;
            double weightChange = 0.0;
            for (int i1 = 0; i1 < delta.length; i1++) {
                weightChange -= -1.0 * delta[i1] * outputLayerWeights[i1][i] * hiddenLayerFnet[i] * (1-hiddenLayerFnet[i]) * bias;
            }
            weightsToUpdate[i][j] = weightsToUpdate[i][j] - (eta * weightChange);

        }
    }

    private void updateWeightsK(double[][] weightsToUpdate, double[] delta, double[] inputVectors) {
        // 1 offset for bias as bias is first weight
        for (int i = 0; i < weightsToUpdate.length; i++) {
            for (int j = 1; j < weightsToUpdate[i].length; j++) {
                double currentWeight = weightsToUpdate[i][j];
                double weightChange = -1.0 * delta[i] * inputVectors[j - 1];
                weightsToUpdate[i][j] = currentWeight - (eta * weightChange);
            }
            // update bias
            int j = 0;
            double currentWeight = weightsToUpdate[i][j];
            double weightChange = -1.0 * eta * delta[i] * bias;
            weightsToUpdate[i][j] = currentWeight + weightChange;

        }
    }

    private double[] computeDeltaK(double[] outputLayerFnet, int[] characterBinary) {
        double[] deltaK = new double[outputLayerFnet.length];
        for (int i = 0; i < outputLayerFnet.length; i++) {
            double output = outputLayerFnet[i];
            deltaK[i] = (characterBinary[i] - output) * (1 - output) * output;
        }
        return deltaK;
    }

    private void computeFnet(double[] inputVectors, double[][] neuronWeights, double[] outputContainer) {
        for (int i = 0; i < outputContainer.length; i++) {
            double sumOfInputs = 0.0;
            sumOfInputs += bias * inputVectors[0];

            for (int j = 1; j < neuronWeights.length; j++) {
                sumOfInputs += inputVectors[j - 1] * neuronWeights[j][i];
            }
            outputContainer[i] = sigmoid(sumOfInputs);
        }
    }

    private double sigmoid(double sumOfInputs) {
        return 1.0 / (1.0 + (Math.pow(Math.E, -1.0 * sumOfInputs)));
    }

    private void initializeEmptyWeights() {

        // Initalizer hidden fnet container
        for (int i = 0; i < hiddenLayerFnet.length; i++) {
            hiddenLayerFnet[i] = 0.0;
        }

        // Initalizer output fnet container
        for (int i = 0; i < outputLayerFnet.length; i++) {
            outputLayerFnet[i] = 0.0;
        }

        // initialize weights - hidden layer
        for (int i = 0; i < hiddenLayerWeights.length; i++) {
            for (int i1 = 0; i1 < hiddenLayerWeights[i].length; i1++) {
//                hiddenLayerWeights[i][i1] = new Random(1).nextGaussian(); // -1.0 to 1.0
                // Appropriate weight init
                hiddenLayerWeights[i][i1] = generalizedfWeight(new Random(1).nextGaussian()); // -1.0 to 1.0
            }
        }

        // initialize weights - output layer
        for (int i = 0; i < outputLayerWeights.length; i++) {
            for (int i1 = 0; i1 < outputLayerWeights[i].length; i1++) {
//                outputLayerWeights[i][i1] = new Random(1).nextGaussian(); // -1.0 to 1.0
                outputLayerWeights[i][i1] = generalizedfWeight(new Random(1).nextGaussian()); // -1.0 to 1.0
            }
        }
    }

    private double generalizedfWeight(double v) {
        return 1 / Math.sqrt(v) ;
    }
}

