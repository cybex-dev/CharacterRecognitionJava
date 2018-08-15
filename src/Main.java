import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Main {
    final int numIterations = Integer.MAX_VALUE;
    final int NumHiddenNeurons = 10;
    final int NumOutputNeurons = 5;
    final int numHiddenLayerInputs = 105;
    final double eta = 0.1; //learning rate
    final int bias = -1;

    final double accuracyLowerBound = 0.27;
    final double accuracyUpperBound = 0.73;

    //    Hidden layer Weights  + 1 for bias
    double[][] hiddenLayerWeights = new double[NumHiddenNeurons][numHiddenLayerInputs + 1];

    //    Hidden layer Output
    double[] hiddenLayerFnet = new double[NumHiddenNeurons];

    // Output layer weights + 1 for bias
    double[][] outputLayerWeights = new double[NumOutputNeurons][NumHiddenNeurons + 1];

    // Output layer weights
    double[] outputLayerFnet = new double[NumOutputNeurons];

    public Main() throws IOException {
        List<Character> characters = readCharactersFromFile("/char_profiles_alphabet.txt");
        List<Character> testingSet = readCharactersFromFile("/test_char_profiles_alphabet.txt");

        initializeEmptyWeights();

        double previousTestingSSE = Double.MAX_VALUE,
                trainingSSE = Double.MAX_VALUE,
                currentTestingSSE = Double.MAX_VALUE;

        int iterationCount = 0;

        System.out.println("Starting training:");
        while (iterationCount < numIterations &&
                previousTestingSSE >= currentTestingSSE) {

            int testCorrect = 0, trainCorrect = 0;
            trainingSSE = 0.0;
            currentTestingSSE = 0.0;

            System.out.println("Iteration: " + iterationCount);
//            System.out.printf("Training...");
            int trainCharCorrect = 0;
            for (int i = 0; i < characters.size(); i++) {
                Character c = characters.get(i);

                // fnet of inputs to hidden   20 * 106              20
                computeFnet(c.getVectors(), hiddenLayerWeights, hiddenLayerFnet);

                // fnet of hidden to output
                computeFnet(hiddenLayerFnet, outputLayerWeights, outputLayerFnet);

                // delta k (error signal of hidden to output)
                double[] deltaK = computeDeltaK(outputLayerFnet, c.getBinary());

                // delta J (error signal input to hidden)
                double[] deltaJ = computeDeltaJ(hiddenLayerFnet, deltaK, outputLayerWeights);

                // Adjust weights output
                updateWeights(outputLayerWeights, deltaK, hiddenLayerFnet);

                // Adjust weights hidden
                updateWeights(hiddenLayerWeights, deltaJ, c.getVectors());

                // Calculate SSE for pattern p
                trainCorrect = 0;
                for (int j = 0; j < outputLayerFnet.length; j++) {
                    double neuronError = c.getBinary()[j] - outputLayerFnet[j];
                    if ((c.getBinary()[j] == 0.0 && neuronError <= accuracyLowerBound)
                            || (c.getBinary()[j] == 1.0 && neuronError >= accuracyUpperBound))
                        trainCorrect++;

                    trainingSSE += Math.pow(neuronError, 2);
                }

                if (trainCorrect == NumOutputNeurons) {
                    System.out.println("(Train) Correctly identified: " + c.desired);
                    trainCharCorrect++;
                }
//                System.out.printf("Output:\t#1 - %f\t#2 - %f\t#3 - %f\t#4 - %f\t#5 - %f\n", outputLayerFnet[0], outputLayerFnet[1], outputLayerFnet[2], outputLayerFnet[3], outputLayerFnet[4]);
            }

            if (trainCorrect == characters.size())
                System.out.println("All training characters identified");


            // Calculate real SSE
            trainingSSE = (trainingSSE/(characters.size() * NumOutputNeurons));

            // Calculate SSE for all patterns
            System.out.println("Train SSE: " + trainingSSE);

//            Testing
//            System.out.println("Testing:");
//            int testCharCorrect = 0;
//            for (int i = 0; i < testingSet.size(); i++) {
//                // fnet of inputs to hidden   20 * 106              20
//                double[] arrHidden = new double[NumHiddenNeurons];
//                computeFnet(testingSet.get(i).getVectors(), hiddenLayerWeights, arrHidden);
//
//                // fnet of hidden to output
//                double[] arrOutput = new double[NumOutputNeurons];
//                computeFnet(arrHidden, outputLayerWeights, arrOutput);
//
//                testCorrect = 0;
//                for (int j = 0; j < arrOutput.length; j++) {
//                    double neuronError = testingSet.get(i).getBinary()[j] - arrOutput[j];
//                    currentTestingSSE = Math.pow(neuronError, 2);
//                    if (testingSet.get(i).getBinary()[j] == 0.0 && neuronError <= accuracyLowerBound) {
//                        testCorrect++;
//                    } else if (testingSet.get(i).getBinary()[j] == 1.0 && neuronError >= accuracyUpperBound) {
//                        testCorrect++;
//                    }
//                }
//
//                if (testCorrect == NumOutputNeurons) {
//                    System.out.println("(Test) Correctly identified: " + testingSet.get(i).desired);
//                    testCharCorrect++;
//                }
//            }
//
//            if (testCharCorrect == testingSet.size())
//                System.out.println("All testing characters identified");
//
//            previousTestingSSE = trainingSSE;
            iterationCount++;
        }

        if (previousTestingSSE < currentTestingSSE){
            System.out.println("Stoped due to over-fiting");
        } else if (iterationCount == numIterations) {
            System.out.println("Max num iterations reached");
        } else {
            System.out.println("Unknown stopping condition");
        }

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

    private void updateSSE() {

    }

    private void updateWeights(double[][] weightsToUpdate, double[] delta, double[] inputVectors) {
        // 1 offset for bias as bias is first weight
        for (int i = 0; i < delta.length; i++) {
            for (int j = 1; j < weightsToUpdate[i].length; j++) {
                double currentWeight = weightsToUpdate[i][j];
                double weightChange = -1.0 * eta * delta[i] * inputVectors[j - 1];
                weightsToUpdate[i][j] = currentWeight - weightChange;
            }

            // update bias
            int j = 0;
            double currentWeight = weightsToUpdate[i][j];
            double weightChange = -1.0 * eta * delta[i] * bias;
            weightsToUpdate[i][j] = currentWeight - weightChange;

        }
    }

    private double[] computeDeltaJ(double[] hiddenLayerFnet, double[] deltaK, double[][] outputLayerWeights) {
        double[] deltaJ = new double[hiddenLayerFnet.length];
        // for every output neuron getting an input from the current hidden layer neuron
        for (int i = 0; i < hiddenLayerFnet.length; i++) {
            for (int j = 0; j < deltaK.length; j++) {
                double hiddenLayerOutput = hiddenLayerFnet[i];
                deltaJ[i] = deltaK[j] * outputLayerWeights[j][i] * (1 - hiddenLayerOutput) * hiddenLayerOutput;
            }
        }
        return deltaJ;
    }

    private double[] computeDeltaK(double[] outputLayerFnet, int[] characterBinary) {
        double[] deltaK = new double[outputLayerFnet.length];
        for (int i = 0; i < outputLayerFnet.length; i++) {
            double output = outputLayerFnet[i];
            deltaK[i] = -1.0 * (characterBinary[i] - output) * (1 - output) * output;
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
                hiddenLayerWeights[i][i1] = new Random().nextGaussian(); // -1.0 to 1.0
            }
        }

        // initialize weights - output layer
        for (int i = 0; i < outputLayerWeights.length; i++) {
            for (int i1 = 0; i1 < outputLayerWeights[i].length; i1++) {
                outputLayerWeights[i][i1] = new Random().nextGaussian(); // -1.0 to 1.0
            }
        }
    }
}

