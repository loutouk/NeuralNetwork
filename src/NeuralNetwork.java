import java.util.Random;

public class NeuralNetwork {

    private Layer[] layers;
    private DataSet dataSet;
    private String errorFunction;

    public NeuralNetwork(int inputNeuronNumber,
                         int hiddenLayerNumber,
                         int hiddenNeuronNumber,
                         int outputNeuronNumber,
                         String errorFunction){

        layers = new Layer[1+hiddenLayerNumber+1];

        this.errorFunction = errorFunction;

        // Init the input layer
        // There is no function but an input
        layers[0] = new Layer(inputNeuronNumber, hiddenNeuronNumber, "");

        // Init the hidden layers
        for(int i=0 ; i<hiddenLayerNumber ; i++){
            // If we are on the last hidden layer, we should connect to the number of output neuron
            if(i+1 == hiddenLayerNumber){
                layers[i+1] = new Layer(hiddenNeuronNumber, outputNeuronNumber, "sigmoid");
            }
            // Otherwise, connect to the next layer
            else{
                layers[i+1] = new Layer(hiddenNeuronNumber, hiddenNeuronNumber, "sigmoid");
            }

        }

        // Init the output Layer
        layers[1+hiddenLayerNumber+1-1] = new Layer(outputNeuronNumber,0, "sigmoid");

        initWeightAndBias();
        initDelta();

    }

    public void setDataSet(DataSet dataSet) throws trainingError {
        // The number of solution output must match the number of neuron output
        if(dataSet.getDatas()[0].getSolution().length != layers[layers.length-1].neuronNumber){
            throw new trainingError("The number of the dataset solution output must match the number of neuron output of the network.");
        }
        this.dataSet = dataSet;
    }

    private void initWeightAndBias(){
//        layers[0].bias[0] = 0.0;
//        layers[0].bias[1] = 0.0;
//
//        layers[1].bias[0] = -0.3;
//        layers[1].bias[1] = 0.4;
//        layers[2].bias[0] = -0.7;
//
//        layers[0].weightsPerNeurons[0][0] = -0.3;
//        layers[0].weightsPerNeurons[1][0] = 0.2;
//        layers[0].weightsPerNeurons[0][1] = -0.8;
//        layers[0].weightsPerNeurons[1][1] = 0.4;
//
//        layers[1].weightsPerNeurons[0][0] = 0.5;
//        layers[1].weightsPerNeurons[1][0] = -0.2;



        // each layer
        for(int i=0 ; i<layers.length ; i++){
            // each neuron
            for(int j=0 ; j<layers[i].weightsPerNeurons.length ; j++){

                // as there is one bias per neuron, we init the bias in the same time
                // the first layer has no bias
                if(i==0){
                    layers[i].bias[j] = 0;
                }else{
                    layers[i].bias[j] = getRandom();
                }
                //System.out.println("Layer " + i + " neuron " + j + " bias " + layers[i].bias[j]);

                // each weight for the current neuron
                for(int k=0 ; k<layers[i].weightsPerNeurons[j].length ; k++){
                    layers[i].weightsPerNeurons[j][k] = getRandom();
                    //System.out.println("Layer " + i + " neuron " + j + " weight " + k + " " + layers[i].weightsPerNeurons[j][k]);
                }

            }
        }
        //System.out.println("********************************************************************");
    }

    private void initDelta(){
        // each layer
        for(int i=0 ; i<layers.length ; i++){
            // each delta neuron
            layers[i].deltaPerNeurons = new double[layers[i].neuronNumber];
            // each neuron
            for(int j=0 ; j<layers[i].deltaPerWeights.length ; j++){
                // each delta weight for the current neuron
                for(int k=0 ; k<layers[i].deltaPerWeights[j].length ; k++){
                    layers[i].deltaPerWeights[j][k] = 0;
                }
            }
        }
    }

    public void train(double learningRate, double targetError, int maxEpochs) throws trainingError {

        if(dataSet == null){
            throw new trainingError("The data set must be initialized before the training.");
        }

        // Iterate over the number of epochs to be completed
        for(int epoch=0 ; epoch<maxEpochs ; epoch++) {
            double totalError = 0.0;
            // Iterate over all training examples
            for(int i=0 ; i<dataSet.getDatas().length; i++){

                // Feed forward
                double[] outputs = forwardPropagation(dataSet.getDatas()[i].getInputs());

                // Compute the errors for each output / solution
                double[] errors = new double[outputs.length];

                for(int error=0 ; error<errors.length ; error++){
                    //System.out.println("Input " + dataSet.getDatas()[i].getInputs()[0] + " " + dataSet.getDatas()[i].getInputs()[1]);
                    //System.out.println("Output " + error + " = " + outputs[error]);
                    errors[error] = applyErrorFunction(outputs[error], dataSet.getDatas()[i].getSolution()[error]);
                    //System.out.println("Error " + error + " = " + errors[error]);
                    totalError += errors[error];
                }


                // Run backpropagation
                backPropagation(dataSet.getDatas()[i].getInputs(), dataSet.getDatas()[i].getSolution(), outputs);

                // update the weights
                updateWeights(learningRate);
            }

            if(totalError < targetError){
                System.out.println("Solution found after " + epoch + " epochs for target error at " + targetError);
                break;
            }
        }

    }

    private double[] forwardPropagation(double[] inputs) throws trainingError {
        // Our first layer start with the data input
        double[] currentInputs = inputs;
        double[] nextLayerInputs;
        for(int currentLayer=0 ; currentLayer<layers.length - 1 ; currentLayer++){

            nextLayerInputs = new double[layers[currentLayer].neuronNumberNextLayer];

            // Compute the input * weight for each combinations of neurons/weights
            for(int currentNeuron=0 ; currentNeuron<layers[currentLayer].weightsPerNeurons.length; currentNeuron++){
                for(int currentWeight=0 ; currentWeight<layers[currentLayer].weightsPerNeurons[currentNeuron].length ; currentWeight++){

                    nextLayerInputs[currentWeight] += layers[currentLayer].weightsPerNeurons[currentNeuron][currentWeight] *
                            currentInputs[currentNeuron];

                }
            }

            // Add the bias
            for(int nextLayerNeuron=0 ; nextLayerNeuron<nextLayerInputs.length ; nextLayerNeuron++){
                nextLayerInputs[nextLayerNeuron] += layers[currentLayer + 1].bias[nextLayerNeuron];
            }

            // Compute the activation function
            for(int nextLayerNeuron=0 ; nextLayerNeuron<nextLayerInputs.length ; nextLayerNeuron++){
                nextLayerInputs[nextLayerNeuron] = layers[currentLayer + 1].activationFunction(nextLayerInputs[nextLayerNeuron]);
                // Save the output for the backpropagation algorithm
                layers[currentLayer + 1].outputPerNeurons[nextLayerNeuron] = nextLayerInputs[nextLayerNeuron];
                //System.out.println("OUTPUT Layer " + (currentLayer + 1) + " neuron " + nextLayerNeuron + " = " + nextLayerInputs[nextLayerNeuron]);
            }

            currentInputs = nextLayerInputs;

        }


        return currentInputs;
    }

    private void backPropagation(double[] inputs, double[] solutions, double[] outputs) throws trainingError {
        // Reset delta weights and delta neuronsinitDelta();

        // A: Compute all node delta from the right to the left, this will also give us the delta for the bias

        // For the last layer, the computation is different because it is the last layer
        Layer lastLayer = layers[layers.length-1];
        // For each neuron of the last layer
        for(int i=0 ; i<lastLayer.neuronNumber ; i++){
            // compute the gradient ascent of the neuron
            lastLayer.deltaPerNeurons[i] = outputs[i]*(1-outputs[i]) * applyDerivativeErrorFunction(solutions[i], outputs[i]);
//            //System.out.println("FIRST " + i);
//            //System.out.println(lastLayer.deltaPerNeurons[i] );
        }

        // For the other layer, the computation is the same
        // For each layer, but we stop before the first layer, as it is the input layer we do not compute a delta
        for(int i=layers.length-2 ; i>0 ; i--){
            Layer currentLayer = layers[i];
            // For each neuron, compute the gradient ascent of the neuron
            for(int j=0 ; j <currentLayer.weightsPerNeurons.length ; j++){
                // For each weight
                double sumWeightByPreviousNeuron = 0;
                for(int k=0 ; k<currentLayer.weightsPerNeurons[j].length ; k++){
                    sumWeightByPreviousNeuron += currentLayer.weightsPerNeurons[j][k] * layers[i+1].deltaPerNeurons[k];
                }
                currentLayer.deltaPerNeurons[j] = sumWeightByPreviousNeuron * currentLayer.derivativeActivationFunction(currentLayer.outputPerNeurons[j]);
//                //System.out.println("SECOND " + i);
//                //System.out.println(currentLayer.deltaPerNeurons[j]);
            }
        }

        // B: Compute all weight delta from the beginning to the end

        // For the first layer, the output to consider is the input given to the neural network
        Layer firstLayer = layers[0];
        for(int i=0 ; i<firstLayer.deltaPerWeights.length ; i++){
            for(int j=0 ; j<firstLayer.deltaPerWeights[i].length ; j++){
//                ////System.out.println("DELTA FIRST LAYER " + layers[1].deltaPerNeurons[i]);
                firstLayer.deltaPerWeights[i][j] = inputs[i] * layers[1].deltaPerNeurons[j];
                //System.out.println("First layer neuron " + i + " weight " + j + " = " + inputs[i] + " *" +  layers[1].deltaPerNeurons[j]);
            }
        }

        // For the other layers, the output to consider is the output of the precedent layer
        for(int i=1 ; i<layers.length-1; i++){
            Layer currentLayer = layers[i];
            for(int j=0 ; j<currentLayer.weightsPerNeurons.length ; j++){
                for(int k=0 ; k<currentLayer.weightsPerNeurons[j].length ; k++){
//                    ////System.out.println("DELTA OTHER " + layers[i+1].deltaPerNeurons[k]);
                    currentLayer.deltaPerWeights[j][k] = layers[i+1].deltaPerNeurons[k] * currentLayer.outputPerNeurons[j];
                }
            }
        }
    }

    private void updateWeights(double learningRate) {
        for(int i=0 ; i<layers.length-1 ; i++){
            Layer currentLayer = layers[i];
            for(int j=0 ; j<currentLayer.weightsPerNeurons.length ; j++){
                for(int k=0 ; k<currentLayer.weightsPerNeurons[j].length ; k++){
                    // Update the weight
                    //System.out.println("-------------------------------------------------------------------------------------");
                    ////System.out.println("BEFORE UPDATE: Layer " + i + " neuron " + j + " weight " + k + " --> "  + currentLayer.weightsPerNeurons[j][k]);
                    ////System.out.println("DELTA WEIGHT" + currentLayer.deltaPerWeights[j][k]);
                    currentLayer.weightsPerNeurons[j][k] -= currentLayer.deltaPerWeights[j][k] * learningRate;
                    //System.out.println("AFTER UPDATE: Layer " + i + " neuron " + j + " weight " + k + " --> "  +  + currentLayer.weightsPerNeurons[j][k]);

                }
                // Update the bias
                //System.out.println("-------------------------------------------------------------------------------------");
                ////System.out.println("BEFORE UPDATE BIAS: Layer " + i + " neuron " + j + " --> " + currentLayer.bias[j]);
                ////System.out.println("DELTA NEURON" + currentLayer.deltaPerNeurons[j]);
                currentLayer.bias[j] -= currentLayer.deltaPerNeurons[j] * learningRate;
                //System.out.println("AFTER UPDATE BIAS: Layer " + i + " neuron " + j + " --> " + currentLayer.bias[j]);
            }
        }
        // Update the last layer bias neuron
        Layer lastLayer = layers[layers.length-1];
        // For each neuron of the last layer
        for(int i=0 ; i<lastLayer.neuronNumber ; i++){
            //System.out.println("-------------------------------------------------------------------------------------");
            ////System.out.println("BEFORE UPDATE BIAS: Layer " + (layers.length-1) + " neuron " + i + " --> " + lastLayer.bias[i]);
            ////System.out.println("DELTA NEURON" + lastLayer.deltaPerNeurons[i]);
            lastLayer.bias[i] -= lastLayer.deltaPerNeurons[i] * learningRate;
            //System.out.println("AFTER UPDATE BIAS: Layer " + (layers.length-1) + " neuron " + i + " --> " + lastLayer.bias[i]);
        }
    }

    class Layer{

        public double[][] weightsPerNeurons; // double[neuron][neuron weights]
        public double[][] deltaPerWeights;
        public double[] deltaPerNeurons;
        public double[] outputPerNeurons;
        public double[] bias;
        public int neuronNumber;
        public int neuronNumberNextLayer;
        public String activationFunction;


        public Layer(int neuronNumber, int neuronNumberNextLayer, String activationFunction){
            this.neuronNumber = neuronNumber;
            this.neuronNumberNextLayer = neuronNumberNextLayer;
            weightsPerNeurons = new double[neuronNumber][neuronNumberNextLayer];
            deltaPerWeights = new double[neuronNumber][neuronNumberNextLayer];
            deltaPerNeurons = new double[neuronNumber];
            outputPerNeurons = new double[neuronNumber];
            this.activationFunction = activationFunction;
            // We set 1 bias per neuron
            this.bias = new double[neuronNumber];
        }

        public double activationFunction(double x) throws trainingError {
            switch (activationFunction){
                case "sigmoid":
                    return (1.0 / (1.0 + Math.exp(-x)));
                default:
                    throw new trainingError("The error function " + errorFunction + " is not defined.");
            }
        }

        public double derivativeActivationFunction(double x) throws trainingError {
            switch (activationFunction){
                case "sigmoid":
                    return x * (1.0 - x);
                default:
                    throw new trainingError("The error function " + errorFunction + " is not defined.");
            }
        }
    }

    private double getRandom(){
        return 1/(Math.sqrt(2))-new Random().nextDouble();
    }

    public double applyErrorFunction(double target, double input) throws trainingError {
        switch (errorFunction){
            case "squaredError":
                return 0.5 * Math.pow((target - input), 2);
            default:
                throw new trainingError("The error function " + errorFunction + " is not defined.");
        }

    }

    public double applyDerivativeErrorFunction(double solution, double output) throws trainingError {
        switch (errorFunction){
            case "squaredError":
                return output - solution;
            default:
                throw new trainingError("The error function " + errorFunction + " is not defined.");
        }

    }
}

class trainingError extends Exception
{
    public trainingError(String message)
    {
        super(message);
    }
}