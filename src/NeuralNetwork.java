import java.util.Random;

public class NeuralNetwork {

    private Layer[] layers;
    private DataSet dataSet;

    public NeuralNetwork(int inputNeuronNumber,
                         int hiddenLayerNumber,
                         int hiddenNeuronNumber,
                         int outputNeuronNumber){

        layers = new Layer[1+hiddenLayerNumber+1];

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
        initDeltaWeights();

    }

    public void setDataSet(DataSet dataSet) {
        this.dataSet = dataSet;
    }

    private void initWeightAndBias(){
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
                System.out.println("Layer " + i + " neuron " + j + " bias " + layers[i].bias[j]);

                // each weight for the current neuron
                for(int k=0 ; k<layers[i].weightsPerNeurons[j].length ; k++){
                    layers[i].weightsPerNeurons[j][k] = getRandom();
                    System.out.println("Layer " + i + " neuron " + j + " weight " + k + " " + layers[i].weightsPerNeurons[j][k]);
                }

            }
        }
    }

    private void initDeltaWeights(){
        // each layer
        for(int i=0 ; i<layers.length ; i++){
            // each neuron
            for(int j=0 ; j<layers[i].deltaPerNeurons.length ; j++){
                // each delta weight for the current neuron
                for(int k=0 ; k<layers[i].deltaPerNeurons[j].length ; k++){
                    layers[i].deltaPerNeurons[j][k] = 0;
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
                    System.out.println("Data " + dataSet.getDatas()[i].getInputs()[0] + " " + dataSet.getDatas()[i].getInputs()[1]);
                    System.out.println("Output " + error + " = " + outputs[error]);
                    errors[error] = squaredError(outputs[error], dataSet.getDatas()[i].getSolution()[error]);
                    totalError += errors[error];
                }


                // Run backpropagation 
                backPropagation(dataSet.getDatas()[i].getSolution(), errors);

                // update the weights
                updateWeights(learningRate);
            }

            if(totalError < targetError){
                System.out.println("Solution found after " + epoch + " epochs for target error at " + targetError);
                break;
            }
        }

    }

    public static double squaredError(double target, double input){
        return 0.5 * Math.pow((target - input), 2);
    }

    private double[] forwardPropagation(double[] inputs) {
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
            }

            currentInputs = nextLayerInputs;

        }


        return currentInputs;
    }

    private void backPropagation(double[] solutions, double[] errors) {
        // TODO
    }

    private void updateWeights(double learningRate) {
        // TODO
        // Reset delta weights
        initDeltaWeights();
    }

    class Layer{

        public double[][] weightsPerNeurons; // double[neuron][neuron weights]
        public double[][] deltaPerNeurons;
        public double[] bias;
        public int neuronNumber;
        public int neuronNumberNextLayer;
        public String activationFunction;


        public Layer(int neuronNumber, int neuronNumberNextLayer, String activationFunction){
            this.neuronNumber = neuronNumber;
            this.neuronNumberNextLayer = neuronNumberNextLayer;
            weightsPerNeurons = new double[neuronNumber][neuronNumberNextLayer];
            deltaPerNeurons = new double[neuronNumber][neuronNumberNextLayer];
            this.activationFunction = activationFunction;
            // We set 1 bias per neuron
            this.bias = new double[neuronNumber];
        }

        public double activationFunction(double x){
            switch (activationFunction){
                case "sigmoid":
                    return (1 / (1 + Math.exp(-x)));
                default:
                    System.err.println("No activation function found for this layer.");
                    return 0.0;
            }
        }

        public double derivativeActivationFunction(double x){
            switch (activationFunction){
                case "sigmoid":
                    return x * (1.0 - x);
                default:
                    System.err.println("No activation function found for this layer.");
                    return 0.0;
            }
        }
    }

    private static double getRandom(){
        return 1/(Math.sqrt(2))-new Random().nextDouble();
    }
}

class trainingError extends Exception
{
    public trainingError(String message)
    {
        super(message);
    }
}
