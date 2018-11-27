public class Main {

    public static void main(String[] a) throws buildingError {

        NeuralNetwork nn = new NeuralNetwork(2,
                1,
                2,
                1,
                new String[]{"prelu", "sigmoid"},
                "squaredError");
        // The number of solution output must match the number of neuron output
        Data dataA = new Data(new double[]{0.0}, new double[]{1.0, 1.0});
        Data dataB = new Data(new double[]{1.0}, new double[]{0.0, 1.0});
        Data dataC = new Data(new double[]{1.0}, new double[]{1.0, 0.0});
        Data dataD = new Data(new double[]{0.0}, new double[]{0.0, 0.0});
        DataSet dataSet = new DataSet(new Data[]{dataA, dataB, dataC, dataD});
        try {
            nn.setDataSet(dataSet);
            try {
                nn.train( 0.1, 0.1,100000);
                System.out.println(nn.forwardPropagation(new double[]{0.0, 0.0})[0]);
                System.out.println(nn.forwardPropagation(new double[]{1.0, 1.0})[0]);
                System.out.println(nn.forwardPropagation(new double[]{1.0, 0.0})[0]);
                System.out.println(nn.forwardPropagation(new double[]{0.0, 1.0})[0]);
            } catch (trainingError trainingError) {
                trainingError.printStackTrace();
            }
        } catch (trainingError trainingError) {
            trainingError.printStackTrace();
        }
    }

}
