import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * The training file should only contains numerical value in a CSV type file
 * Place the solution(s) column(s) first on the left at the beginning
 */

public class Main {

    public static void main(String[] args) throws buildingError {

        List<String> file;
        int outputNeurons; // the number of solution output for a data element
        int inputNeurons;

        if(args.length != 3){
            System.err.println("Usage: train.csv test.csv outputNeuronsUmber");
            return;
        }

        outputNeurons = Integer.parseInt(args[2]);

        String fileName = args[0];
        Path path = FileSystems.getDefault().getPath("./", fileName);

        try {

            file = Files.readAllLines(path);
            inputNeurons = file.get(0).split(",").length - outputNeurons;
            Data[] datas = new Data[file.size()];
            for(int lineIndex=0 ; lineIndex<datas.length ; lineIndex++){
                String line = file.get(lineIndex);
                String[] cells = line.split(",");
                double[] solutions = new double[outputNeurons];
                double[] inputs = new double[inputNeurons];
                for(int i=0 ; i<outputNeurons ; i++) solutions[i] = Double.valueOf(cells[i]);
                for(int i=outputNeurons ; i<cells.length ; i++) inputs[i-outputNeurons] = Double.valueOf(cells[i]);
                Data data = new Data(solutions, inputs);
                datas[lineIndex] = data;
            }

            DataSet dataSet = new DataSet(datas);

            // The number of input must match the number of neuron input
            // The number of solution output must match the number of neuron output
            // The number of hidden neuron number should be between inputNeurons and outputNeurons
            // The number of hidden layer is generally 1
            NeuralNetwork nn = new NeuralNetwork(inputNeurons, 1, inputNeurons, outputNeurons,
                    new String[]{"prelu","sigmoid"}, "squaredError", "Xavier");
            try {
                nn.setDataSet(dataSet);
                try {
                    nn.train( 0.01, 10000);
                    //System.out.println(nn.test(args[1], outputNeurons, inputNeurons));
                } catch (trainingError trainingError) {
                    trainingError.printStackTrace();
                }
            } catch (trainingError trainingError) {
                trainingError.printStackTrace();
            }
        } catch (IOException e) {
            System.err.println("File " + fileName + " can not be found");
            e.printStackTrace();
            return;
        }


    }

}
