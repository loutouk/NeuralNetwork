public class Data {

    private double[] solution;
    private double[] inputs;

    public Data(double[] solution, double[] inputs) {
        this.solution = solution;
        this.inputs = inputs;
    }

    public double[] getSolution() {
        return solution;
    }

    public double[] getInputs() {
        return inputs;
    }
}
