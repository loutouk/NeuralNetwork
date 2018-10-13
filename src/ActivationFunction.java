public class ActivationFunction {

    public static double logistic(double input){
        return (1 / (1 + Math.exp(-input)));
    }
    public static double logisticDerivative(double input){
        return input * (1.0 - input);
    }

}
