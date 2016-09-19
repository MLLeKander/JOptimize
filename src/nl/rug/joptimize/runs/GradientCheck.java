package nl.rug.joptimize.runs;

import java.util.Random;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.SigmoidCostFunction;
import nl.rug.joptimize.learn.glvq.GLVQCostFunction;
import nl.rug.joptimize.learn.glvq.GLVQOptParam;
import nl.rug.joptimize.learn.gmlvq.GMLVQCostFunction;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.learn.grlvq.GRLVQCostFunction;
import nl.rug.joptimize.learn.grlvq.GRLVQOptParam;
import nl.rug.joptimize.learn.simple.LinearCostFunction;
import nl.rug.joptimize.learn.simple.QuadraticCostFunction;
import nl.rug.joptimize.learn.simple.SingleVarOptParam;
import nl.rug.joptimize.opt.AbstractOptParam;
import nl.rug.joptimize.opt.CostFunction;

public class GradientCheck {
    public final static int NUM_ELEMENTS = 100;
    public final static int NUM_DIMS = 10;
    public final static int NUM_TESTS = 10;
    public final static double EPS = 1e-4;
    public final static double TOLERANCE = 5e-6;
    
    public static final String GLVQ_ID = "GLVQ";
    public static final String GRLVQ_ID = "GRLVQ";
    public static final String GMLVQ_ID = "GMLVQ";
    public static final String QUAD_ID = "QUAD";
    public static final String LINEAR_ID = "LINEAR";
    public static final String SIGMOID_ID = "SIGMOID";

    public static LabeledDataSet randomDataset(int elements, int dims) {
        double[][] data = new double[elements][dims];
        for (int i = 0; i < elements; i++) {
            for (int j = 0; j < dims; j++) {
                data[i][j] = Math.random()*20;
            }
        }
        int[] labels = new int[elements];
        for (int i = 0; i < elements; i++) {
            labels[i] = i%2 == 0 ? 1 : 0;
        }
        
        return new LabeledDataSet(data, labels);
    }

    @SuppressWarnings("rawtypes")
    public static AbstractOptParam getParams(LabeledDataSet data, String[] args) {
        String name = args[0].toUpperCase();
        if (name.equals(GLVQ_ID)) {
            return new GLVQOptParam(data);
        } else if (name.equals(GRLVQ_ID)) {
            return new GRLVQOptParam(data);
        } else if (name.equals(GMLVQ_ID)) {
            return new GMLVQOptParam(data);
        } else if (name.equals(QUAD_ID)) {
            return new SingleVarOptParam();
        } else if (name.equals(LINEAR_ID)) {
            return new SingleVarOptParam();
        } else if (name.equals(SIGMOID_ID)) {
            return new GLVQOptParam(data);
        } else {
            throw new IllegalArgumentException(args[0]);
        }
    }

    @SuppressWarnings("rawtypes")
    public static CostFunction getCostFunction(LabeledDataSet data, String[] args) {
        String name = args[0].toUpperCase();
        if (name.equals(GLVQ_ID)) {
            return new GLVQCostFunction(data);
        } else if (name.equals(GRLVQ_ID)) {
            return new GRLVQCostFunction(data);
        } else if (name.equals(GMLVQ_ID)) {
            return new GMLVQCostFunction(data);
        } else if (name.equals(QUAD_ID)) {
            return new QuadraticCostFunction();
        } else if (name.equals(LINEAR_ID)) {
            return new LinearCostFunction();
        } else if (name.equals(SIGMOID_ID)) {
            return new SigmoidCostFunction<GLVQOptParam>(new GLVQCostFunction(data), 5);
        } else {
            throw new IllegalArgumentException(args[0]);
        }
    }
    
    public static double relativeDiff(double a, double b) {
        return Math.abs(a-b)/Math.max(EPS,(Math.abs(a)+Math.abs(b))/2);
    }
    
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static boolean check(CostFunction cf, AbstractOptParam params) {
        AbstractOptParam analyticGrad = (AbstractOptParam) cf.deriv(params);
        AbstractOptParam analyticHesse = (AbstractOptParam) cf.hesseDiag(params);
        boolean out = true;

        double[] distFactors =  {      -4,     -3,    -2,    -1,    1,     2,      3,       4}; 
        double[] gradWeights =  { 1./280, -4./105,  1./5, -4./5, 4./5, -1./5, 4./105, -1./280};
        for (int dim = 0; dim < params.length(); dim++) {
            double preVal = params.get(dim);
            
            double gradSum = 0, hesseSum = 0;
            for (int i = 0; i < distFactors.length; i++) {
                params.set(dim, preVal + distFactors[i]*EPS);
                double tmpErr = cf.error(params);
                gradSum += gradWeights[i]*tmpErr;
                hesseSum += gradWeights[i]*((AbstractOptParam)cf.deriv(params)).get(dim);
            }
            double gradEst = gradSum/EPS;
            double hesseEst = hesseSum/EPS;
            params.set(dim, preVal);
            
            if (relativeDiff(gradEst,analyticGrad.get(dim)) > TOLERANCE) {
                System.out.printf("Grad issue (%d %f): %f (%f  %f)\n",dim,preVal,relativeDiff(gradEst,analyticGrad.get(dim)),gradEst,analyticGrad.get(dim));
                out = false;
            }
            if (relativeDiff(hesseEst,analyticHesse.get(dim)) > TOLERANCE) {
                System.out.printf("Hesse issue (%d %f): %f (%f  %f)\n",dim,preVal,relativeDiff(hesseEst,analyticHesse.get(dim)),hesseEst,analyticHesse.get(dim));
                out = false;
            }
            params.set(dim, preVal);
        }
        return out;
    }
    
    @SuppressWarnings("rawtypes")
    public static void main(String[] args) {
        LabeledDataSet data = randomDataset(NUM_ELEMENTS, NUM_DIMS);

        CostFunction cf = getCostFunction(data, args);
        AbstractOptParam params = getParams(data, args);
        Random r = new Random(10);
        
        for (int i = 0; i < NUM_TESTS; i++) {
            if (!check(cf, params)) {
                System.out.println("Something was wrong.");
            }
            params.random_s(r).multiply_s(20);
        }
    }
}
