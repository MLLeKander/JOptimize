package nl.rug.joptimize.opt.optimizers;

import java.util.ArrayDeque;
import java.util.Arrays;

import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GMLVQPapari extends AbstractOptimizer<GMLVQOptParam> {
    private double initProtoLearningRate;
    private double initMatrixLearningRate;
    private double protoLearningRate;
    private double matrixLearningRate;
    private int histSize;
    private double histInv;
    private ArrayDeque<GMLVQOptParam> hist;
    private double loss;
    private double gain;
    
    public GMLVQPapari(double initProtoLearningRate, double initMatrixLearningRate, int histSize, double loss, double gain, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.initProtoLearningRate = initProtoLearningRate;
        this.initMatrixLearningRate = initMatrixLearningRate;
        this.histSize = histSize;
        this.histInv = 1./histSize;
        this.loss = loss;
        this.gain = gain;
    }
    
    @Override
    public void init(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam initParams) {
        this.protoLearningRate = initProtoLearningRate;
        this.matrixLearningRate = initMatrixLearningRate;
        this.hist = new ArrayDeque<>(histSize);
    }
    
    private GMLVQOptParam normalizedGrad(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam params) {
        GMLVQOptParam grad = cf.deriv(params);
        normalize(grad.prototypes);
        normalize(grad.weights);
        System.out.println("Grad: "+grad);
        for (int i = 0; i < grad.prototypes.length; i++) {
            for (int j = 0; j < grad.prototypes[i].length; j++) {
                grad.prototypes[i][j] *= -protoLearningRate;
            }
        }
        for (int i = 0; i < grad.weights.length; i++) {
            for (int j = 0; j < grad.weights[i].length; j++) {
                grad.weights[i][j] *= -matrixLearningRate;
            }
        }
        return grad;
    }
    
    private void normalize(double[][] m) {
        double norm = 0;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                norm += m[i][j] * m[i][j];
            }
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] /= norm;
            }
        }
    }

    @Override
    public GMLVQOptParam optimizationStep(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam params) {
        System.out.println("Params: "+params);
        GMLVQOptParam grad = normalizedGrad(cf,params);
        GMLVQOptParam outParams = grad.add_s(params);
        
        if (hist.size() >= histSize-1) {
            GMLVQOptParam waypointAvg = outParams.zero();
            for (GMLVQOptParam p : hist) {
                waypointAvg.add_s(p);
            }
            waypointAvg.multiply_s(histInv);
//            System.out.println("outParams: "+outParams);
//            System.out.println("waypointAvg: "+waypointAvg);
            GMLVQOptParam protoAvg;
            GMLVQOptParam matrixAvg;

            matrixLearningRate *= gain;
            protoLearningRate *= gain;

            double err = cf.error(outParams);
            if (cf.error(waypointAvg) < err) {
                // Average prototypes and matrix
                matrixLearningRate /= loss;
                protoLearningRate /= loss;
                outParams = waypointAvg;
                System.out.println("WA1");
            } else if (cf.error(protoAvg = new GMLVQOptParam(waypointAvg.prototypes, outParams.weights, outParams.labels)) < err) {
                // Average prototypes, normal step matrix
                protoLearningRate /= loss;
                outParams = protoAvg;
                System.out.println("WA_Proto");
            } else if (cf.error(matrixAvg = new GMLVQOptParam(outParams.prototypes, waypointAvg.weights, outParams.labels)) < err) {
                // Average matrix, normal step prototypes
                matrixLearningRate /= loss;
                outParams = matrixAvg;
                System.out.println("WA_Matrix");
            }
            hist.remove();
        }
        hist.add(outParams.copy());
        return outParams;
    }
}
