package nl.rug.joptimize.opt.optimizers;

import java.util.ArrayDeque;

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
    private double prevErr = Double.MAX_VALUE;
    
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
    	super.init(cf, initParams);
        this.protoLearningRate = initProtoLearningRate;
        this.matrixLearningRate = initMatrixLearningRate;
        this.hist = new ArrayDeque<>(histSize);
    }
    
    private GMLVQOptParam normalizedGrad(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam params) {
        GMLVQOptParam grad = cf.deriv(params);
        normalize(grad.prototypes);
        normalize(grad.weights);
        return grad;
    }
    
    private GMLVQOptParam weightedGrad(GMLVQOptParam grad) {
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
//        System.out.println("Params: "+params);
//        GMLVQOptParam grad = cf.deriv(params);
        GMLVQOptParam grad = normalizedGrad(cf,params);
//        System.out.println("Params: "+params);
//        System.out.println("Grad: "+grad);
        GMLVQOptParam outParams = weightedGrad(grad).add_s(params);
        
        if (hist.size() >= histSize-1) {
            GMLVQOptParam waypointAvg = outParams.zero();
            for (GMLVQOptParam p : hist) {
                waypointAvg.add_s(p);
            }
            waypointAvg.multiply_s(histInv);
            GMLVQOptParam protoAvg = new GMLVQOptParam(waypointAvg.prototypes, outParams.weights, outParams.labels);
            GMLVQOptParam matrixAvg = new GMLVQOptParam(outParams.prototypes, waypointAvg.weights, outParams.labels);

            matrixLearningRate *= gain;
            protoLearningRate *= gain;

            double err = cf.error(outParams), errAvg, errAvgP, errAvgM;
//            System.out.println("tmpParams: "+outParams+", "+err);
            err = Math.min(err, errAvg=cf.error(waypointAvg));
            err = Math.min(err, errAvgP=cf.error(protoAvg));
            err = Math.min(err, errAvgM=cf.error(protoAvg));
            if (errAvg == err) {
                // Average prototypes and matrix
//                System.out.println("WA wins");
                matrixLearningRate /= loss;
                protoLearningRate /= loss;
                outParams = waypointAvg;
            } else if (errAvgP == err) {
                // Average prototypes, normal step matrix
//                System.out.println("WA_p wins");
                protoLearningRate /= loss;
                outParams = protoAvg;
            } else if (errAvgM == err) {
                // Average matrix, normal step prototypes
//                System.out.println("WA_m wins");
                matrixLearningRate /= loss;
                outParams = matrixAvg;
            } else if (prevErr < err) {
//                System.out.println("Normal step wins... oops, went too far.");
                matrixLearningRate /= loss;
                protoLearningRate /= loss;
            } else {
//                System.out.println("Normal step wins.");
            }
            hist.remove();
            
            prevErr = err;
        }
        normalize(outParams.weights);
//        System.out.printf("mrate:%f, prate:%f\n",matrixLearningRate,protoLearningRate);
        hist.add(outParams.copy());
        return outParams;
    }
}
