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
    private boolean useNormalization;

    public GMLVQPapari(double initProtoLearningRate, double initMatrixLearningRate, int histSize, double loss, double gain, boolean useNormalization, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.initProtoLearningRate = initProtoLearningRate;
        this.initMatrixLearningRate = initMatrixLearningRate;
        this.histSize = histSize;
        this.histInv = 1./histSize;
        this.loss = loss;
        this.gain = gain;
        this.useNormalization = useNormalization;
    }

    public GMLVQPapari(double initProtoLearningRate, double initMatrixLearningRate, int histSize, double loss, double gain, double epsilon, int tMax) {
        this(initProtoLearningRate, initMatrixLearningRate, histSize, loss, gain, true, epsilon, tMax);
    }
    
    @Override
    public void init(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam initParams) {
    	super.init(cf, initParams);
        this.protoLearningRate = initProtoLearningRate;
        this.matrixLearningRate = initMatrixLearningRate;
        this.hist = new ArrayDeque<>(histSize);
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

    private GMLVQOptParam maybeNormalizedGrad(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam params) {
        GMLVQOptParam grad = cf.deriv(params);
        if (useNormalization) {
            grad.normalizeProtos().normalizeWeights();
        }
        //System.out.println("grad:\n"+grad);
        return weightedGrad(grad);
    }

    @Override
    public GMLVQOptParam optimizationStep(SeparableCostFunction<GMLVQOptParam> cf, GMLVQOptParam params) {
        GMLVQOptParam outParams;
        
        if (hist.size() >= histSize) {
            //System.out.println("--- Papari step");
            
            GMLVQOptParam oldParams = params.copy();
            GMLVQOptParam newParams = outParams = maybeNormalizedGrad(cf, params).add_s(params);
            
            GMLVQOptParam avgParams = params.zero();
            for (GMLVQOptParam p : hist) {
                avgParams.add_s(p);
            }
            avgParams.multiply_s(histInv);

            matrixLearningRate *= gain;
            protoLearningRate *= gain;
            
            double errAvgProto = cf.error(new GMLVQOptParam(avgParams.prototypes, oldParams.weights, oldParams.labels))/6;
            double errNewProto = cf.error(new GMLVQOptParam(newParams.prototypes, oldParams.weights, oldParams.labels))/6;
            double errAvgMatrix = cf.error(new GMLVQOptParam(oldParams.prototypes, avgParams.weights, oldParams.labels))/6;
            double errNewMatrix = cf.error(new GMLVQOptParam(oldParams.prototypes, newParams.weights, oldParams.labels))/6;
            //System.out.printf("errAvgProto: %.8f\n",errAvgProto);
            //System.out.printf("errNewProto: %.8f\n",errNewProto);
            //System.out.printf("errAvgMatrix: %.8f\n",errAvgMatrix);
            //System.out.printf("errNewMatrix: %.8f\n",errNewMatrix);
            if (errAvgProto <= errNewProto) {
                // Average prototypes
                //System.out.printf("WA_p wins: %.8f %.8f\n", errAvgProto, errNewProto);
                protoLearningRate /= loss;
                outParams.prototypes = avgParams.prototypes;
            }
            if (errAvgMatrix <= errNewMatrix) {
                // Average matrix
                //System.out.printf("WA_m wins: %.8f %.8f\n", errAvgMatrix, errNewMatrix);
                matrixLearningRate /= loss;
                outParams.weights = avgParams.weights;
            }
            hist.remove();
            //System.out.println("matrixLearningRate: "+matrixLearningRate);
            //System.out.println("protoLearningRate: "+protoLearningRate);
        }
        else {
            //System.out.println("--- BGD step");
            outParams =  maybeNormalizedGrad(cf, params).add_s(params);
        }
        outParams.normalizeWeights();
        hist.add(outParams.copy());

        //System.out.println("New params:");
        //System.out.println(outParams);

        return outParams;
    }
}
