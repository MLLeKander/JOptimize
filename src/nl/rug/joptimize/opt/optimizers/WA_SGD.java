package nl.rug.joptimize.opt.optimizers;

import java.util.ArrayDeque;
import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class WA_SGD<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> {
    private double initLearningRate;
    private double learningRate;
    private int histSize;
    private double histInv;
    private ArrayDeque<ParamType> hist;
    private ParamType runningSum;
    private double loss;
    private double gain;
    private Random rand;
    
    public WA_SGD(long seed, double initialLearningRate, int histSize, double loss, double gain, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.initLearningRate = initialLearningRate;
        this.histSize = histSize;
        this.histInv = 1./histSize;
        this.loss = loss;
        this.gain = gain;
        this.rand = new Random(seed);
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        this.learningRate = initLearningRate;
        this.runningSum = initParams.zero();
        this.hist = new ArrayDeque<>(histSize);
    }
    
    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy();
        int size = cf.size();
        for (int i = 0; i < size; i++) {
            ParamType partialGrad = cf.deriv(outParams, rand.nextInt(size));
            outParams.sub_s(partialGrad.multiply_s(learningRate));
        }
        
        //TODO: .multiply(histInv) before adding to runningSum?
        runningSum.add_s(outParams);
        if (hist.size() >= histSize-1) {
            ParamType waypointAverage = runningSum.multiply(histInv);
            
            //TODO: Wasting a full error calculation here...
            if (cf.error(waypointAverage) < cf.error(outParams)) {
                runningSum.sub_s(outParams).add_s(waypointAverage);
                outParams = waypointAverage.copy();
                learningRate /= loss;
            } else {
                learningRate *= gain;
            }
            runningSum.sub_s(hist.remove());
        }
        hist.add(outParams.copy());
        return outParams;
    }
    
    @Override
    public String toString() {
        return learningRate+"";
    }
}
