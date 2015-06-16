package nl.rug.joptimize.opt.optimizers;

import java.util.ArrayDeque;
import java.util.Map;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeperableCostFunction;

public class WA_BGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double initLearningRate;
    private double epsilon;
    private int tMax;
    private int histSize;
    private ArrayDeque<ParamType> hist;
    private double loss;
    private double gain;
    
    public static double pDbl(Map<String, String> params, String key) {
        if (!params.containsKey("--"+key)) {
            throw new IllegalArgumentException("Required argument: "+key);
        }
        return Double.parseDouble(params.get("--"+key));
    }
    
    public static double pDbl(Map<String, String> params, double deflt, String key) {
        if (!params.containsKey("--"+key)) {
            return deflt;
        }
        return Double.parseDouble(params.get("--"+key));
    }
    
    public static int pInt(Map<String, String> params, String key) {
        if (!params.containsKey("--"+key)) {
            throw new IllegalArgumentException("Required argument: "+key);
        }
        return Integer.parseInt(params.get("--"+key));
    }
    
    public static int pInt(Map<String, String> params, int deflt, String key) {
        if (!params.containsKey("--"+key)) {
            return deflt;
        }
        return Integer.parseInt(params.get("--"+key));
    }
    
    public WA_BGD(Map<String, String> p) {
        this(pDbl(p,"rate"),pDbl(p,"epsilon"),pInt(p,"tmax"),pInt(p,"hist"),pDbl(p,1,"loss"),pDbl(p,1,"gain"));
    }

    public WA_BGD(double initialLearningRate, double epsilon, int tMax, int histSize, double loss) {
        this(initialLearningRate, epsilon, tMax, histSize, loss, 1);
    }

    public WA_BGD(double initialLearningRate, double epsilon, int tMax, int histSize, double loss, double gain) {
        this.initLearningRate = initialLearningRate;
        this.epsilon = epsilon;
        this.tMax = tMax;
        this.histSize = histSize;
        this.hist = new ArrayDeque<>(histSize);
        this.loss = loss;
        this.gain = gain;
    }

    public ParamType optimize(SeperableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();
        double learningRate = initLearningRate;
        
        ParamType runningSum = initParams.zero();

        double diff = Double.MAX_VALUE, histInv = 1./histSize;
        for (int t = 0; t < tMax && diff >= epsilon; t++) {
            ParamType grad = cf.deriv(params);
            
            diff = grad.squaredNorm();
            params.sub_s(grad.multiply_s(learningRate));///Math.sqrt(grad.squaredNorm())));
            double err = cf.error(params);
            
            runningSum.add_s(params);
            if (hist.size() >= histSize) {
                runningSum.sub_s(hist.remove());
                ParamType waypointAverage = runningSum.multiply(histInv);
                double errTilde = cf.error(waypointAverage);
                
                if (errTilde < err) {
                    diff = Double.MAX_VALUE;
                    err = errTilde;
                    params = waypointAverage.copy();
                    //System.out.print("Before: "+learningRate+" ");
                    learningRate *= loss;
                    //System.out.println("After: "+learningRate);
                } else {
                    learningRate *= gain;
                }
            }
            hist.add(params.copy());

            this.notifyEpoch(params, err);
        }
        return params;
    }
    
    public String toString() {
        return String.format("%s (nu=%.2f,ep=%.2f,tMax=%d,hist=%d,loss=%.2f,gain=%.2f)", this.getClass().getSimpleName(),initLearningRate,epsilon,tMax,histSize,loss,gain);
    }
}
