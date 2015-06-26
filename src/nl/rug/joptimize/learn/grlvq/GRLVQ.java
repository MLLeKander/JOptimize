package nl.rug.joptimize.learn.grlvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;

public class GRLVQ implements Classifier {

    GRLVQOptParam params;
    GRLVQOptParam init;
    public GRLVQCostFunction cf;
    Optimizer<GRLVQOptParam> opt;

    public GRLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, GRLVQOptParam init) {
        this.opt = opt;
        this.init = init;
        this.cf = new GRLVQCostFunction(ds);
        this.train(ds);
    }

    public GRLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt) {
        this(ds, opt, new GRLVQOptParam(ds));
    }

    public GRLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, int prototypesPerClass) {
        this(ds, opt, new GRLVQOptParam(prototypesPerClass, ds));
    }

    public GRLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, int[] prototypesPerClass) {
        this(ds, opt, new GRLVQOptParam(prototypesPerClass, ds));
    }

    public void setInit(GRLVQOptParam init) {
        this.init = init;
    }

    @Override
    public void train(LabeledDataSet ds) {
        this.params = opt.optimize(cf, this.init);
    }

    @Override
    public int classify(double[] data) {
        if (params == null) {
            throw new IllegalStateException("Classifier must be trained before classifying.");
        }

        return params.getClosestProtoLabel(data);
    }

    public GRLVQOptParam getParams() {
        return this.params;
    }
}