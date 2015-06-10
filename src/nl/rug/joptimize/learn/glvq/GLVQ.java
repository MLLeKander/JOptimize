package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;

public class GLVQ implements Classifier {

    GRLVQOptParam params;
    GRLVQOptParam init;
    Optimizer<GRLVQOptParam> opt;

    public GLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, GRLVQOptParam init) {
        this.opt = opt;
        this.init = init;
        this.train(ds);
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt) {
        this(ds, opt, new GRLVQOptParam(ds));
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, int prototypesPerClass) {
        this(ds, opt, new GRLVQOptParam(prototypesPerClass, ds));
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GRLVQOptParam> opt, int[] prototypesPerClass) {
        this(ds, opt, new GRLVQOptParam(prototypesPerClass, ds));
    }

    public void setInit(GRLVQOptParam init) {
        this.init = init;
    }

    @Override
    public void train(LabeledDataSet ds) {
        this.params = opt.optimize(new GLVQCostFunction(ds), this.init);
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
