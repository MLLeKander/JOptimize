package nl.rug.joptimize.learn.gmlvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;

public class GMLVQ implements Classifier {

    GMLVQOptParam params;
    GMLVQOptParam init;
    public GMLVQCostFunction cf;
    Optimizer<GMLVQOptParam> opt;

    public GMLVQ(LabeledDataSet ds, Optimizer<GMLVQOptParam> opt, GMLVQOptParam init) {
        this.opt = opt;
        this.init = init;
        this.cf = new GMLVQCostFunction(ds);
        this.train(ds);
    }

    public GMLVQ(LabeledDataSet ds, Optimizer<GMLVQOptParam> opt) {
        this(ds, opt, new GMLVQOptParam(ds));
    }

    public GMLVQ(LabeledDataSet ds, Optimizer<GMLVQOptParam> opt, int prototypesPerClass) {
        this(ds, opt, new GMLVQOptParam(prototypesPerClass, ds));
    }

    public GMLVQ(LabeledDataSet ds, Optimizer<GMLVQOptParam> opt, int[] prototypesPerClass) {
        this(ds, opt, new GMLVQOptParam(prototypesPerClass, ds));
    }

    public void setInit(GMLVQOptParam init) {
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

    public GMLVQOptParam getParams() {
        return this.params;
    }
}
