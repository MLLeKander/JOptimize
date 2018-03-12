package nl.rug.joptimize.runs;

import java.io.IOException;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.glvq.GLVQClassifier;
import nl.rug.joptimize.learn.glvq.GLVQCostFunction;
import nl.rug.joptimize.learn.glvq.GLVQOptParam;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GLVQ_2 extends AbstractRunner<GLVQOptParam> {

    public GLVQ_2(String[] args) throws IOException {
        super(args);
    }

    @Override
    public GLVQOptParam getInitParams(Arguments args, LabeledDataSet ds) {
        return new GLVQOptParam(ds);
    }

    @Override
    public SeparableCostFunction<GLVQOptParam> getCostFunction(Arguments args, LabeledDataSet ds, GLVQOptParam init) {
        return new GLVQCostFunction(ds);
    }

    @Override
    public Classifier<GLVQOptParam> getClassifier(Arguments args, LabeledDataSet ds, Optimizer<GLVQOptParam> opt, GLVQOptParam init) {
        return new GLVQClassifier(opt, init);
    }

    public static void main(String[] args) throws IOException {
        GLVQ_2 runner = new GLVQ_2(args);
        runner.run();
        runner.close();
    }
}
