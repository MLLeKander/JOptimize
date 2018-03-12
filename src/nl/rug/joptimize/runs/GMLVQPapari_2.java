package nl.rug.joptimize.runs;

import java.io.IOException;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQClassifier;
import nl.rug.joptimize.learn.gmlvq.GMLVQCostFunction;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.SeparableCostFunction;
import nl.rug.joptimize.opt.optimizers.GMLVQPapari;

public class GMLVQPapari_2 extends AbstractRunner<GMLVQOptParam> {

    public GMLVQPapari_2(String[] args) throws IOException {
        super(args);
    }
    
    @Override
    public AbstractOptimizer<GMLVQOptParam> getOpt(Arguments a, LabeledDataSet ds) {
        if (a.get("opt").toUpperCase().replaceAll("[\\p{Punct} ]+", "").equals("PAPARI")) {
            double prate = a.getDbl("prate",1);
            double mrate = a.getDbl("mrate",2);
            int hist = a.getInt("hist", 5);
            double loss = a.getDbl("loss",1.5);
            double gain = a.getDbl("gain",1.1);
            boolean normalize = a.getBool("normalize", true);
            double eps = a.getDbl("epsilon");
            int tmax = a.getInt("tmax");
            long nsmax = a.getLong("nsmax");
            return new GMLVQPapari(prate,mrate,hist,loss,gain,normalize,eps,tmax,nsmax);
        } else {
            return OptimizerFactory.createOptimizer(a);
        }
    }

    @Override
    public GMLVQOptParam getInitParams(Arguments args, LabeledDataSet ds) {
        return new GMLVQOptParam(ds);
    }

    @Override
    public SeparableCostFunction<GMLVQOptParam> getCostFunction(Arguments args, LabeledDataSet ds, GMLVQOptParam init) {
        return new GMLVQCostFunction(ds);
    }

    @Override
    public Classifier<GMLVQOptParam> getClassifier(Arguments args, LabeledDataSet ds, Optimizer<GMLVQOptParam> opt, GMLVQOptParam init) {
        return new GMLVQClassifier(opt, init);
    }

    public static void main(String[] args) throws IOException {
        GMLVQPapari_2 runner = new GMLVQPapari_2(args);
        runner.run();
        runner.close();
    }
}