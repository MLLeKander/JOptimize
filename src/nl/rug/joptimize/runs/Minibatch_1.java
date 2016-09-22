package nl.rug.joptimize.runs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.optimizers.BGD;
import nl.rug.joptimize.opt.optimizers.Minibatch;

public class Minibatch_1 extends AbstractCompositeRun {

    public static void main(String[] args) throws IOException {
        new Minibatch_1().main_(args);
    }

    @Override
    public List<AbstractOptimizer<GMLVQOptParam>> getOpts() {
        double eps = 1e-5, learningRate = 0.1;
        int tMax = 5000, seed = 5;
        List<AbstractOptimizer<GMLVQOptParam>> opts = new ArrayList<AbstractOptimizer<GMLVQOptParam>>();
        opts.add(new BGD<GMLVQOptParam>(learningRate, eps, tMax));
        for (int i = 1; 20 * (i-1) < ds.size(); i++) {
            opts.add(new Minibatch<GMLVQOptParam>(seed, learningRate, 20 * i, eps, tMax));
        }
        return opts;
    }
}
