package nl.rug.joptimize.runs;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQClassifier;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.optimizers.Composite;
import nl.rug.joptimize.opt.optimizers.MultiBGD;
import nl.rug.joptimize.opt.optimizers.SGD;
import nl.rug.joptimize.opt.optimizers.VSGD;
import nl.rug.joptimize.opt.optimizers.WA_BGD;
import nl.rug.joptimize.opt.optimizers.WA_SGD;

public class GMLVQ_2 {

    public static void main(String[] args) throws IOException {
        final LabeledDataSet ds = LabeledDataSet.parseDataFile(new File("segment.dat"));
        double[][] protos = new double[ds.classes()][ds.dimensions()];//ds.averageProtos();
        double[][] weights = new double[ds.dimensions()][ds.dimensions()];
        int[] labels = new int[ds.classes()];
        for (int i = 0; i < weights.length; i++) {
            weights[i][i] = 1;
        }
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i;
        }
        GMLVQOptParam init = new GMLVQOptParam(protos, weights, labels);
        AbstractOptimizer<GMLVQOptParam> opt = new Composite<>(getOpts(),1e-5,5000);
        final long start = System.nanoTime();
        System.out.println("---- "+opt);
        opt.addObs(new OptObserver<GMLVQOptParam>() {
            @Override
            public void notifyEpoch(GMLVQOptParam params, double costError) {
                long t = System.nanoTime()-start;
                int errCnt = 0;
                for (int i = 0; i < ds.size(); i++) {
                    if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                        errCnt++;
                    }
                }
                System.out.println(t+","+errCnt+","+costError);
            }
        });
        GMLVQClassifier gmlvq = new GMLVQClassifier(opt, init);
        gmlvq.train(ds);
        double[][] w = gmlvq.getParams().weights;
        for (double[] row : w) {
            for (double d : row) {
                System.out.print(d+" ");
            }
            System.out.println();
        }
    }

    public static List<AbstractOptimizer<GMLVQOptParam>> getOpts() {
        double eps = 1e-5;
        int tMax = 5000, seed = 5;
        List<AbstractOptimizer<GMLVQOptParam>> opts = Arrays.asList(
                (AbstractOptimizer<GMLVQOptParam>) new VSGD<GMLVQOptParam>(seed, eps, tMax),
                (AbstractOptimizer<GMLVQOptParam>) new MultiBGD<GMLVQOptParam>(Arrays.asList(0.0001,0.0005,0.001,0.005,0.01, 0.05,
                        0.1, 0.5, 1., 5., 10., 50.), eps, tMax),
                (AbstractOptimizer<GMLVQOptParam>) new SGD<GMLVQOptParam>(seed, 0.1, eps, tMax),
                (AbstractOptimizer<GMLVQOptParam>) new WA_BGD<GMLVQOptParam>(0.1, 5, 0.9, 1.1, eps, tMax),
                (AbstractOptimizer<GMLVQOptParam>) new WA_SGD<GMLVQOptParam>(seed,0.1, 5, 0.9, 1.1, eps, tMax)
        );
        return opts;
    }
}
