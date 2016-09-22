package nl.rug.joptimize.runs;

import java.io.File;
import java.io.IOException;
import java.util.List;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQClassifier;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.optimizers.Composite;

public abstract class AbstractCompositeRun {
    protected LabeledDataSet ds;
    public void main_(String[] argArray) throws IOException {
        Arguments args = new Arguments(argArray);
        String fileName = args.hasDefault() ? args.getDefault() : "segment.dat";
        ds = LabeledDataSet.parseDataFile(new File(fileName));
        
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
//        final long start = System.nanoTime();
        System.out.println("---- "+opt);
//        opt.addObs(new OptObserver<GMLVQOptParam>() {
//            @Override
//            public void notifyEpoch(GMLVQOptParam params, double costError) {
//                long t = System.nanoTime()-start;
//                int errCnt = 0;
//                for (int i = 0; i < ds.size(); i++) {
//                    if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
//                        errCnt++;
//                    }
//                }
//                System.out.println(t+","+errCnt+","+costError);
//            }
//        });
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

    public abstract List<AbstractOptimizer<GMLVQOptParam>> getOpts();
}
