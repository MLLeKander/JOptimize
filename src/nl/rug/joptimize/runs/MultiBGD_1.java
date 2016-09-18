package nl.rug.joptimize.runs;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQClassifier;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.optimizers.MultiBGD;

public class MultiBGD_1 {

    public static void main(String[] argArray) throws FileNotFoundException {
        Arguments args = new Arguments(argArray);
        String fileName = args.hasDefault() ? args.getDefault() : "segment.dat";
        final LabeledDataSet ds = LabeledDataSet.parseDataFile(new File(fileName));
        double[][] protos = ds.averageProtos();
        double[][] weights = new double[ds.dimensions()][ds.dimensions()];
        int[] labels = new int[ds.classes()];
        for (int i = 0; i < weights.length; i++) {
            weights[i][i] = 1;
        }
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i;
        }
        double eps = 1e-100;
        int tMax = 5000;
        
        ArrayList<Double> learningRates = new ArrayList<>(100);
        for (int i = 0; i < 62; i++) {
            learningRates.add(1e-10 * (1L<<i));
        }
        System.out.println(learningRates);
        GMLVQOptParam init = new GMLVQOptParam(protos, weights, labels);
        final MultiBGD<GMLVQOptParam> opt = new MultiBGD<>(learningRates, eps, tMax);
        
        final long start = System.nanoTime();
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
                System.out.println(opt.minRate+","+t+","+errCnt+","+costError);
            }
        });
        new GMLVQClassifier(opt, init).train(ds);
    }
}
