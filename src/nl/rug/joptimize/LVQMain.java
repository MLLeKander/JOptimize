// java -cp bin nl.rug.joptimize.LVQMain segment.dat  --rate 0.1 --epsilon 1e-4 --tmax 1000000 --freq 500 --opt bgd
package nl.rug.joptimize;

import java.io.File;
import java.io.FileNotFoundException;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.grlvq.GRLVQ;
import nl.rug.joptimize.learn.grlvq.GRLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.observers.CountObserver;
import nl.rug.joptimize.opt.observers.TimeObserver;

public class LVQMain {

    public static void printUsage() {
        String className = LVQMain.class.getName();
        System.out.println("java " + className
                + " dataFile [--numProtos numProtos] [--protoCounts proto1,proto2] [--opt optName]");
        System.exit(-1);
    }
    
    public static void main(String[] argArr) throws FileNotFoundException {
        Arguments args = new Arguments(argArr);
        if (!args.hasDefault()) {
            System.err.println("No data file given.");
            printUsage();
        }
        final LabeledDataSet ds = LabeledDataSet.parseDataFile(new File(args.getDefault()));
        AbstractOptimizer<GRLVQOptParam> opt = OptimizerFactory.createOptimizer(args);
        System.out.println("Proceeding with optimizer: "+opt.getClass().getName());
        
        final int freq = args.getInt("freq", 50);
        opt.addObs(new OptObserver<GRLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GRLVQOptParam params, double error) {
                t++;
                if (t%freq == 0) {
                    //System.out.println(params);
                    int err = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            err++;
                        }
                    }
                    System.out.println(t+": "+err+" / "+ds.size()+", "+error);//+ " | "+params);
                }
            }
        });
        
        CountObserver<GRLVQOptParam> counter = new CountObserver<>();
        TimeObserver<GRLVQOptParam> timer = new TimeObserver<>();
        opt.addObs(counter);
        opt.addObs(timer);

        //GRLVQOptParam p = new GRLVQOptParam(new double[][]{{0,0.5},{0,-0.5}}, new double[]{1,1}, new int[]{0,1});
        GRLVQ lvq = new GRLVQ(ds, opt, 3);//, p);
        
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        System.out.println(err+" / "+ds.size()+", "+lvq.cf.error(lvq.getParams()));
        System.out.println(counter.getEpochCount());
        
        System.out.println(lvq.getParams());
        System.out.println(timer.end - timer.start + " nanoseconds");
    }

}
