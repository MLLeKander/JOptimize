// java -cp bin nl.rug.joptimize.runs.GMLVQPapari_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt GMLVQPapari --hist 3
// java -cp bin nl.rug.joptimize.runs.GMLVQPapari_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt vsgd
package nl.rug.joptimize.runs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQ;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.observers.CountObserver;
import nl.rug.joptimize.opt.optimizers.GMLVQPapari;

public class GMLVQPapari_1 {
    public static AbstractOptimizer<GMLVQOptParam> getOpt(Arguments a, LabeledDataSet ds) {
        if (a.get("opt").toUpperCase().replaceAll("[\\p{Punct} ]+", "").equals("GMLVQPAPARI")) {
            int dims = ds.dimensions();
            double prate = a.getDbl("prate",10.0/dims);
            double mrate = a.getDbl("mrate",20.0/dims);
            int hist = a.getInt("hist");
            double loss = a.getDbl("loss",1.2);
            double gain = a.getDbl("gain",1.05);
            double eps = a.getDbl("epsilon");
            int tmax = a.getInt("tmax");
            return new GMLVQPapari(prate,mrate,hist,loss,gain,eps,tmax);
        } else {
            return OptimizerFactory.createOptimizer(a);
        }
    }
    
    public static void ensureDir(String path) {
    	new File(path).mkdir();
    }
    
    public static void main(String[] argArr) throws FileNotFoundException {
        Arguments args = new Arguments(argArr);
        if (!args.hasDefault()) {
            System.err.println("No data file given.");
            return;
        }

        String outDir = args.get("outdir");
        ensureDir(outDir);
        final PrintStream run = new PrintStream(new File(outDir, "run.txt"));
        final PrintStream trained = new PrintStream(new File(outDir, "trained.txt"));
        final PrintStream init = new PrintStream(new File(outDir, "init.txt"));
        init.println("java "+GMLVQPapari_1.class.getCanonicalName()+" "+args.toString());
        
        LabeledDataSet dsTmp = LabeledDataSet.parseDataFile(new File(args.getDefault()));
        final LabeledDataSet ds = args.getBool("zscore",false) ? dsTmp.zscore() : dsTmp;
        AbstractOptimizer<GMLVQOptParam> opt = getOpt(args, ds);
        
        final int freq = args.getInt("freq", 50);
        final long startTime = System.nanoTime();
        opt.addObs(new OptObserver<GMLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GMLVQOptParam params, double cfError) {
                t++;
                if (t%freq == 0) {
                    int classificationErr = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            classificationErr++;
                        }
                    }
                    run.printf("%d,%d,%d,%f\n",t,System.nanoTime()-startTime,classificationErr,cfError/ds.size());
                }
            }
        });
        
        CountObserver<GMLVQOptParam> counter = new CountObserver<>();
        opt.addObs(counter);
        
        GMLVQOptParam p = new GMLVQOptParam(ds);
        GMLVQ lvq = new GMLVQ(ds, opt, p);
        
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        run.printf("%d,%d,%d,%f\n",counter.getEpochCount(),System.nanoTime()-startTime,err,lvq.cf.error(lvq.getParams())/ds.size());
        
        trained.println(lvq.getParams());

    }
}
