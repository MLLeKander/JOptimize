// java -cp bin nl.rug.joptimize.runs.GMLVQPapari_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt GMLVQPapari --hist 3
// java -cp bin nl.rug.joptimize.runs.GMLVQPapari_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt vsgd
package nl.rug.joptimize.runs;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.SigmoidCostFunction;
import nl.rug.joptimize.learn.SplitLabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQClassifier;
import nl.rug.joptimize.learn.gmlvq.GMLVQCostFunction;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.SeparableCostFunction;
import nl.rug.joptimize.opt.observers.CountObserver;
import nl.rug.joptimize.opt.optimizers.GMLVQPapari;

public class GMLVQPapari_1 {
    public static AbstractOptimizer<GMLVQOptParam> getOpt(Arguments a, LabeledDataSet ds) {
        if (a.get("opt").toUpperCase().replaceAll("[\\p{Punct} ]+", "").equals("PAPARI")) {
            double prate = a.getDbl("prate",1);
            double mrate = a.getDbl("mrate",2);
            int hist = a.getInt("hist", 5);
            double loss = a.getDbl("loss",1.5);
            double gain = a.getDbl("gain",1.1);
            boolean normalize = a.getBool("normalize", true);
            double eps = a.getDbl("epsilon");
            int tmax = a.getInt("tmax");
            return new GMLVQPapari(prate,mrate,hist,loss,gain,normalize,eps,tmax);
        } else {
            return OptimizerFactory.createOptimizer(a);
        }
    }
    
    public static void ensureDir(String path) {
        new File(path).mkdir();
    }
    
    public static void main(String[] argArr) throws IOException {
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
        LabeledDataSet dsTestTmp = null;
        if (args.hasArg("split")) {
            SplitLabeledDataSet split = dsTmp.split(args.getInt("split"),args.getLong("splitSeed"));
            dsTmp = split.a;
            dsTestTmp = split.b;
        }
        if (args.getBool("zscore", false)) {
            dsTmp = dsTmp.zscore();
        }
        final LabeledDataSet ds = dsTmp;
        final LabeledDataSet dsTest = dsTestTmp; 
        AbstractOptimizer<GMLVQOptParam> opt = getOpt(args, ds);
        
        final int freq = args.getInt("freq", 50);
        final long startTime = System.nanoTime();
        opt.addObs(new OptObserver<GMLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GMLVQOptParam params, double cfError) {
                if (t++%freq == 0) {
                    int classificationErr = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            classificationErr++;
                        }
                    }
                    run.printf("%d,%d,%f,%f",t,System.nanoTime()-startTime,classificationErr/(double)ds.size(),cfError/ds.size());
                    if (dsTest != null) {
                        classificationErr = 0;
                        for (int i = 0; i < dsTest.size(); i++) {
                            if (params.getClosestProtoLabel(dsTest.getData(i)) != dsTest.getLabel(i)) {
                                classificationErr++;
                            }
                        }
                        run.printf(",%f", classificationErr/(double)dsTest.size());
                    }
                    run.println();
                }
            }
        });
        
        CountObserver<GMLVQOptParam> counter = new CountObserver<>();
        opt.addObs(counter);
        
        GMLVQOptParam p = args.hasArg("rank") ? new GMLVQOptParam(ds, args.getInt("rank"), args.getLong("seed")): new GMLVQOptParam(ds);
        System.out.println(p.weights.length);
        //System.out.println("init:\n"+p);
        p.normalizeWeights();
        if (args.hasArg("initseed")) {
            Random r = new Random(args.getLong("initseed"));
            for (int i = 0; i < p.prototypes.length; i++) {
                for (int j = 0; j < p.prototypes[0].length; j++) {
                    p.prototypes[i][j] = r.nextDouble()*2 - 1;
                }
            }
        }
        GMLVQClassifier lvq = new GMLVQClassifier(opt, p);
        SeparableCostFunction<GMLVQOptParam> cf = new GMLVQCostFunction(ds);
        if (args.getBool("sigmoid",false)) {
            cf = new SigmoidCostFunction<>(cf, args.getDbl("sigmoidAlpha",5));
        }
        lvq.train(cf);
        
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        run.printf("%d,%d,%f,%f\n",counter.getEpochCount(),System.nanoTime()-startTime,err/(double)ds.size(),cf.error(lvq.getParams())/ds.size());
        
        trained.println(lvq.getParams());

        run.close();
        trained.close();
        init.close();
    }
}
