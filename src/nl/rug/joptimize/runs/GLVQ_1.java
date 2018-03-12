// java -cp bin nl.rug.joptimize.runs.GLVQ_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt GMLVQPapari --hist 3
// java -cp bin nl.rug.joptimize.runs.GLVQ_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt vsgd
package nl.rug.joptimize.runs;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.SigmoidCostFunction;
import nl.rug.joptimize.learn.SplitLabeledDataSet;
import nl.rug.joptimize.learn.glvq.GLVQClassifier;
import nl.rug.joptimize.learn.glvq.GLVQCostFunction;
import nl.rug.joptimize.learn.glvq.GLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.SeparableCostFunction;
import nl.rug.joptimize.opt.observers.CountObserver;

public class GLVQ_1 {
    public static AbstractOptimizer<GLVQOptParam> getOpt(Arguments a, LabeledDataSet ds) {
        return OptimizerFactory.createOptimizer(a);
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
        init.println("java "+GLVQ_1.class.getCanonicalName()+" "+args.toString());
        
        LabeledDataSet dsTmp = LabeledDataSet.parseDataFile(new File(args.getDefault()));
        LabeledDataSet dsTestTmp = null;
        if (args.getBool("zscore", false)) {
            dsTmp = dsTmp.zscore();
        }
        if (args.hasArg("split")) {
            SplitLabeledDataSet split = dsTmp.split(args.getInt("split"),args.getLong("splitSeed"));
            dsTmp = split.a;
            dsTestTmp = split.b;
        }
        final LabeledDataSet ds = dsTmp;
        final LabeledDataSet dsTest = dsTestTmp; 
        AbstractOptimizer<GLVQOptParam> opt = getOpt(args, ds);
        
        final int freq = args.getInt("freq", 50);
        final long startTime = System.nanoTime();
        opt.addObs(new OptObserver<GLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GLVQOptParam params, double cfError) {
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
        
        CountObserver<GLVQOptParam> counter = new CountObserver<>();
        opt.addObs(counter);

        //System.out.println(args.getDefault()+","+ds.size()+","+ds.dimensions()+","+ds.classes());
        GLVQOptParam p = new GLVQOptParam(ds);
        //System.out.println("init:\n"+p);
        if (args.hasArg("initseed")) {
            Random r = new Random(args.getLong("initseed"));
            for (int i = 0; i < p.prototypes.length; i++) {
                for (int j = 0; j < p.prototypes[0].length; j++) {
                    p.prototypes[i][j] = r.nextDouble()*2 - 1;
                }
            }
        }
        GLVQClassifier lvq = new GLVQClassifier(opt, p);
        SeparableCostFunction<GLVQOptParam> cf = new GLVQCostFunction(ds);
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
