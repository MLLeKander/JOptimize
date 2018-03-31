// java -cp bin nl.rug.joptimize.runs.GLVQ_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt GMLVQPapari --hist 3
// java -cp bin nl.rug.joptimize.runs.GLVQ_1 segment.dat  --epsilon 1e-5 --tmax 10000 --freq 5 --opt vsgd
package nl.rug.joptimize.runs;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.ROC;
import nl.rug.joptimize.learn.SigmoidCostFunction;
import nl.rug.joptimize.learn.SplitLabeledDataSet;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.ClassificationOptParam;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.OptimizerFactory;
import nl.rug.joptimize.opt.SeparableCostFunction;

public abstract class AbstractRunner<ParamType extends ClassificationOptParam<ParamType>> {
    LabeledDataSet ds = null, dsTest = null;
    long startTime;
    AbstractOptimizer<ParamType> opt = null;
    ParamType init = null, trained = null;
    SeparableCostFunction<ParamType> cf = null;
    Classifier<ParamType> classifier = null;
    
    PrintStream runFile = null, trainedFile = null, initFile = null, optFile = null, rocFile = null;
    
    Arguments args = null;
    
    int freq;
    
    public AbstractRunner(String[] args) throws IOException {
        this(new Arguments(args));
    }
    
    public AbstractRunner(Arguments args) throws IOException {
        this.args = args;
        
        loadDataset();
        
        opt = getOpt(args, ds);
        init = getInitParams(args, ds);
        cf = getCostFunction(args, ds, init);
        classifier = getClassifier(args, ds, opt, init);
        
        if (args.getBool("sigmoid",false)) {
            cf = new SigmoidCostFunction<>(cf, args.getDbl("sigmoidAlpha",5));
        }

        freq = args.getInt("freq", 1);
        
        opt.addObs(new RunnerObserver());
        
        openFiles();
    }
    
    public void run() {
        startTime = System.nanoTime();
        
        trained = classifier.train(cf);
        
        trainedFile.println(trained);
    }
    
    public void close() throws IOException {
        closeFiles();
    }
    
    public void loadDataset() throws IOException {
        if (!args.hasDefault()) {
            throw new RuntimeException("No data file given.");
        }
        
        ds = LabeledDataSet.parseDataFile(new File(args.getDefault()));
        dsTest = null;
        
        if (args.getBool("zscore", false)) {
            ds = ds.zscore();
        }
        
        if (args.hasArg("split")) {
            SplitLabeledDataSet split = ds.split(args.getInt("split"),args.getLong("splitSeed"));
            ds = split.a;
            dsTest = split.b;
        }
        
        if (args.hasArg("oversampleSeed")) {
            ds = ds.oversample(args.getLong("oversampleSeed"));
        }
    }
    
    public void openFiles() throws IOException {
        String outDir = args.get("outdir");
        ensureDir(outDir);
        runFile = new PrintStream(new File(outDir, "run.txt"));
        trainedFile = new PrintStream(new File(outDir, "trained.txt"));
        initFile = new PrintStream(new File(outDir, "init.txt"));
        optFile = new PrintStream(new File(outDir, "opt.txt"));
        if (ds.classes() == 2 && dsTest != null) {
            rocFile = new PrintStream(new File(outDir, "roc.txt"));
        }
        
        initFile.println("java "+AbstractRunner.class.getCanonicalName()+" "+args.toString());
    }
    
    public void closeFiles() throws IOException {
        runFile.close();
        trainedFile.close();
        initFile.close();
        optFile.close();
        if (rocFile != null) {
            rocFile.close();
        }
    }
    
    public AbstractOptimizer<ParamType> getOpt(Arguments args, LabeledDataSet ds) {
        return OptimizerFactory.createOptimizer(args);
    }
    
    public abstract ParamType getInitParams(Arguments args, LabeledDataSet ds);
    
    public abstract SeparableCostFunction<ParamType> getCostFunction(Arguments args, LabeledDataSet ds, ParamType init);
    
    public abstract Classifier<ParamType> getClassifier(Arguments args, LabeledDataSet ds, Optimizer<ParamType> opt, ParamType init);
    
    public static void ensureDir(String path) {
        new File(path).mkdir();
    }
    
    class RunnerObserver implements OptObserver<ParamType> {
        int t = 0;
        @Override
        public void notifyEpoch(ParamType param, double cfError) {
            if (t++%freq == 0) {
                int classificationErr = 0;
                for (int i = 0; i < ds.size(); i++) {
                    if (param.classify(ds.getData(i)) != ds.getLabel(i)) {
                        classificationErr++;
                    }
                }
                runFile.printf("%d,%d,%f,%f",t,System.nanoTime()-startTime,classificationErr/(double)ds.size(),cfError/ds.size());
                if (dsTest != null) {
                    classificationErr = 0;
                    for (int i = 0; i < dsTest.size(); i++) {
                        if (param.classify(dsTest.getData(i)) != dsTest.getLabel(i)) {
                            classificationErr++;
                        }
                    }
                    runFile.printf(",%f", classificationErr/(double)dsTest.size());
                }
                runFile.println();
                
                optFile.println(opt);
                
                if (rocFile != null) {
                    rocFile.println(ROC.auc(param, dsTest));
                }
            }
        }
    }
}
