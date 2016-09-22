package nl.rug.joptimize.opt;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.opt.optimizers.Adadelta;
import nl.rug.joptimize.opt.optimizers.BGD;
import nl.rug.joptimize.opt.optimizers.ControlledBGD;
import nl.rug.joptimize.opt.optimizers.ControlledSGD;
import nl.rug.joptimize.opt.optimizers.SGD;
import nl.rug.joptimize.opt.optimizers.SlowStartSGD;
import nl.rug.joptimize.opt.optimizers.VSGD;
import nl.rug.joptimize.opt.optimizers.WA_BGD;
import nl.rug.joptimize.opt.optimizers.WA_SGD;
import nl.rug.joptimize.opt.optimizers.WaypointAverage;

public class OptimizerFactory {
    public static <ParamType extends OptParam<ParamType>> AbstractOptimizer<ParamType> createOptimizer(Arguments args) {
        if (!args.hasArg("opt")) {
            return new BGD<>(0.1, 1e-5, 10000);
        }
        return createFromName(args.get("opt"), args);
    }
    public static <ParamType extends OptParam<ParamType>> AbstractOptimizer<ParamType> createFromName(String opt, Arguments args) {
        String name = opt.toUpperCase().replaceAll("[\\p{Punct} ]+", "");
        if (name.equals("BGD")) {
            return createBGD(args);
        } else if (name.equals("SGD")) {
            return createSGD(args);
        } else if (name.equals("VSGD")) {
            return createVSGD(args);
        } else if (name.equals("WABGD")) {
            return createWABGD(args);
        } else if (name.equals("WASGD")) {
            return createWASGD(args);
        } else if (name.equals("WA")) {
            return createWA(args);
        } else if (name.equals("CONTROLLEDBGD")) {
            return createControlledBGD(args);
        } else if (name.equals("CONTROLLEDSGD")) {
            return createControlledSGD(args);
        } else if (name.equals("SLOWSTARTSGD")) {
            return createSlowStartSGD(args);
        } else if (name.equals("ADADELTA")) {
            return createAdadelta(args);
        } else if (name.equals("ADADELTABATCH")) {
            return createAdadeltaBatch(args);
        }
        throw new IllegalArgumentException("Unknown optimizer: "+name);
    }
    
    public static <ParamType extends OptParam<ParamType>> BGD<ParamType> createBGD(Arguments a) {
        return new BGD<>(a.getDbl("rate"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> SGD<ParamType> createSGD(Arguments a) {
        return new SGD<>(a.getLong("seed",1),a.getDbl("rate"), a.getDbl("epsilon"), a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> VSGD<ParamType> createVSGD(Arguments a) {
        return new VSGD<>(a.getLong("seed",1),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> WA_SGD<ParamType> createWASGD(Arguments a) {
        return new WA_SGD<>(a.getLong("seed", 1),a.getDbl("rate"),a.getInt("hist"),a.getDbl("loss",1),a.getDbl("gain",1),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> WA_BGD<ParamType> createWABGD(Arguments a) {
        return new WA_BGD<>(a.getDbl("rate"),a.getInt("hist"),a.getDbl("loss",1),a.getDbl("gain",1),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> Adadelta<ParamType> createAdadelta(Arguments a) {
        return new Adadelta<>(a.getLong("seed"),a.getInt("batchSize",1),a.getDbl("rho",0.95),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> Adadelta<ParamType> createAdadeltaBatch(Arguments a) {
        return new Adadelta<>(a.getLong("seed"),a.getInt("batchSize",Integer.MAX_VALUE),a.getDbl("rho",0.95),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> SlowStartSGD<ParamType> createSlowStartSGD(Arguments a) {
        double[] rates = {};
        if (a.hasArg("rates")) {
            String[] split = a.get("rates").split(",");
            rates = new double[split.length];
            for (int i = 0; i < split.length; i++) {
                rates[i] = Double.valueOf(split[i]);
            }
        } else {
            rates = new double[]{100,10,5,1,0.1,0.05,0.01,0.005,0.001,5e-4,1e-4};
        }
        return new SlowStartSGD<>(rates,a.getLong("seed"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> WaypointAverage<ParamType> createWA(Arguments a) {
        AbstractOptimizer<ParamType> base = createFromName(a.get("base"), a);
        return new WaypointAverage<>(base,a.getInt("hist"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> ControlledBGD<ParamType> createControlledBGD(Arguments a) {
        return new ControlledBGD<>(a.getDbl("rate"),a.getDbl("loss"),a.getDbl("gain"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
    
    public static <ParamType extends OptParam<ParamType>> ControlledSGD<ParamType> createControlledSGD(Arguments a) {
        return new ControlledSGD<>(a.getLong("seed", 1),a.getDbl("rate"),a.getDbl("loss"),a.getDbl("gain"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
}
