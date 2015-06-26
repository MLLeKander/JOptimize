package nl.rug.joptimize.opt;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.opt.optimizers.BGD;
import nl.rug.joptimize.opt.optimizers.SGD;
import nl.rug.joptimize.opt.optimizers.WA_BGD;
import nl.rug.joptimize.opt.optimizers.WA_SGD;
import nl.rug.joptimize.opt.optimizers.WaypointAverage;
import nl.rug.joptimize.opt.optimizers.VSGD;

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
    
    public static <ParamType extends OptParam<ParamType>> WaypointAverage<ParamType> createWA(Arguments a) {
        AbstractOptimizer<ParamType> base = createFromName(a.get("base"), a);
        return new WaypointAverage<>(base,a.getInt("hist"),a.getDbl("epsilon"),a.getInt("tmax"));
    }
}
