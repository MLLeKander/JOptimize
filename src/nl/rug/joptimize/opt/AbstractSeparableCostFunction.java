package nl.rug.joptimize.opt;


public abstract class AbstractSeparableCostFunction<ParamType extends OptParam<ParamType>> implements SeparableCostFunction<ParamType> {
    @Override
    public double error(ParamType params) {
        // TODO Vectorized?
        double out = 0;
        int size = this.size();
        for (int i = 0; i < size; i++) {
            out += error(params, i);
        }
        return out;
    }

    @Override
    public ParamType deriv(ParamType params) {
        // TODO Vectorized?
        ParamType out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            deriv(params, i, out);
        }
        return out;
    }

    @Override
    public ParamType deriv(ParamType params, int exampleNdx) {
            ParamType out =  deriv(params, exampleNdx, params.zero());
    //        System.out.println("deriv: "+exampleNdx+" "+Arrays.toString(ds.getData(exampleNdx)));
            return out;
        }

    @Override
    public ParamType hesseDiag(ParamType params) {
        // TODO Vectorized?
        ParamType out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            hesseDiag(params, i, out);
        }
        return out;
    }

    @Override
    public ParamType hesseDiag(ParamType params, int exampleNdx) {
        return hesseDiag(params, exampleNdx, params.zero());
    }

}