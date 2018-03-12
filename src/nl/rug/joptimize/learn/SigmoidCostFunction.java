package nl.rug.joptimize.learn;

import nl.rug.joptimize.opt.AbstractSeparableCostFunction;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;
import static java.lang.Math.exp;

public class SigmoidCostFunction<ParamType extends OptParam<ParamType>> extends AbstractSeparableCostFunction<ParamType> {
    SeparableCostFunction<ParamType> baseCF;
    double alpha;
    
    public SigmoidCostFunction(SeparableCostFunction<ParamType> baseCF, double alpha) {
        this.baseCF = baseCF;
        this.alpha = alpha;
    }

    @Override
    public double error(ParamType params, int exampleNdx) {
        //return 4/(1+exp(-alpha*baseCF.error(params, exampleNdx))) - 2;
        return 1/(1+exp(-alpha*baseCF.error(params, exampleNdx)));
    }

    @Override
    public ParamType deriv(ParamType params, int exampleNdx, ParamType out) {
        // S'(x) = (S(x) - S(x)^2)*a*x'
        double s = error(params, exampleNdx);
        ParamType exampleDeriv = baseCF.deriv(params, exampleNdx);
        exampleDeriv.multiply_s(alpha*(s-s*s));
        return out.add_s(exampleDeriv);
    }

    @Override
    public ParamType hesseDiag(ParamType params, int exampleNdx, ParamType out) {
        // S''(x) = (S(x)-S(x)^2)*a*x'' + a*a*x'*x'*(1-2*S(x))*(S(x)-S(x)^2)
        //        = (x'*x'*a*a*(1-2*S(x)) + a*x'')*(S(x)-S(x)^2)
        double s = error(params, exampleNdx);
        ParamType xP  = baseCF.deriv(params, exampleNdx);
        ParamType xPP = baseCF.hesseDiag(params, exampleNdx);
        xP.dotprod_s(xP).multiply_s(alpha*alpha*(1-2*s)).add_s(xPP.multiply_s(alpha)).multiply_s(s-s*s);

        return out.add_s(xP);
    }

    @Override
    public int size() {
        return baseCF.size();
    }
}
