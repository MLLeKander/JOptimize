package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.SeperableCostFunction;

public class GLVQCostFunction implements SeperableCostFunction<GRLVQOptParam> {
    private final LabeledDataSet ds;

    public GLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
    }

    @Override
    public double error(GRLVQOptParam params) {
        // TODO Vectorized?
        double out = 0;
        int size = this.size();
        for (int i = 0; i < size; i++) {
            out += error(params, i);
        }
        return out;
    }

    @Override
    public double error(GRLVQOptParam params, int exampleNdx) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);
        double dj = params.dist(j, data), dk = params.dist(k, data);
        return (dj - dk) / (dk + dj);
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params) {
        // TODO Vectorized?
        GRLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            deriv(params, i, out);
        }
        return out;
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params, int exampleNdx) {
        return deriv(params, exampleNdx, params.zero());
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double dj = params.dist(j, data), dk = params.dist(k, data);
        double dSum = dj+dk;

        double tmp = 4 / (dSum*dSum);
        double tmpJ = tmp * dk, tmpK = -tmp * dj;
        int dims = params.dimensions();

        for (int ndx = 0; ndx < dims; ndx++) {
            opj[ndx] += tmpJ * (pj[ndx] - data[ndx]);
            //double sav = tmpJ * (pj[ndx] - data[ndx]);
            //check(firstDerivLVQ(dj, dk, 2*(pj[ndx]-data[ndx]), 0),sav);

            opk[ndx] += tmpK * (pk[ndx] - data[ndx]);
            //sav = tmpK * (pk[ndx] - data[ndx]);
            //check(firstDerivLVQ(dj, dk, 0, 2*(pk[ndx]-data[ndx])),sav);
        }

        return out;
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params) {
        // TODO Vectorized?
        GRLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            hesseDiag(params, i, out);
        }
        return out;
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params, int exampleNdx) {
        return hesseDiag(params, exampleNdx, params.zero());
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double dj = params.dist(j, data), dk = params.dist(k, data);
        double dSum = dj + dk;

        double tmp = 4 / (dSum * dSum * dSum);
        // TODO Check signs.
        double tmpJ = tmp * dk * dSum, tmpJ_ = tmp * dk * -2;
        double tmpK = -tmp * dj * dSum, tmpK_ = tmp * dj * 2;
        int dims = params.dimensions();

        for (int ndx = 0; ndx < dims; ndx++) {
            opj[ndx] += tmpJ + tmpJ_ * (pj[ndx] - data[ndx]);
            check(secondDerivLVQ(dj, dk, 2*(pj[ndx]-data[ndx]), 0, 2, 0), tmpJ + tmpJ_ * (pj[ndx] - data[ndx]));
            opk[ndx] += tmpK + tmpK_ * (pk[ndx] - data[ndx]);
            check(secondDerivLVQ(dj, dk, 0, 2*(pk[ndx]-data[ndx]), 0, 2), tmpK + tmpK_ * (pk[ndx] - data[ndx]));
        }

        return out;
    }
    
    public void check(double a, double b) {
        if (Math.abs(a-b) / Math.min(a,b) > 1e-3) {
            throw new IllegalStateException();
        }
    }
    
    private double firstDerivLVQ(double dj, double dk, double djP, double dkP) {
        double dSum = dj+dk;
        return 2*(dk*djP - dj*dkP)/(dSum*dSum);
    }
    
    private double secondDerivLVQ(double dj, double dk, double djP, double dkP, double djPP, double dkPP) {
        double dSum = dj+dk;
        return (djPP-dkPP)/(dj+dk) - 2*(djP*djP-dkP*dkP)/(dSum*dSum) + (dj-dk)*(2*(djP-dkP)*(djP-dkP)/(dSum*dSum*dSum) - (djPP+dkPP)/(dSum*dSum));
    }

    @Override
    public int size() {
        return ds.size();
    }
}
