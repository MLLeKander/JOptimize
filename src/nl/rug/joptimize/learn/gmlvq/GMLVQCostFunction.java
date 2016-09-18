package nl.rug.joptimize.learn.gmlvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.AbstractSeparableCostFunction;

public class GMLVQCostFunction extends AbstractSeparableCostFunction<GMLVQOptParam> {
    private final LabeledDataSet ds;

    public GMLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
    }

    @Override
    public double error(GMLVQOptParam params, int exampleNdx) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);
        double dj = params.dist(j, data), dk = params.dist(k, data);
        return (dj - dk) / (dk + dj);
    }

    @Override
    public GMLVQOptParam deriv(GMLVQOptParam params, int exampleNdx, GMLVQOptParam out) {
        // (2*dk*dj' - 2*dj*dk')/(dj+dk)^2
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        if (j < 0 || k < 0) System.out.println("WTF is happening? "+params);
        
        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[][] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);
        //double dSum = dj+dk;

        int dims = params.dimensions(), numWeights = params.numWeights();
        //double tmp = 2 / (dSum * dSum);
        // TODO Check signs.
        //double tmpJ = tmp * 2 * dk, tmpK = -tmp * 2 * dj;
        //double tmpWj = tmp * dk, tmpWk = -tmp * dj; 

        /*double djP, dkP, sav;
        for (int ndx = 0; ndx < dims; ndx++) {
            double diffJ = pj[ndx] - data[ndx];
            double diffK = pk[ndx] - data[ndx];
                        
            djP = 2*weights[ndx]*diffJ; dkP = 0;
            sav = tmpJ * diffJ * weights[ndx];
            opj[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP),sav, "opj'");
            
            djP = 0; dkP = 2*weights[ndx]*diffK;
            sav = tmpK * diffK * weights[ndx];
            opk[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP), sav, "opk'");
            
            // (2*dk*(pj-data)^2 - 2*dj*(pk-data)^2)/(dj+dk)^2
            djP = weights[ndx] * diffJ * diffJ; dkP = weights[ndx] * diffK * diffK;
            sav = weights[ndx] * (tmpWj * diffJ * diffJ + tmpWk * diffK * diffK);
            oweights[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP),sav, "oweights'");
        }*/
        
        //TODO: This can probably be sped up by not using the firstDerivLVQ calls...
        double[] dj_pjP = new double[dims], dk_pkP = new double[dims];
        for (int i = 0; i < numWeights; i++) {
            double compJ = 0, compK = 0;
            double[] wRow = weights[i], owRow = oweights[i];
            for (int o = 0; o < dims; o++) {
                compJ += wRow[o]*(pj[o] - data[o]);
                compK += wRow[o]*(pk[o] - data[o]);
            }
            for (int o = 0; o < dims; o++) {
                double dj_wP = 2*compJ*(pj[o] - data[o]);
                double dk_wP = 2*compK*(pk[o] - data[o]);
                owRow[o] += firstDerivLVQ(dj, dk, dj_wP, dk_wP);
                dj_pjP[o] += compJ*wRow[o];
                dk_pkP[o] += compK*wRow[o];
            }
        }
        for (int o = 0; o < dims; o++) {
            opj[o] += firstDerivLVQ(dj, dk, 2*dj_pjP[o], 0);
            opk[o] += firstDerivLVQ(dj, dk, 0, 2*dk_pkP[o]);
        }

        return out;
    }

    @Override
    public GMLVQOptParam hesseDiag(GMLVQOptParam params, int exampleNdx, GMLVQOptParam out) {
        // (dj''-dk'')/(dj+dk) - 2(dj'^2-dk'^2)/(dk+dj)^2 + (dj-dk)(2(dj'-dk')^2/(dj+dk)^3 - (dj''+dk'')/(dj+dk)^2)
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[][] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);
//        double dSum = dj + dk;

//        double tmp = 4 / (dSum * dSum * dSum);
        // TODO Check signs.
//        double tmpJ = tmp * dk * dSum, tmpJ_ = tmp * dk * -2;
//        double tmpK = -tmp * dj * dSum, tmpK_ = tmp * dj * 2;
        int dims = params.dimensions(), numWeights = params.numWeights();

        //TODO: This can probably be sped up by not using the secondDerivLVQ calls...
        double[] dj_pjP = new double[dims], dj_pjPP = new double[dims];
        double[] dk_pkP = new double[dims], dk_pkPP = new double[dims];
        for (int i = 0; i < numWeights; i++) {
            double compJ = 0, compK = 0;
            double[] wRow = weights[i], owRow = oweights[i];
            for (int o = 0; o < dims; o++) {
                compJ += wRow[o]*(pj[o] - data[o]);
                compK += wRow[o]*(pk[o] - data[o]);
            }
            for (int o = 0; o < dims; o++) {
                double diffJ = pj[o] - data[o], diffK = pk[o] - data[o];
                double w = wRow[0];
                double dj_wP = 2*compJ*diffJ, dj_wPP = 2*diffJ*diffJ;
                double dk_wP = 2*compK*diffK, dk_wPP = 2*diffK*diffK;
                owRow[o] += secondDerivLVQ(dj, dk, dj_wP, dk_wP, dj_wPP, dk_wPP);
                dj_pjP[o] += compJ*w;
                dk_pkP[o] += compK*w;
                dj_pjPP[o] += w*w;
                dk_pkPP[o] += w*w;
            }
        }
        for (int o = 0; o < dims; o++) {
            opj[o] += secondDerivLVQ(dj, dk, 2*dj_pjP[o], 0, 2*dj_pjPP[o], 0);
            opk[o] += secondDerivLVQ(dj, dk, 0, 2*dk_pkP[o], 0, 2*dk_pkPP[o]);
        }

        return out;
    }

    public void check(double a, double b, String str) {
//    	System.out.printf("??? %f %f %s %f\n", a, b, str, Math.abs((a-b) / Math.min(a,b)));
        if (Math.abs((a-b) / Math.min(a,b)) > 1e-4 && Math.abs(Math.max(a, b)) > 1e-10) {
            System.out.println("SOMETHING WRONG!!! "+str+" "+a+" "+b);
            //throw new IllegalStateException();
        } else {
            //System.out.println("Here");
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
