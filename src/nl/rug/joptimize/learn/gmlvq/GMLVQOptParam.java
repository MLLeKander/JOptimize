// Lots of redundant code to ensure premature optimization.

package nl.rug.joptimize.learn.gmlvq;

import java.util.Arrays;
import java.util.Random;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.AbstractOptParam;

public class GMLVQOptParam extends AbstractOptParam<GMLVQOptParam> {
    public double[][] prototypes;
    public double[][] weights;
    public int[] labels;
    public static final Random rand = new Random();

    public GMLVQOptParam(double[][] prototypes, double[][] weights, int[] labels) {
        this.prototypes = prototypes;
        this.weights = weights;
        this.labels = labels;
        assert (labels.length == prototypes.length);
        assert (prototypes[0].length == weights[0].length);
    }

    public GMLVQOptParam(double[][] prototypes, int[] labels) {
        this.prototypes = prototypes;
        this.labels = labels;
        int dims = prototypes[0].length;
        this.weights = new double[dims][dims];
        for (int i = 0; i < dims; i++) {
            this.weights[i][i] = 1;
        }
    }

    public GMLVQOptParam(double[][] prototypes, int classes) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, prototypes[0].length);
        this.prototypes = prototypes;
    }

    public GMLVQOptParam(int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, dimensions);
    }

    public GMLVQOptParam(LabeledDataSet ds) {
        this(ds.averageProtos(), ds.labels());
    }

    public GMLVQOptParam(int prototypesPerClass, int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, prototypesPerClass);
        init(ppc, dimensions);
    }

    public GMLVQOptParam(int prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.classes(), ds.dimensions());
    }

    public GMLVQOptParam(int[] prototypesPerClass, int dimensions) {
        init(prototypesPerClass, dimensions);
    }

    public GMLVQOptParam(int[] prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.dimensions());
        assert (ds.classes() == prototypesPerClass.length);
    }

    private void init(int[] ppc, int dimensions) {
        int protoCount = 0;
        for (int i : ppc) {
            protoCount += i;
        }

        this.prototypes = new double[protoCount][dimensions];
        for (double[] row : prototypes) {
            for (int i = 0; i < row.length; i++) {
                row[i] += (rand.nextDouble() - 0.5) / 100;
            }
        }

        this.labels = new int[protoCount];
        for (int i = 0, ndx = 0; i < ppc.length; i++) {
            for (int j = 0; j < ppc[i]; j++) {
                this.labels[ndx++] = i;
            }
        }

        this.weights = new double[dimensions][dimensions];
        for (int i = 0; i < dimensions; i++) {
            this.weights[i][i] = 1 / dimensions;
        }
    }
    
    public GMLVQOptParam normalizeWeights() {
        normalize(weights);
        return this;
    }
    
    public GMLVQOptParam normalizeProtos() {
        normalize(prototypes);
        return this;
    }
    
    public static void normalize(double[][] m) {
        double norm = 0;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                norm += m[i][j] * m[i][j];
            }
        }
        norm = Math.sqrt(norm);
        if (norm == 0) { // Necessary if gradients are zero.
            return;
        }
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] /= norm;
            }
        }
    }

    public void setProto(int ndx, double[] data) {
        assert (data.length == this.prototypes[ndx].length);
        this.prototypes[ndx] = data;
    }

    public int getClosestProtoNdx(double[] data) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestProtoLabel(double[] data) {
        return this.labels[this.getClosestProtoNdx(data)];
    }

    public double[] getClosestProto(double[] data) {
        return this.prototypes[this.getClosestProtoNdx(data)];
    }

    public int getClosestCorrectProtoNdx(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label) {
                continue;
            }

            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestCorrectProtoNdx(double[] data, int label, int exNdx) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label) {
                continue;
            }

            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public double[] getClosestCorrectProto(double[] data, int label) {
        return this.prototypes[this.getClosestCorrectProtoNdx(data, label)];
    }

    public int getClosestIncorrectProtoNdx(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] == label) {
                continue;
            }

            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestIncorrectProtoNdx(double[] data, int label, int exNdx) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] == label) {
                continue;
            }

            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public double[] getClosestIncorrectProto(double[] data, int label) {
        return this.prototypes[this.getClosestIncorrectProtoNdx(data, label)];
    }

    public int getLabel(int ndx) {
        return this.labels[ndx];
    }

    public int numProtos() {
        return this.prototypes.length;
    }

    public int dimensions() {
        return this.prototypes[0].length;
    }

    public int numWeights() {
        return this.weights.length;
    }
    
    @Override
    public double get(int ndx) {
        int protSize = this.numProtos()*this.dimensions();
        if (ndx < protSize) {
            return getFlatIndex(prototypes, ndx);
        } else {
            return getFlatIndex(weights, ndx-protSize);
        }
    }

    @Override
    public void set(int ndx, double value) {
        int protSize = prototypes.length*prototypes[0].length;
        if (ndx < protSize) {
            setFlatIndex(prototypes, ndx, value);
        } else {
            setFlatIndex(weights, ndx-protSize, value);
        }
    }

    @Override
    public int length() {
        return prototypes.length*prototypes[0].length+weights.length*weights[0].length;
    }

    @Override
    public GMLVQOptParam add_s(GMLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o.prototypes[i][j];
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] += o.weights[i][j];
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam sub_s(GMLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] -= o.prototypes[i][j];
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] -= o.weights[i][j];
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam zero_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 0d);
        }
        for (double[] weight : this.weights) {
            Arrays.fill(weight, 0d);
        }
        return this;
    }

    @Override
    public GMLVQOptParam one_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 1d);
        }
        for (double[] weight : this.weights) {
            Arrays.fill(weight, 1d);
        }
        return this;
    }

    @Override
    public GMLVQOptParam dotprod_s(GMLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o.prototypes[i][j];
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] *= o.weights[i][j];
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam sqrt_s() {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = Math.sqrt(this.prototypes[i][j]);
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] = Math.sqrt(this.weights[i][j]);
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam inv_s() {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = 1 / this.prototypes[i][j];
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] = 1 / this.weights[i][j];
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam abs_s() {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] < 0) {
                    this.prototypes[i][j] *= -1;
                }
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.weights[i][j] < 0) {
                    this.weights[i][j] *= -1;
                }
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam lbound_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] < o) {
                    this.prototypes[i][j] = o;
                }
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.weights[i][j] < o) {
                    this.weights[i][j] = o;
                }
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam ubound_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] > o) {
                    this.prototypes[i][j] = o;
                }
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.weights[i][j] > o) {
                    this.weights[i][j] = o;
                }
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam add_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o;
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] += o;
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam multiply_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o;
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] *= o;
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam random_s(Random r) {
        int protos = this.numProtos(), dims = this.dimensions(), weights = this.numWeights();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = r.nextDouble();
            }
        }
        for (int i = 0; i < weights; i++) {
            for (int j = 0; j < dims; j++) {
                this.weights[i][j] = r.nextDouble();
            }
        }
        return this;
    }

    @Override
    public GMLVQOptParam copy() {
        double[][] newProtos = this.newProtos();
        double[][] newWeights = newWeights();
        int dims = this.dimensions();
        for (int i = 0; i < newProtos.length; i++) {
            System.arraycopy(this.prototypes[i], 0, newProtos[i], 0, dims);
        }
        for (int i = 0; i < newWeights.length; i++) {
            System.arraycopy(this.weights[i], 0, newWeights[i], 0, dims);
        }
        return new GMLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public double squaredNorm() {
        double out = 0;
        for (double[] row : this.prototypes) {
            for (double d : row) {
                out += d * d;
            }
        }
        for (double[] row : this.weights) {
            for (double d : row) {
                out += d * d;
            }
        }
        return out;
    }

    private double[][] newProtos() {
        return new double[this.numProtos()][this.dimensions()];
    }

    private double[][] newWeights() {
        return new double[this.numWeights()][this.dimensions()];
    }

    public double dist(int ndx, double[] data) {
        double[] proto = this.prototypes[ndx];
        double out = 0;
        for (double[] wRow : this.weights) {
            double tmp = 0;
            for (int i = 0; i < data.length; i++) {
                tmp += wRow[i] * (proto[i] - data[i]);
            }
            out += tmp * tmp;
        }
        // double diff = out-distO(ndx,data);
        // if (diff > 1e-8) {
        // System.err.println("PROBLEM!!! "+diff);
        // }
        return out;
    }

//    private double distO(int ndx, double[] data) {
//        int dims = this.dimensions();
//        double[] diff = new double[dims];
//        double[] proto = this.prototypes[ndx];
//        for (int i = 0; i < dims; i++) {
//            diff[i] = proto[i] - data[i];
//        }
//        double[] tmp = new double[dims];
//        double[][] omega = this.omega();
//        for (int i = 0; i < dims; i++) {
//            for (int j = 0; j < dims; j++) {
//                tmp[i] += diff[j] * omega[i][j];
//            }
//        }
//        double out = 0;
//        for (int i = 0; i < dims; i++) {
//            out += tmp[i] * diff[i];
//        }
//        return out;
//    }

    private double dist(int ndx, double[] data, double minDist) {
        double[] proto = this.prototypes[ndx];
        double out = 0;
        for (double[] wRow : this.weights) {
            double tmp = 0;
            for (int i = 0; i < data.length; i++) {
                tmp += wRow[i] * (proto[i] - data[i]);
            }
            out += tmp * tmp;
            if (out > minDist) {
                break;
            }
        }
        return out;
    }

    public double rowMult(int a, int b) {
        int w = this.numWeights();
        double out = 0;
        for (int i = 0; i < w; i++) {
            out += this.weights[i][a] * this.weights[i][b];
        }
        return out;
    }

    public double[][] omega() {
        int dims = this.dimensions();
        double[][] out = new double[dims][dims];
        for (int i = 0; i < dims; i++) {
            for (int j = 0; j < dims; j++) {
                out[i][j] = rowMult(i, j);
            }
        }

        return out;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
//        sb.append(prototypes.length+","+prototypes[0].length+"\n");
        String rowSep = "";
        for (double[] row : prototypes) {
            sb.append(rowSep);
            rowSep = "\n";
            String sep = "";
            for (double val : row) {
                sb.append(sep);
                sb.append(String.format("%.4f",val));
                sep = "  ";
            }
        }

        sb.append("\n\n");
//        sb.append(weights.length+","+weights[0].length+"\n");
        rowSep = "";
        for (double[] row : weights) {
            sb.append(rowSep);
            rowSep = "\n";
            String sep = "";
            for (double val : row) {
                sb.append(sep);
                sb.append(String.format("%.4f",val));
                sep = "  ";
            }
        }
        return sb.toString();
    }
}
