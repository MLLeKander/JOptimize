// Lots of redundant code to ensure premature optimization.

package nl.rug.joptimize.learn.grlvq;

import java.util.Arrays;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.OptParam;

public class GRLVQOptParam implements OptParam<GRLVQOptParam> {
    double[][] prototypes;
    double[] weights;
    int[] labels;

    public GRLVQOptParam(double[][] prototypes, double[] weights, int[] labels) {
        this.prototypes = prototypes;
        this.labels = labels;
    }

    public GRLVQOptParam(double[][] prototypes, int[] labels) {
        this.prototypes = prototypes;
        this.labels = labels;
    }

    public GRLVQOptParam(int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, dimensions);
    }

    public GRLVQOptParam(LabeledDataSet ds) {
        this(ds.classes(), ds.dimensions());
    }

    public GRLVQOptParam(int prototypesPerClass, int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, prototypesPerClass);
        init(ppc, dimensions);
    }

    public GRLVQOptParam(int prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.classes(), ds.dimensions());
    }

    public GRLVQOptParam(int[] prototypesPerClass, int dimensions) {
        init(prototypesPerClass, dimensions);
    }

    public GRLVQOptParam(int[] prototypesPerClass, LabeledDataSet ds) {
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
                row[i] += (Math.random()-0.5)/100;
            }
        }
        
        this.labels = new int[protoCount];
        for (int i = 0, ndx = 0; i < ppc.length; i++) {
            for (int j = 0; j < ppc[i]; j++) {
                this.labels[ndx++] = i;
            }
        }
        
        this.weights = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            this.weights[i] = 1;
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

    @Override
    public GRLVQOptParam add(GRLVQOptParam o) {
        double[][] newProtos = newProtos();
        double[] newWeights = newWeights();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] + o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            newWeights[i] = this.weights[i] + o.weights[i];
        }
        return new GRLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public GRLVQOptParam add_s(GRLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            this.weights[i] += o.weights[i];
        }
        return this;
    }

    @Override
    public GRLVQOptParam sub(GRLVQOptParam o) {
        double[][] newProtos = newProtos();
        double[] newWeights = newWeights();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] - o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            newWeights[i] = this.weights[i] - o.weights[i];
        }
        return new GRLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public GRLVQOptParam sub_s(GRLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] -= o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            this.weights[i] += o.weights[i];
        }
        return this;
    }

    @Override
    public GRLVQOptParam zero() {
        return new GRLVQOptParam(newProtos(), newWeights(), this.labels);
    }

    @Override
    public GRLVQOptParam zero_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 0d);
        }
        Arrays.fill(this.weights, 0d);
        return this;
    }

    @Override
    public GRLVQOptParam dotprod(GRLVQOptParam o) {
        double[][] newProtos = newProtos();
        double[] newWeights = newWeights();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            newWeights[i] = this.weights[i] * o.weights[i];
        }
        return new GRLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public GRLVQOptParam dotprod_s(GRLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o.prototypes[i][j];
            }
        }
        for (int i = 0; i < dims; i++) {
            this.weights[i] *= o.weights[i];
        }
        return this;
    }

    @Override
    public GRLVQOptParam multiply(double o) {
        double[][] newProtos = newProtos();
        double[] newWeights = newWeights();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * o;
            }
        }
        for (int i = 0; i < dims; i++) {
            newWeights[i] = this.weights[i] * o;
        }
        return new GRLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public GRLVQOptParam multiply_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o;
            }
        }
        for (int i = 0; i < dims; i++) {
            this.weights[i] += o;
        }
        return this;
    }

    @Override
    public GRLVQOptParam copy() {
        double[][] newProtos = this.newProtos();
        double[] newWeights = newWeights();
        int dims = this.dimensions();
        for (int i = 0; i < newProtos.length; i++) {
            System.arraycopy(this.prototypes[i], 0, newProtos[i], 0, dims);
        }
        System.arraycopy(this.weights, 0, newWeights, 0, dims);
        return new GRLVQOptParam(newProtos, newWeights, this.labels);
    }

    @Override
    public double squaredNorm() {
        double out = 0;
        for (double[] row : this.prototypes) {
            for (double d : row) {
                out += d*d;
            }
        }
        for (double d : this.weights) {
            out += d*d;
        }
        return out;
    }

    private double[][] newProtos() {
        return new double[this.numProtos()][this.dimensions()];
    }

    private double[] newWeights() {
        return new double[this.dimensions()];
    }

    public double dist(int ndx, double[] data) {
        double[] proto = this.prototypes[ndx];
        assert (proto.length == data.length);
        double out = 0;
        for (int i = 0; i < data.length; i++) {
            double tmp = proto[i] - data[i];
            out += this.weights[i] * tmp * tmp;
        }
        return out;
    }

    private double dist(int ndx, double[] data, double minDist) {
        double[] proto = this.prototypes[ndx];
        double out = 0;
        for (int i = 0; i < data.length; i++) {
            double tmp = proto[i] - data[i];
            out += this.weights[i] * tmp * tmp;
            if (out > minDist) {
                break;
            }
        }
        return out;
    }

    public String toString() {
        return Arrays.deepToString(prototypes);
    }
}
