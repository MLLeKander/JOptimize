// Lots of redundant code to ensure premature optimization.

package nl.rug.joptimize.learn.glvq;

import java.util.Arrays;
import java.util.Random;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.AbstractOptParam;

public class GLVQOptParam extends AbstractOptParam<GLVQOptParam> {
    public double[][] prototypes;
    public int[] labels;
    public static final Random rand = new Random();


    public GLVQOptParam(double[][] prototypes, int[] labels) {
        this.prototypes = prototypes;
        this.labels = labels;
    }

    public GLVQOptParam(int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, dimensions);
    }

    public GLVQOptParam(LabeledDataSet ds) {
        this(ds.averageProtos(), ds.labels());
    }

    public GLVQOptParam(int prototypesPerClass, int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, prototypesPerClass);
        init(ppc, dimensions);
    }

    public GLVQOptParam(int prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.classes(), ds.dimensions());
    }

    public GLVQOptParam(int[] prototypesPerClass, int dimensions) {
        init(prototypesPerClass, dimensions);
    }

    public GLVQOptParam(int[] prototypesPerClass, LabeledDataSet ds) {
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
            double currDist = dist(i, data);//, minDist);
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

            double currDist = dist(i, data);//, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestCorrectProtoNdx(double[] data, int label, int exNdx) {
        //System.err.println(Arrays.toString(data));
        //System.err.println(Arrays.toString(this.prototypes[0]));
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label) {
                continue;
            }

            double currDist = dist(i, data);
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

            double currDist = dist(i, data);//, minDist);
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

            double currDist = dist(i, data);
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
    public double get(int ndx) {
        return getFlatIndex(prototypes, ndx);
    }

    @Override
    public void set(int ndx, double value) {
        setFlatIndex(prototypes, ndx, value);
    }

    @Override
    public int length() {
        return prototypes.length*prototypes[0].length;
    }

    @Override
    public GLVQOptParam add_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam sub_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] -= o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam zero_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 0d);
        }
        return this;
    }

    @Override
    public GLVQOptParam one_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 1d);
        }
        return this;
    }

    @Override
    public GLVQOptParam dotprod_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam sqrt_s() {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = Math.sqrt(this.prototypes[i][j]);
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam inv_s() {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = 1 / this.prototypes[i][j];
            }
        }
        return this;
    }
    
    @Override
    public GLVQOptParam abs_s() {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] < 0) {
                    this.prototypes[i][j] *= -1;
                }
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam lbound_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] < o) {
                    this.prototypes[i][j] = o;
                }
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam ubound_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                if (this.prototypes[i][j] > o) {
                    this.prototypes[i][j] = o;
                }
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam add_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o;
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam multiply_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o;
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam random_s(Random r) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = r.nextDouble();
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam copy() {
        double[][] newProtos = this.newProtos();
        int dims = this.dimensions();
        for (int i = 0; i < newProtos.length; i++) {
            System.arraycopy(this.prototypes[i], 0, newProtos[i], 0, dims);
        }
        return new GLVQOptParam(newProtos, this.labels);
    }

    @Override
    public double squaredNorm() {
        double out = 0;
        for (double[] row : this.prototypes) {
            for (double d : row) {
                out += d*d;
            }
        }
        return out;
    }

    private double[][] newProtos() {
        return new double[this.numProtos()][this.dimensions()];
    }

    public double dist(int ndx, double[] data) {
        double[] proto = this.prototypes[ndx];
        assert (proto.length == data.length);
        double out = 0;
        for (int i = 0; i < data.length; i++) {
            double tmp = proto[i] - data[i];
            out += tmp * tmp;
        }
        return out;
    }

    @Override
    public String toString() {
        return Arrays.deepToString(prototypes);
    }
}
